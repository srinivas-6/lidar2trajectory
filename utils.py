import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

# Augmentations
lidar_transforms = {
    'lidar': transforms.Compose([transforms.Normalize(mean=0.5, std=1)]),
}


def visualize_bev_image(bev_tensor: torch.Tensor, title: str = "BEV Image"):
    """
    Visualizes a BEV image tensor (3, H, W) as an RGB image with origin marker.
    
    Args:
        bev_tensor (torch.Tensor): Tensor of shape (3, H, W)
        title (str): Title for the plot
    """
    assert bev_tensor.shape[0] == 3, "Expected 3 channels (height, above, intensity)"
    bev_np = bev_tensor.numpy()
    
    # Normalize all channels to [0, 1] for display
    bev_np = np.clip(bev_np, 0, 1)

    # Transpose to (H, W, 3) for visualization
    bev_rgb = np.transpose(bev_np, (1, 2, 0))

    H, W, _ = bev_rgb.shape
    origin_x, origin_y = W // 2, H // 2  # center pixel as origin

    # Overlay origin marker in red
    bev_rgb[origin_y-2:origin_y+3, origin_x-2:origin_x+3] = [1, 0, 0]  # RGB red

    plt.figure(figsize=(6, 6))
    plt.imshow(bev_rgb)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()



def visualize_lidar_histogram(lidar_features: np.ndarray, title: str = "LiDAR Histogram"):
    """
    Visualizes the LiDAR point cloud features (above, below ground, and intensity) with a white background.

    Args:
        lidar_features (np.ndarray): LiDAR features, shape (H, W, 3).
        title (str): Title for the plot.
    """
    assert lidar_features.shape[2] == 3, "LiDAR features must have shape (H, W, 3)"
    
    # Create a figure with a white background
    plt.figure(figsize=(8, 8))
    # plt.imshow(np.zeros_like(lidar_features[..., 0]), cmap='gray', vmin=0, vmax=1)  # White background

    # Overlay the three channels (above-ground, below-ground, intensity)
    plt.imshow(lidar_features)

    # Add title and labels
    plt.title(title)
    plt.xlabel('X-meters')
    plt.ylabel('Y-meters')
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.axis('equal')
    plt.grid(False)  # Disable grid for a cleaner look
    plt.show()


def pose_err(est_pose, gt_pose, mean, std):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx3, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx3, N is the batch size)
    :param mean: (float) mean value for denormalization
    :param std: (float) standard deviation for denormalization
    :return: position error(s) and orientation errors(s)
    """
    est_pose[:, 0:3] = est_pose[:, 0:3] * std + mean
    gt_pose[:, 0:3] = gt_pose[:, 0:3] * std + mean
    est_pose = torch.from_numpy(est_pose).float()
    gt_pose = torch.from_numpy(gt_pose).float()
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1, p=2)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err