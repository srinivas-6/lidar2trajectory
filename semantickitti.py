import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class SemanticKITTIDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        lookahead: int = 30,
        num_past_frames: int = 1,
        transforms=None,
        mode: str = "train",  # "train", or "test"
        split_ratios: Tuple[float, float] = (0.8, 0.2),
    ):
        self.data_path = Path(data_path)
        self.lookahead = lookahead
        self.transforms = transforms.get('lidar')
        self.poses = self._load_poses()
        self.lidar_files = sorted(list((self.data_path / "velodyne").glob("*.bin")))
        self.mean , self.std = None, None
        self.num_past_frames = num_past_frames
        # Ensure split ratios sum to 1.0
        assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
        total_files = len(self.lidar_files)
        train_end = int(total_files * split_ratios[0])
        if mode == "train":
            self.lidar_files = self.lidar_files[:train_end]
            self.poses = self.poses[:train_end]
            self.mean, self.std = self.compute_translation_stats(self.poses, lookahead=self.lookahead)
        elif mode == "test":
            self.lidar_files = self.lidar_files[train_end:]
            self.poses = self.poses[train_end:]
            print('......... Loading pose stats from file ..........')
            stats = np.load(self.data_path / "pose_stats.npz")
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)
            print(f'Loaded mean: {self.mean}, std: {self.std}')
        else:
            raise ValueError("Invalid mode. Must be 'train' or 'test'.")

    def _load_poses(self) -> np.ndarray:
        calib = self._parse_calibration(self.data_path / "calib.txt")
        poses = self._parse_poses(self.data_path / "poses.txt", calib)
        return poses

    @staticmethod
    def _parse_calibration(filename: Path) -> Dict[str, np.ndarray]:
        calib = {}
        for line in filename.read_text().splitlines():
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose
        return calib

    @staticmethod
    def _parse_poses(filename: Path, calibration: Dict[str, np.ndarray]) -> np.ndarray:
        poses = []
        cab_tr = calibration["Tr"]
        tr_inv = np.linalg.inv(cab_tr)
        for line in filename.read_text().splitlines():
            values = [float(v) for v in line.strip().split()]
            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            pose = tr_inv @ pose @ cab_tr
            poses.append(pose.astype(np.float32))
        return np.array(poses, dtype=np.float32)

    @staticmethod
    def _load_lidar(lidar_file: Path) -> np.ndarray:
        scan = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))
        return scan
    
    @staticmethod
    def yaw_to_quaternion(yaw):
        q_w = np.cos(yaw / 2)
        q_z = np.sin(yaw / 2)
        return np.array([q_w, 0, 0, q_z], dtype=np.float32)  # (qw, qx, qy, qz)
    
    def compute_translation_stats(self, poses: np.ndarray, lookahead: int = 30):
        translations = []
        for i in tqdm(range(len(poses) - lookahead)):
            current_pose = poses[i]
            target_pose = poses[i + lookahead]
            relative = np.linalg.inv(current_pose) @ target_pose
            translation = relative[:3, 3]
            translations.append(translation)
        translations = np.stack(translations)
        mean = translations.mean(axis=0)
        std = translations.std(axis=0)
        print(f"[RESULT] Mean: {mean}, Std: {std}")
        if not (self.data_path / "pose_stats.npz").exists():
            np.savez(self.data_path / "pose_stats.npz", mean=mean, std=std)
            print(f"[SAVED] pose_stats.npz to {self.data_path}")
        return mean, std
    
    def __len__(self) -> int:
        return len(self.lidar_files) - self.lookahead
    
    def encode_lidar_to_bev_image(self,
        lidar: np.ndarray,
        x_range=(-32.0, 32.0),
        y_range=(-32.0, 32.0),
        resolution=0.25,
        max_height=3.0,
        max_density=64.0
    ) -> np.ndarray:
        """
        Converts raw LiDAR point cloud to BEV image for ResNet input.

        Args:
            lidar (np.ndarray): Raw LiDAR point cloud (N, 4) with [x, y, z, intensity].
            x_range (tuple): Min/max X range in meters.
            y_range (tuple): Min/max Y range in meters.
            resolution (float): Meters per pixel.
            max_height (float): Max Z value to clip the height channel.
            max_density (float): Max point count per pixel (for normalization).

        Returns:
            bev_image (np.ndarray): BEV image with shape (H, W, 3) — float32.
        """
        # Define grid size
        x_min, x_max = x_range
        y_min, y_max = y_range
        W = int((x_max - x_min) / resolution)
        H = int((y_max - y_min) / resolution)

        # Create empty BEV image with 3 channels: height, density, intensity
        bev = np.zeros((H, W, 3), dtype=np.float32)

        # Filter points within the defined range
        mask = (
            (lidar[:, 0] >= x_min) & (lidar[:, 0] < x_max) &
            (lidar[:, 1] >= y_min) & (lidar[:, 1] < y_max)
        )
        lidar = lidar[mask]

        # Convert coordinates to BEV grid indices, clipping prevents any out-of-bounds access
        x_indices = np.clip(((lidar[:, 0] - x_min) / resolution).astype(np.int32), 0, W - 1)
        y_indices = np.clip(((lidar[:, 1] - y_min) / resolution).astype(np.int32), 0, H - 1)

        # Convert to image coordinate system (origin at bottom-left)
        y_indices = H - 1 - y_indices

        for i in range(len(lidar)):
            x_idx = x_indices[i]
            y_idx = y_indices[i]

            z = lidar[i, 2]
            intensity = lidar[i, 3]

            # Channel 0: Height (normalized)
            bev[y_idx, x_idx, 0] = max(bev[y_idx, x_idx, 0], min(z / max_height, 1.0))

            # Channel 1: Density (count-based, capped)
            bev[y_idx, x_idx, 1] = min(bev[y_idx, x_idx, 1] + 1.0 / max_density, 1.0)

            # Channel 2: Intensity (max per pixel)
            bev[y_idx, x_idx, 2] = max(bev[y_idx, x_idx, 2], min(intensity, 1.0))

        return bev  # Shape: (H, W, 3)

    def lidar_to_histogram_features(self, lidar):
        """
        Convert LiDAR point cloud into a 3-bin histogram over a 512x512 grid with intensity as the third channel.

        Args:
            lidar: Numpy array of shape (N, 4), where each point is [x, y, z, intensity].

        Returns:
            features: Numpy array of shape (512, 512, 3), where the channels are:
                    - Channel 0: Above-ground histogram
                    - Channel 1: Below-ground histogram
                    - Channel 2: Intensity values
        """
        def splat_points(point_cloud, weights=None):
            # 256 x 256 grid
            grid_size = 256
            x_meters_max = 50
            x_meters_min = -50
            y_meters_max = 50
            y_meters_min = -50
            xbins = np.linspace(x_meters_min, x_meters_max, grid_size + 1)
            ybins = np.linspace(y_meters_min, y_meters_max, grid_size + 1)
            hist, _, _ = np.histogram2d(
                point_cloud[:, 0], point_cloud[:, 1],
                bins=(xbins, ybins),
                weights=weights
            )
            return hist

        # Separate points into below-ground and above-ground
        below = lidar[lidar[..., 2] <= 1.0]
        above = lidar[lidar[..., 2] > 1.0]

        # Compute histograms for below-ground and above-ground points
        below_features = splat_points(below)
        above_features = splat_points(above)

        # Compute intensity map
        intensity_map = splat_points(lidar, weights=lidar[:, 3])

        # Normalize intensity map to [0, 1]
        if np.max(intensity_map) > 0:
            intensity_map /= np.max(intensity_map)

        # Stack the features into a 3D array
        features = np.stack([above_features, below_features, intensity_map], axis=-1).astype(np.float32)

        return features
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # load lidar data
        lidar_data = self._load_lidar(self.lidar_files[idx])
        lidar_bev_tensor = self.encode_lidar_to_bev_image(lidar_data)
        lidar_bev_tensor = torch.from_numpy(lidar_bev_tensor).permute(2, 0, 1).contiguous()
        # apply transforms if any
        if self.transforms:
            lidar_bev_tensor = self.transforms(lidar_bev_tensor)
        # load pose data
        current_pose = self.poses[idx]
        target_pose = self.poses[idx + self.lookahead]
        # get relative pose from current to target 
        relative_pose = np.linalg.inv(current_pose) @ target_pose
        # Extract relative translation
        delta_xyz = np.array([relative_pose[0, 3],
                              relative_pose[1, 3],
                                relative_pose[2, 3]
                              ], dtype=np.float32)
        # Normalize translation
        delta_xyz_norm = (delta_xyz - self.mean) / self.std
        # Extract yaw angle
        # yaw is the rotation around the z-axis
        delta_yaw = np.arctan2(relative_pose[1, 0], relative_pose[0, 0])
        quaternion = self.yaw_to_quaternion(delta_yaw)
        #ensure unit norm 
        quaternion /= np.linalg.norm(quaternion)
        # Final target: normalized [x, y] + quaternion
        target = torch.tensor([*delta_xyz_norm, *quaternion], dtype=torch.float32)
        return lidar_bev_tensor.float(), target