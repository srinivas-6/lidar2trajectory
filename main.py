import os
import argparse
import json
import logging
import time
import wandb
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

from semantickitti import SemanticKITTIDataset
from utils import pose_err, visualize_bev_image, lidar_transforms

from model import BEVTrajNet
from models.pose_loss import CameraPoseLoss

def main(args):
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    general_params = config['general']
    model_params = config['model']
    data_params = config['data']
    config = {**model_params, **general_params, **data_params}
    # Initialize wandb
    wandb.init(project='trajectory-prediction', entity=args.entity, config=config)
    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)
    # Create the model
    model = BEVTrajNet(config)
    model.to(device)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint, map_location=device_id))
        logging.info("Initializing from checkpoint: {}".format(args.load_checkpoint))
    if args.mode == 'train':
        # Set to train mode
        model.train()
        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)
        # Set the loss function
        pose_loss = CameraPoseLoss(config).to(device)
        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                  lr=config.get('lr'),
                                  eps=config.get('eps'),
                                  weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))
        print("Optimizer and scheduler initialized.")
        dataset = SemanticKITTIDataset(args.data_path, lookahead=args.lookahead, transforms=lidar_transforms,
                                        split_ratios=config.get('split_ratios'),
                                        mode=args.mode)
        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = DataLoader(dataset, **loader_params, drop_last=True)
        print("DataLoader initialized.")
        # Training loop
        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")
        # Train
        if args.checkpoint_path:
            checkpoint_prefix = os.path.join(args.checkpoint_path, "checkpoint")
        else:
            checkpoint_prefix = os.path.join(args.data_path, "checkpoint")

        n_total_samples = 0.0
        loss_vals = []
        sample_count = []
        for epoch in range(n_epochs):
            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0
            for batch_idx, batch_sample in enumerate((dataloader)):
                lidar_bev_tensor, target = batch_sample
                lidar_bev_tensor = lidar_bev_tensor.to(device)
                target = target.to(device)
                batch_size = lidar_bev_tensor.shape[0]
                # Zero the parameter gradients
                optim.zero_grad()
                # Forward pass
                output = model(lidar_bev_tensor)
                est_pose = output.get('pose')
                # Compute the loss
                criterion = pose_loss(est_pose, target)
                # Backward pass
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)
                # Back prop
                criterion.backward()
                optim.step()
                wandb.log({'lr': optim.param_groups[0]["lr"]})
                pose_error, orientation_error = pose_err(est_pose, target)
                n_samples += batch_size
                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0: 
                    print("[Batch-{}/Epoch-{}] running loss: {:.3f},"
                           "pose error: {:.2f}[m], {:.2f}[deg]".format(batch_idx+1, epoch+1, running_loss / n_samples,
                            pose_error.mean().item(), orientation_error.mean().item()))
                    wandb.log({'train_loss': running_loss / n_samples})
                    wandb.log({'pose error': pose_error.mean().item()})
                    wandb.log({'orient_error': orientation_error.mean().item()})
                    running_loss = 0.0
                    n_samples = 0
             # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))
            scheduler.step()

        print('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth')
        # Save the loss values to a CSV file
        plt.figure(figsize=(10, 5))
        plt.plot(loss_vals)
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True)
        plt.savefig(checkpoint_prefix + '_loss_curve.png')
        plt.show()

    else:
        # Set to eval mode
        model.eval()
        loader_params = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': config.get('n_workers')}
        dataset =  SemanticKITTIDataset(args.data_path, lookahead=args.lookahead, transforms=lidar_transforms,
                                        split_ratios=config.get('split_ratios'),
                                        mode=args.mode)
        dataloader = DataLoader(dataset, **loader_params, drop_last=False)
        print("DataLoader initialized.")
        # Evaluation loop 
        preds = []
        targets = []
        stats = np.zeros((len(dataloader.dataset), 3))
        with torch.no_grad():
            for batch_idx, batch_sample in enumerate((dataloader)):
                lidar_bev_tensor, target = batch_sample
                lidar_bev_tensor = lidar_bev_tensor.to(device)
                target = target.to(device)
                tic = time.time()
                output = model(lidar_bev_tensor)
                est_pose = output.get('pose')
                toc = time.time()
                est_pose = est_pose.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                posit_err, orient_err = pose_err(torch.from_numpy(est_pose), torch.from_numpy(target))
                preds.append(est_pose)
                targets.append(target)
                stats[batch_idx, 0] = posit_err.item()
                stats[batch_idx, 1] = orient_err.item()
                stats[batch_idx, 2] = (toc - tic)*1000
                print(("Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                    stats[batch_idx, 0],  stats[batch_idx, 1],  stats[batch_idx, 2])))
            wandb.log({'Median pose error [m]': np.nanmedian(stats[:, 0])})
            wandb.log({'Median orient error [deg]': np.nanmedian(stats[:, 1])})
            wandb.log({'inference time': np.mean(stats[:, 2])})

        # write the predictions to a txt file
        pred_file = os.path.join(args.data_path, "predictions.txt")
        with open(pred_file, 'w') as f:
            for pred in preds:
                f.write(f"{pred}\n")
                f.write("\n")
        print(f"Predictions saved to {pred_file}")
        # write the targets to a txt file
        target_file = os.path.join(args.data_path, "targets.txt")
        with open(target_file, 'w') as f:
            for target in targets:
                f.write(f"{target}\n")
                f.write("\n")
        print(f"Targets saved to {target_file}")
                # Record overall statistics
        print("Performance of {}".format(args.load_checkpoint))
        print("Median pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(stats[:, 0]), np.nanmedian(stats[:, 1])))
        print("Mean inference time:{:.2f}[ms]".format(np.mean(stats[:, 2])))
        print(" Mean pose error: {:.3f}[m], {:.3f}[deg]".format(
                    np.nanmean(stats[:, 0]),  np.nanmean(stats[:, 1])))
        

                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the config file')
    parser.add_argument('--lookahead', type=int, default=1, help='Lookahead parameter')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--visualize', action='store_true', help='Flag to visualize BEV')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint file')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train', help='Mode to run the script in')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to the checkpoint file to load')
    parser.add_argument('--entity', type=str, default=None, help='WandB entity name')
    args = parser.parse_args()
    main(args)