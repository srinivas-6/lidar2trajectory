# LiDAR2Trajectory: Predict the future trajectory of a ego vehicle using BEV LiDAR

## Description 

This project focuses on learning future ego-vehicle pose by leveraging sequences of LiDAR point clouds captured at multiple time steps. Using the SemanticKITTI dataset, LiDAR data is encoded into bird's-eye view (BEV) representation and fed into a neural network. The model is trained to predict the vehicle's relative future displacement and heading change (yaw) after a fixed lookahead interval.

To improve rotational learning stability and mitigate issues with angle periodicity, the yaw angle is represented as a unit quaternion ([cos(yaw/2), sin(yaw/2)]). This enables the model to learn smooth and consistent representations of orientation.

Key components of the project include:

* BEV encoding with height, density, and intensity channels
* Learning relative transformation between poses using SE(2) translation and quaternion rotation
* A Transformer-based model where two transformers separately attend to position- and orientation- informative features from a convolutional backbone which takes the LiDAR BEV image as input.


## Repository Overview 
This code implements

* Training of a Transformer-based architecture for relative pose regression, where the relative pose is computed based on the current pose and the lookahead pose (3 secs)
* The LiDAR data is encoded with height, density, and intensity channels as a BEV image with a size of (3, 256,256)
* The model is trained with LiDAR BEV image as input and relative poses as targets
* Testing of the models
* Torch-based dataset pipeline and training setup

## Model Architecture

![image](https://github.com/user-attachments/assets/d389e104-086e-4d01-9b12-604b6ffccaff)

The architecture of the model is adapted to learn from LiDAR BEV images. The original model is from [Multi-Scene Camera Pose Regression with Transformers](https://github.com/yolish/multi-scene-pose-transformer/tree/main)

## Useage
  ```
  python main.py -h
  ```

  ```
  python main.py --data_path <--PATH-TO-DATASET-->  --config_file ./config.json --mode train --checkpoint_path ./checkpoints
  ```
  ```
  python main.py --data_path <--PATH-TO-DATASET-->  --config_file ./config.json --mode test --load_checkpoint <--PATH-TO-MODEL-CHECKPOINT>
  ```


## Results 

### 🚀 Experiment Summary
A complete run of Training and Test experiments of the project can be found [here](https://wandb.ai/ravuri/trajectory-prediction)
### Ablation Studies 

| Input      | Lookahead prediction | Model Checkpoint       | Weights & Biases Run        | Median Translation Error (m) | Median Orientation Error (deg) |
|------------------|--------|-------------------------|------------------------------|------------------------|--------------------------|
| LiDAR BEV      | t+3 sec     | [checkpoint_final_t+3.pth](https://drive.google.com/file/d/1AeXCR1ehoYeTKQPjb-EUnZY5KyqzUjWL/view?usp=sharing) | [wandb/run-train](https://wandb.ai/ravuri/trajectory-prediction/runs/bdvjoxuw?nw=nwuserravuri) [wandb/run-test](https://wandb.ai/ravuri/trajectory-prediction/runs/yka7ak4l?nw=nwuserravuri) | 7.50                | 4.73                   |
| LiDAR BEV         | t+1 sec     | [checkpoint_final_t+1.pth](https://drive.google.com/file/d/1u2fav3xtP3D-kV3d7oWDYlK-StANP_FO/view?usp=sharing) | [wandb/run-train](https://wandb.ai/ravuri/trajectory-prediction/runs/t3vas6y6?nw=nwuserravuri) [wandb/run-test](https://wandb.ai/ravuri/trajectory-prediction/runs/yjzgr3si?nw=nwuserravuri)   | 2.0                  | 1.1                    |
| LiDAR BEV                 | t sec    | [checkpoint_final_t.pth ](https://drive.google.com/file/d/1gCfgXcvmW7Yc0BPjNiN6QZXG1oQY-oXe/view?usp=sharing)       | [wandb/run-train](https://wandb.ai/ravuri/trajectory-prediction/runs/tt9mjdu4?nw=nwuserravuri) [wandb/run-test](https://wandb.ai/ravuri/trajectory-prediction/runs/7x49qygr?nw=nwuserravuri)    | 0.331                   | 0.621         |
| Temporal LiDAR BEV    | TBD     | TBD | TBD | TBD                | TBD                    |

### TODO
* Currently experimenting with multi-frame LiDAR BEV (t-1, t, t+1) temporal sequences as input
* A deeper backbone like EfficientNetB3

### References
* [Multi-Scene Camera Pose Regression with Transformers](https://github.com/yolish/multi-scene-pose-transformer/tree/main)
* [PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kendall_PoseNet_A_Convolutional_ICCV_2015_paper.pdf)
* [RobustLoc: Robust Visual Localization in Changing Conditions](https://github.com/sijieaaa/RobustLoc)
