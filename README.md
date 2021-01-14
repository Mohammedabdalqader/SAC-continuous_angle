# continuous angle for learning synergies between pushing and grasping with Soft Actor Critic (SAC) algorithm 

Visual Pushing and Grasping (VPG) is a method for training robotic agents to learn how to plan complementary pushing and grasping actions for manipulation (*e.g.* for unstructured pick-and-place applications). VPG operates directly on visual observations (RGB-D images), learns from trial and error, trains quickly, and generalizes to new objects and scenarios.



This repository provides a modified Pytorch implementation of VPG  for training and testing VPG policies with continuous angles with Deep Reinforcement Learning in simulation. 


### Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning (research paper & Original implemetation)

[PDF](https://arxiv.org/pdf/1803.09956.pdf) | [Webpage & Video Results](http://vpg.cs.princeton.edu/)

[Andy Zeng](http://andyzeng.github.io/), [Shuran Song](http://vision.princeton.edu/people/shurans/), [Stefan Welker](https://www.linkedin.com/in/stefan-welker), [Johnny Lee](http://johnnylee.net/), [Alberto Rodriguez](http://meche.mit.edu/people/faculty/ALBERTOR@MIT.EDU), [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/)

IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2018

Skilled robotic manipulation benefits from complex synergies between non-prehensile (*e.g.* pushing) and prehensile (*e.g.* grasping) actions: pushing can help rearrange cluttered objects to make space for arms and fingers; likewise, grasping can help displace objects to make pushing movements more precise and collision-free. In this work, we demonstrate that it is possible to discover and learn these synergies from scratch through model-free deep reinforcement learning. Our method involves training two fully convolutional networks that map from visual observations to actions: one infers the utility of pushes for a dense pixel-wise sampling of end effector orientations and locations, while the other does the same for grasping. Both networks are trained jointly in a Q-learning framework and are entirely self-supervised by trial and error, where rewards are provided from successful grasps. In this way, our policy learns pushing motions that enable future grasps, while learning grasps that can leverage past pushes. During picking experiments in both simulation and real-world scenarios, we find that our system quickly learns complex behaviors amid challenging cases of clutter, and achieves better grasping success rates and picking efficiencies than baseline alternatives after only a few hours of training. We further demonstrate that our method is capable of generalizing to novel objects.

# The goal of this project

Extending previous algorithm to allow continuous angle


#### Demo in Simulation (pushing and grasping with continuous angle) - Test Environment

This demo runs our pre-trained model with a UR5 robot arm in simulation on challenging picking scenarios with adversarial clutter, where grasping an object is generally not feasible without first pushing to break up tight clusters of objects. 

<img src="images/continuous_angle.gif" height=358px width=600px align="center" />


## Installation

This implementation requires the following dependencies (tested on Ubuntu 18.04): 

* Python3
* Pytorch 1.4.0 und torchvision 0.5.0 :

    ```shell
        # CUDA 9.2
        conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=9.2 -c pytorch

        # CUDA 10.1
        conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```


* [V-REP](http://www.coppeliarobotics.com/) (now known as [CoppeliaSim](http://www.coppeliarobotics.com/)) simulation environment

### (Optional) GPU Acceleration
Accelerating training/inference with an NVIDIA GPU requires installing [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn). You may need to register with NVIDIA for the CUDA Developer Program (it's free) before downloading. This code has been tested with CUDA 10.1 on a single NVIDIA RTX 2070 (8GB). Running out-of-the-box with our pre-trained models using GPU acceleration requires 8GB of GPU memory. Running with GPU acceleration is **highly recommended**, otherwise each training iteration will take several minutes to run (as opposed to several seconds). This code automatically detects the GPU(s) on your system and tries to use it. If you have a GPU, but would instead like to run in CPU mode, add the tag `--cpu` when running `main.py` below.


### Instructions

1. Checkout this repository .

    ```shell
    git clone https://github.com/Mohammedabdalqader/SAC-continuous_angle.git
    cd SAC-continuous_angle
    ```

1. Run V-REP (navigate to your V-REP/CoppeliaSim directory and run `./vrep.sh` or `./coppeliaSim.sh`). From the main menu, select `File` > `Open scene...`, and open the file `SAC-continuous_angle/simulation/simulation.ttt` from this repository.

1. In another terminal window, run the following (simulation will start in the V-REP window). 

    ```shell
    python main.py --is_sim --obj_mesh_dir 'objects/blocks' --num_obj 10     --push_rewards --experience_replay --explore_rate_decay 
    --is_testing --test_preset_cases --test_preset_file    'simulation/  test-cases/test-10-obj-07.txt'     --load_snapshot --snapshot_folder 'model'     --save_visualizations
    ```

Note: you may get a popup window titled "Dynamics content" in your V-REP window. Select the checkbox and press OK. You will have to do this a total of 3 times before it stops annoying you.

## Training

To train a regular VPG policy from scratch in simulation, first start the simulation environment by running V-REP (navigate to your V-REP directory and run `./vrep.sh`). From the main menu, select `File` > `Open scene...`, and open the file `visual-pushing-grasping/simulation/simulation.ttt`. Then navigate to this repository in another terminal window and run the following:

    ```shell
    python main.py --is_sim --push_rewards --experience_replay --explore_rate_decay
    ```


