# FLAT: Fooling LiDAR Perception via Adversarial Trajectory Perturbation [ICCV2021 Oral]

[Yiming Li*](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Congcong Wen*](https://scholar.google.com/citations?user=OTBgvCYAAAAJ), [Felix Juefei-Xu](https://scholar.google.com/citations?user=dgN8vtwAAAAJ), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ)

**Small perturbations to vehicle trajectory can blind LiDAR perception.**

<p align="center"><img src='docs/pics/FLAT.png' align="center" height="500px"> </p>

Poster Page: https://ai4ce.github.io/FLAT/

[**ArXiv: Fooling LiDAR Perception via Adversarial Trajectory Perturbation**](https://arxiv.org/abs/2103.15326)        



## News

[2021-07]  🔥 FLAT is accepted at ICCV 2021 as oral presentation (210/6236, 3% acceptance rate)

## Abstract
LiDAR point clouds collected from a moving vehicle are functions of its trajectories, because the sensor motion needs to be compensated to avoid distortions. When autonomous vehicles are sending LiDAR point clouds to deep networks for perception and planning, could the motion compensation consequently become a wide-open backdoor in those networks, due to both the adversarial vulnerability of deep learning and GPS-based vehicle trajectory estimation that is susceptible to wireless spoofing? We demonstrate such possibilities for the first time: instead of directly attacking point cloud coordinates which requires tampering with the raw LiDAR readings, only adversarial spoofing of a self-driving car's trajectory with small perturbations is enough to make safety-critical objects undetectable or detected with incorrect positions. Moreover, polynomial trajectory perturbation is developed to achieve a temporally-smooth and highly-imperceptible attack. Extensive experiments on 3D object detection have shown that such attacks not only lower the performance of the state-of-the-art detectors effectively, but also transfer to other detectors, raising a red flag for the community. 

## Installation
For white-box attacks, we use point-based [PointRCNN](https://github.com/sshaoshuai/PointRCNN) as the target detector.  
```point_rcnn.py``` ```rcnn_net.py``` ```rpn.py``` in ```PointRCNN/lib/net``` were modified for introducing attacks.   
```kitti_dataset.py``` ```kitti_rcnn_dataset.py```  in ```PointRCNN/lib/datasets``` were modified for loading our customized nusc_kitti dataset.   
  
The rest code of PointRCNN is left untouched.
### Requirements
* Linux (tested on Ubuntu 18.04)
* Python 3.6
* PyTorch 1.2.0
* CUDA 10.0



### Create Anaconda Environment
```bash
conda create -n flat python=3.6
conda activate flat
```

### CUDA
```bash
export PATH=/usr/local/cuda-10.0/bin:$PATH
export CUDA_PATH=/usr/local/cuda-10.0
export CUDA_HOME=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
```
### Install dependencies
```bash
git clone https://github.com/ai4ce/FLAT.git
cd FLAT
pip install -r requirements.txt
pip install torch==1.2.0 torchvision==0.4.0
cd PointRCNN
sh build_and_install.sh
cd ..
```

## Dataset Preparation
Please download the official [nuscenes dataset](https://www.nuscenes.org/nuscenes)(v1.0-trainval)

Use ```nusc_to_kitti.py``` to generate the dataset.

```bash
python nusc_to_kitti.py nuscenes_gt_to_kitti [--dataroot "Your nuscenes dataroot"]
```

It will generate the dataset in the structure as follows.
```
FLAT
├── dataset
│   ├── nusc_kitti
│   │   ├──val_1000
│   │   │   ├──image_2
│   │   │   ├──ImageSets
│   │   │   ├──label_2
│   │   │   ├──pose
│   │   │   ├──velodyne
```

**NOTICE**: This script converts the first 1000(of 6019 in total) samples from orginal validation split of v1.0-trainval at default. You can use all of the nuscenes samples, and shuffle option is also provided.

## Run FLAT on Evaluation
```bash
python flat.py [--stage STAGE] [--nb_iter NB_ITER]
               [--task TASK] [--attack_type ATTACK_TYPE] 
               [--iter_eps ITER_EPS] [--iter_eps2 ITER_EPS2] [--poly]
```

```
--split SPLIT       
                    The data split for evaluation
--stage STAGE       
                    Attack stage of Point RCNN. Options: "1" for RPN
                    stage, "2" for RCNN stage
--nb_iter NB_ITER   
                    Number of attack iterations in PGD
--task TASK         
                    Task of attacking. Options: "cls" for classification,
                    "reg" for regression
--attack_type ATTACK_TYPE
                    Specify attack type. Options: "all", "translation",
                    "rotation"
--iter_eps ITER_EPS 
                    Primary PGD attack step size for each iteration, in
                    translation only/rotation only attacks, this parameter
                    is used.
--iter_eps2 ITER_EPS2
                    Secondary PGD attack step size for each iteration,
                    only effective when attack_type is "all" and poly mode
                    is disabled.
--poly              
                    Polynomial trajectory perturbation option. Notice: if
                    true, attack_type will be fixed(translation)
```
All the experiments were performed at the [pretrained model](checkpoint_epoch_70.pth) of PointRCNN as provided.

Detection and evaluation results will be save in 
```bash
output/{SPLIT}/{ATTACK_TYPE}/FLAT_{STAGE}_{TASK}_{NB_ITER}_{ITER_EPS}_{ITER_EPS2}
```

## Acknowledgment  
```flat.py``` is modified from the evaluation code of [PointRCNN](https://github.com/sshaoshuai/PointRCNN), for implementing attacks.  
```evaluate.py``` is  borrowed from evaluation code from [Train in Germany, Test in The USA: Making 3D Object Detectors Generalize](https://github.com/cxy1997/3D_adapt_auto_driving), utilizing distance-based difficulty metrics.  
```nusc_to_kitti.py``` is  modified from official [nuscenes-devkit script](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py) to generate kitti-format nuscenes dataset with ego pose for interpolation.  
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN)
* [3D_adapt_auto_driving](https://github.com/cxy1997/3D_adapt_auto_driving)
* [nusSenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

This project is not possible without these great codebases.

## Citation
If you find FLAT useful in your research, please cite:
```
@InProceedings{Li_2021_ICCV,
      title = {Fooling LiDAR Perception via Adversarial Trajectory Perturbation},
      author = {Li, Yiming and Wen, Congcong and Juefei-Xu, Felix and Feng, Chen},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      month = {October},
      year = {2021}
}
```
