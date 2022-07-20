# DevNet
This is offical codes for the methods described in
> **DevNet: Self-supervised Monocular Depth Learning via Density Volume Construction**

## Setup

### Requirements:
The environment for reimplementing our code is described in environment.yml. You can set up and activate required conda environment by 
```bash
conda env create -f environment.yml
conda activate devnet
```

## Datasets Used In Our Paper
### Kitti Depth Estimation:
Downloading description is already provided in [monodepth2](https://github.com/nianticlabs/monodepth2).

### Kitti Odometry Estimation:
Kitti Odometry Dataset is provided in their website with detailed instruction [odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### NYU Depth Dataset V2:
NYU Depth Dataset V2 is provided in their website with detailed instruction [NYUV2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).

### Pretrain Resnets:
We downloaded [Pretrained Models](https://pytorch.org/vision/stable/models.html) in /pretrain.

## Training
We provide training codes for the unsupervised monocular depth estimation task. You can train it from scratch by running
```bash
sh train16.sh
```
