# [DevNet: Self-supervised Monocular Depth Learning via Density Volume Construction](https://arxiv.org/abs/2209.06351) - ECCV 2022.

## Introduction 

This is the PyTorch implementation of **DevNet: Self-supervised Monocular Depth Learning via Density Volume Construction**, a simple and efficient neural architecture for Self-supervised Monocular Depth Estimation.

## Setup
DevNet provides support for multiple versions of Python and Torch, such as:
```
python==3.8 
pytorch==1.12.0
torchvision==0.13.0
```

## Data
[KITTI]: To download this dataset, you can follow instruction of [MonoDepth2](https://github.com/nianticlabs/monodepth2)

[KITTI Odometry]: To download this dataset, you can follow instruction of [MonoDepth2](https://github.com/nianticlabs/monodepth2)

[NYU-V2]: To download this dataset, you can follow instruction of [MonoDepth2](https://github.com/nianticlabs/monodepth2)

## Running the code
### Inference
The trained models for partial experiments presented in the paper could be found in `logs`. The inference script is `test_dev.py`.
- DevNet on `kitti` with `resnet18` backbone and `192 x 640` resolution: 
```
sh start2test.sh
```
### Training
Necessary training details could also be found in `test_dev.py`.



## Citation
If you find this code useful for your research, please cite our paper

```
@inproceedings{zhou2022devnet,
  title={Devnet: Self-supervised monocular depth learning via density volume construction},
  author={Zhou, Kaichen and Hong, Lanqing and Chen, Changhao and Xu, Hang and Ye, Chaoqiang and Hu, Qingyong and Li, Zhenguo},
  booktitle={European Conference on Computer Vision},
  pages={125--142},
  year={2022},
  organization={Springer}
}
```
## Acknowledgements
Our code partially builds on [Monodepth2](https://github.com/nianticlabs/monodepth2).

