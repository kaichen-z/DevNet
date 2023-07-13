#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

class ResNetMultiImageInput(models.ResNet):
    def __init__(self, block, layers, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, pretrained_path=None):
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    if pretrained:
        loaded = torch.load(pretrained_path)
        loaded['conv1.weight'] = torch.cat([loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1, pretrained_path=None):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}
        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))
        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, pretrained_path)
        else:
            self.encoder = resnets[num_layers]()
            if pretrained_path is not None:
                checkpoint = torch.load(pretrained_path)
                self.encoder.load_state_dict(checkpoint)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        print('DepthEncoder------------', pretrained_path)
    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        conv1_out = self.encoder.relu(x)
        block1_out = self.encoder.layer1(self.encoder.maxpool(conv1_out))
        block2_out = self.encoder.layer2(block1_out)
        block3_out = self.encoder.layer3(block2_out)
        block4_out = self.encoder.layer4(block3_out)
        return conv1_out, block1_out, block2_out, block3_out, block4_out
