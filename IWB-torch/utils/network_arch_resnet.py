from __future__ import absolute_import, division, print_function
import torch
import torchvision.models as models
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, isPretrained=False, isGrayscale=False, embDimension=128, poolSize=4):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = '/tmp/models'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained
        self.embDimension = embDimension
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            51: models.resnext50_32x4d,
            101: models.resnet101,
            152: models.resnet152}

        resnets_pretrained_path = {
            18: 'resnet18-5c106cde.pth', 
            34: 'resnet34.pth',
            50: 'resnet50-19c8e357.pth',
            101: 'resnet101.pth',
            152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()
        
        if self.isPretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(
                torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
            
        if self.isGrayscale:
            if num_layers == 34:
                self.encoder.conv1 = nn.Conv2d(
                    1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.encoder.conv1 = nn.Conv2d(
                    1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            if num_layers == 34:
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            else:
                self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] = 2048
        else:
            self.num_ch_enc[1:] = 512
                    
        if self.embDimension > 0:
            self.encoder.fc = nn.Linear(self.num_ch_enc[-1], self.embDimension)


    def forward(self, input_image):
        self.features = []
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)

        x = self.encoder.layer1(x)
        self.features.append(x)

        x = self.encoder.layer2(x)
        self.features.append(x)

        x = self.encoder.layer3(x) 
        self.features.append(x)

        x = self.encoder.layer4(x)
        self.features.append(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        self.x = x.view(x.size(0), -1)

        y = self.x
        x = self.encoder.fc(self.x)

        return x, y
    
