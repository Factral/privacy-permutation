import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

import sys

sys.path.append('..')

import utils.utils as utils

class VGG_permuted(nn.Module):
    """
    PyTorch implementation of deformed VGG.
    """
    def __init__(self,initial_perm):
        super().__init__()
    
        # conv1
        conv1 =  nn.Conv2d(3, 64, 3, padding=1)
        self.offset_1, perm_1 = utils.calculate_offset(32, 1, 1, 3, initial_perm)
        conv2 =  nn.Conv2d(64, 64, 3, padding=1)
        self.offset_2, perm_2 = utils.calculate_offset(self.offset_1.shape[-1], 1, 1, 3, perm_1)

        # conv2
        conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.offset_3, perm_3 = utils.calculate_offset(self.offset_2.shape[-1], 1, 1, 3, perm_2)
        conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.offset_4, perm_4 = utils.calculate_offset(self.offset_3.shape[-1], 1, 1, 3, perm_3)

        # conv3
        conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.offset_5, perm_5 = utils.calculate_offset(self.offset_4.shape[-1], 1, 1, 3, perm_4)

        conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.offset_6, perm_6 = utils.calculate_offset(self.offset_5.shape[-1], 1, 1, 3, perm_5)

        conv7 = nn.Conv2d(256, 256, 3, padding=1)
        self.offset_7, perm_7 = utils.calculate_offset(self.offset_6.shape[-1], 1, 1, 3, perm_6)

        # conv4
        conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.offset_8, perm_8 = utils.calculate_offset(self.offset_7.shape[-1], 1, 1, 3, perm_7)
        conv9 = nn.Conv2d(512, 512, 3, padding=1)
        self.offset_9, perm_9 = utils.calculate_offset(self.offset_8.shape[-1], 1, 1, 3, perm_8)
        conv10 = nn.Conv2d(512, 512, 3, padding=1)
        self.offset_10, perm_10 = utils.calculate_offset(self.offset_9.shape[-1], 1, 1, 3, perm_9)

        # conv5
        conv11 = nn.Conv2d(512, 512, 3, padding=1)
        self.offset_11, perm_11 = utils.calculate_offset(self.offset_10.shape[-1], 1, 1, 3, perm_10)
        conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.offset_12, perm_12 = utils.calculate_offset(self.offset_11.shape[-1], 1, 1, 3, perm_11)
        conv13 = nn.Conv2d(512, 512, 3, padding=1)
        self.offset_13, self.perm_13 = utils.calculate_offset(self.offset_12.shape[-1], 1, 1, 3, perm_12)
        print(self.offset_13.shape)

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )
        
    def forward(self, x):


        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
        