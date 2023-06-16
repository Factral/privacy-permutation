import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

import utils


class ClassifierDeformable(nn.Module):
    def __init__(self, initial_perm, device):
        super(ClassifierDeformable, self).__init__()

        self.conv1 = DeformConv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.offset_1, perm_1 = utils.calculate_offset(33, 1, 0, 3, initial_perm)

        self.conv2 = DeformConv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.offset_2, perm_2 = utils.calculate_offset(self.offset_1.shape[-1], 1, 0, 3, perm_1)

        self.conv3 = DeformConv2d(32, 16, kernel_size=5, stride=1, padding=0)
        self.offset_3, perm_3 = utils.calculate_offset(self.offset_2.shape[-1], 1, 0, 5, perm_2)

        self.conv4 = DeformConv2d(16, 16, kernel_size=7, stride=1, padding=0)
        self.offset_4, perm_4 = utils.calculate_offset(self.offset_3.shape[-1], 1, 0, 7, perm_3)

        self.conv5 = DeformConv2d(16, 8, kernel_size=5, stride=1, padding=0)
        self.offset_5, perm_5 = utils.calculate_offset(self.offset_4.shape[-1], 1, 0, 5, perm_4)

        self.conv6 = DeformConv2d(8, 4, kernel_size=3, stride=1, padding=0)
        self.offset_6, self.perm_6 = utils.calculate_offset(self.offset_5.shape[-1], 1, 0, 3, perm_5)

        # send offsets to device
        self.offset_1 = self.offset_1.to(device)
        self.offset_2 = self.offset_2.to(device)
        self.offset_3 = self.offset_3.to(device)
        self.offset_4 = self.offset_4.to(device)
        self.offset_5 = self.offset_5.to(device)
        self.offset_6 = self.offset_6.to(device)

        self.perceptron  = nn.Sequential(
            nn.Linear(13 * 13 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  
        )


    def forward(self, x):
        x = torch.relu(self.conv1(x, self.offset_1))
        x = torch.relu(self.conv2(x, self.offset_2))
        x = torch.relu(self.conv3(x, self.offset_3))
        x = torch.relu(self.conv4(x, self.offset_4))
        x = torch.relu(self.conv5(x, self.offset_5))
        x = torch.relu(self.conv6(x, self.offset_6))

        x = self.perm_6.ordenar(x.squeeze(0))

        x = x.flatten(start_dim=1)
        x = self.perceptron(x)

        return x
