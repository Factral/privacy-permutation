import torch
import torch.nn as nn


class ClassifierStandard(nn.Module):
    def __init__(self, initial_perm):
        super(ClassifierStandard, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=7, stride=1, padding=0)
        self.conv5 = nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=0)
        self.conv6 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=0)

        self.perceptron  = nn.Sequential(
            nn.Linear(13 * 13 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  
        )


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))

        x = x.flatten(start_dim=1)
        x = self.perceptron(x)

        return x
