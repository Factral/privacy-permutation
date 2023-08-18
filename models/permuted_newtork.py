from torchvision.ops import DeformConv2d
import torch
import torch.nn as nn


class PermutedNetwork(nn.Module):
    def __init__(self):
        super(PermutedNetwork, self).__init__()

        self.conv_offset_1 = nn.Conv2d(1,3*3*2, kernel_size=3, padding=0, bias=True)
        self.conv1 = DeformConv2d(1, 16, kernel_size=3, stride=1, padding=0)

        self.conv_offset_2 = nn.Conv2d(16,3*3*2, kernel_size=3, padding=0, bias=True)
        self.conv2 = DeformConv2d(16, 32, kernel_size=3, stride=1, padding=0)
        
        self.conv_offset_3 = nn.Conv2d(32,5*5*2, kernel_size=5, padding=0, bias=True)
        self.conv3 =DeformConv2d(32, 16, kernel_size=5, stride=1, padding=0)
        
        self.conv_offset_4 = nn.Conv2d(16,7*7*2, kernel_size=7, padding=0, bias=True)
        self.conv4 = DeformConv2d(16, 16, kernel_size=7, stride=1, padding=0)
        
        self.conv_offset_5 = nn.Conv2d(16,5*5*2, kernel_size=5, padding=0, bias=True)
        self.conv5 = DeformConv2d(16, 8, kernel_size=5, stride=1, padding=0)
        
        self.conv_offset_6 = nn.Conv2d(8,3*3*2, kernel_size=3, padding=0, bias=True)
        self.conv6 = DeformConv2d(8, 4, kernel_size=3, stride=1, padding=0)


        self.perceptron  = nn.Sequential(
            nn.Linear(13 * 13 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  
        )


    def forward(self, x):


        offset = self.conv_offset_1(x)
        x = torch.relu(self.conv1(x, offset))

        offset = self.conv_offset_2(x)
        x = torch.relu(self.conv2(x, offset))

        offset = self.conv_offset_3(x)
        x = self.conv3(x, offset)

        offset = self.conv_offset_4(x)
        x = torch.relu(self.conv4(x, offset))

        offset = self.conv_offset_5(x)
        x = torch.relu(self.conv5(x, offset))

        offset = self.conv_offset_6(x)
        x = torch.relu(self.conv6(x, offset))

        x = x.flatten(start_dim=1)
        x = self.perceptron(x)

        return x
    


class PermutedNetwork_2(nn.Module):
    def __init__(self):
        super(PermutedNetwork_2, self).__init__()

        self.conv_offset_1 = nn.Conv2d(1,3*3*2, kernel_size=3, padding=1, bias=True)
        self.conv1 = DeformConv2d(1, 16, kernel_size=3, stride=1, padding=1)

        self.conv_offset_2 = nn.Conv2d(16,3*3*2, kernel_size=3, padding=1, bias=True)
        self.conv2 = DeformConv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        self.conv_offset_3 = nn.Conv2d(32,3*3*2, kernel_size=3, padding=1, bias=True)
        self.conv3 = DeformConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        

        self.pool = nn.MaxPool2d(2, 2)  
        self.batchnorm = nn.BatchNorm2d(16)
        self.gap = nn.AdaptiveAvgPool2d( (3, 3) )

        self.fc = nn.Linear(16 * 3 * 3, 10)


    def forward(self, x):


        offset = self.conv_offset_1(x)
        x = torch.relu(self.conv1(x, offset)) # 33x33

        x = self.pool(x) # 15x15

        offset = self.conv_offset_2(x)
        x = torch.relu(self.conv2(x, offset)) # 14x14

        x = self.pool(x) # 7x7

        offset = self.conv_offset_3(x) 
        x = torch.relu( self.conv3(x, offset) )

        x = self.gap(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
        

class NormalNetwork(nn.Module):
    def __init__(self):
        super(NormalNetwork, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d( (5, 5) )
        ) 

        
        self.fc = nn.Linear(64 * 5 * 5, 10)


    def forward(self, x):

        x = self.encoder(x)

        x = x.flatten(start_dim=1)

        x = self.fc(x)

        return x
        



class NormalNetwork_latentspace(nn.Module):
    def __init__(self):
        super(NormalNetwork_latentspace, self).__init__()
        

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d( (5, 5) )
        )


    def forward(self, x):

        x = self.encoder(x)

        return x