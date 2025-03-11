import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # self.dropout = nn.Dropout(0.2) # dropout

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out = self.dropout(out) # dropout
        out += self.shortcut(x)
        out = F.relu(out)
        return out


##########################################################
# A "SmallResNet" Architecture
##########################################################
class SmallResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2],
                 num_classes=10, base_channels=32):
        """
        Args:
            block:          Residual block type (BasicBlock).
            num_blocks:     List of 4 integers, # of blocks in each layer.
            num_classes:    Number of output classes (10 for CIFAR-10).
            base_channels:  # of channels in first layer. Adjust to control total params.
        """
        super(SmallResNet, self).__init__()
        self.in_planes = base_channels

        # Initial conv layer
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # Create 4 stages (layers)
        self.layer1 = self._make_layer(block, base_channels,   num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8, num_blocks[3], stride=2)

        # Add dropout before linear layer
        self.dropout = nn.Dropout(0.2)

        # Global average pooling and linear layer
        self.linear = nn.Linear(base_channels*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global Average Pooling
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def SmallResNet0():
    return SmallResNet(base_channels=24, num_blocks=[3, 4, 6, 3])

def SmallResNet1():
    return SmallResNet(base_channels=28, num_blocks=[3, 4, 6, 3])