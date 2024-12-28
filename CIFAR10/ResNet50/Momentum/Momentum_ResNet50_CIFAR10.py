import torch
import torch.nn as nn
import copy
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
import numpy as np

import math
import sys
import time
import os
import csv


lr           = 0.01
start_epoch  = 1
num_epochs   = 200
batch_size   = 128

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_valid  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset   = torchvision.datasets.CIFAR10(root='./data', transform=transform_train, train=True, download=True)
train_dataset = torch.utils.data.Subset(train_dataset, range(45000))
print(len(train_dataset))
valid_dataset   = torchvision.datasets.CIFAR10(root='./data', transform=transform_valid, train=True, download=True)
valid_dataset = torch.utils.data.Subset(valid_dataset, range(45000,50000))
print(len(valid_dataset))
test_dataset    = torchvision.datasets.CIFAR10(root='./data', transform=transform_test, train=False, download=True)
train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
valid_loader    = torch.utils.data.DataLoader(valid_dataset, batch_size=80, num_workers=2, shuffle=False)
test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=80, num_workers=2, shuffle=False)


def sish(input):
    return input * torch.tanh(torch.exp(input))

def smish(input):
    return input * torch.tanh(torch.log( 1+torch.sigmoid(input) ))

def logish(input):
    return input * torch.log( 1+torch.sigmoid(input) )


class BasicBlock_Sish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Sish, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = sish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = sish(out)
        return out

class Bottleneck_Sish(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_Sish, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = sish(self.bn1(self.conv1(x)))
        out = sish(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = sish(out)
        return out

class ResNet_Sish(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Sish, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = sish(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_Sish():
    return ResNet_Sish(Bottleneck_Sish, [3, 4, 6, 3])

def conv_init_Sish(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
SishNet = ResNet18_Sish()
SishNet.apply(conv_init_Sish)
if is_use_cuda:
    SishNet.to(device)
    SishNet = nn.DataParallel(SishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class BasicBlock_GELU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_GELU, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class Bottleneck_GELU(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_GELU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

class ResNet_GELU(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_GELU, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_GELU():
    return ResNet_GELU(Bottleneck_GELU, [3, 4, 6, 3])

def conv_init_GELU(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
GELUNet = ResNet18_GELU()
GELUNet.apply(conv_init_GELU)
if is_use_cuda:
    GELUNet.to(device)
    GELUNet = nn.DataParallel(GELUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class BasicBlock_Smish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Smish, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = smish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = smish(out)
        return out

class Bottleneck_Smish(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_Smish, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = smish(self.bn1(self.conv1(x)))
        out = smish(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = smish(out)
        return out

class ResNet_Smish(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Smish, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = smish(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_Smish():
    return ResNet_Smish(Bottleneck_Smish, [3, 4, 6, 3])

def conv_init_Smish(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
SmishNet = ResNet18_Smish()
SmishNet.apply(conv_init_Smish)
if is_use_cuda:
    SmishNet.to(device)
    SmishNet = nn.DataParallel(SmishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class BasicBlock_Logish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Logish, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = logish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = logish(out)
        return out

class Bottleneck_Logish(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_Logish, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = logish(self.bn1(self.conv1(x)))
        out = logish(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = logish(out)
        return out

class ResNet_Logish(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Logish, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = logish(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_Logish():
    return ResNet_Logish(Bottleneck_Logish, [3, 4, 6, 3])

def conv_init_Logish(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
LogishNet = ResNet18_Logish()
LogishNet.apply(conv_init_Logish)
if is_use_cuda:
    LogishNet.to(device)
    LogishNet = nn.DataParallel(LogishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class BasicBlock_ReLU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_ReLU, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

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
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_ReLU(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_ReLU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet_ReLU(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_ReLU, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_ReLU():
    return ResNet_ReLU(Bottleneck_ReLU, [3, 4, 6, 3])

def conv_init_ReLU(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
ReLUNet = ResNet18_ReLU()
ReLUNet.apply(conv_init_ReLU)
if is_use_cuda:
    ReLUNet.to(device)
    ReLUNet = nn.DataParallel(ReLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class BasicBlock_Mish(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Mish, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.mish(out)
        return out

class Bottleneck_Mish(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_Mish, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.mish(self.bn1(self.conv1(x)))
        out = F.mish(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.mish(out)
        return out

class ResNet_Mish(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_Mish, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.mish(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_Mish():
    return ResNet_Mish(Bottleneck_Mish, [3, 4, 6, 3])

def conv_init_Mish(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
MishNet = ResNet18_Mish()
MishNet.apply(conv_init_Mish)
if is_use_cuda:
    MishNet.to(device)
    MishNet = nn.DataParallel(MishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class BasicBlock_SiLU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_SiLU, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out

class Bottleneck_SiLU(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_SiLU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = F.silu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.silu(out)
        return out

class ResNet_SiLU(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_SiLU, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18_SiLU():
    return ResNet_SiLU(Bottleneck_SiLU, [3, 4, 6, 3])

def conv_init_SiLU(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        init.kaiming_uniform_(m.weight)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
       
SiLUNet = ResNet18_SiLU()
SiLUNet.apply(conv_init_SiLU)
if is_use_cuda:
    SiLUNet.to(device)
    SiLUNet = nn.DataParallel(SiLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
        
    return lr * math.pow(0.5, optim_factor)

def train(epoch, activation):
    #torch.autograd.set_detect_anomaly(True)
    if activation == "Sish":
        net = SishNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "Smish":
        net = SmishNet
    elif activation == "Logish":
        net = LogishNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "SiLU":
        net = SiLUNet
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.9, weight_decay=1e-3)#was1e-3
    
    # print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs        = net(inputs)
        loss           = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total      += labels.size(0)
        #correct    += predict.eq(labels).cpu().sum().double()
        correct += (predict == labels).cpu().sum().item()
        
        # sys.stdout.write('\r')
        # sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
        #                 % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        #                    epoch, num_epochs, idx, len(train_dataset) // batch_size, 
        #                   train_loss / (batch_size * (idx + 1)), correct / total))
        # sys.stdout.flush()
    return (100 * correct / total), train_loss

def validate(epoch, activation):
    if activation == "Sish":
        net = SishNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "Smish":
        net = SmishNet
    elif activation == "Logish":
        net = LogishNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "SiLU":
        net = SiLUNet
    
    net.eval()
    valid_loss = 0
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(valid_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs        = net(inputs)
        loss           = criterion(outputs, labels)
        
        valid_loss  += loss.item()
        _, predict = torch.max(outputs, 1)
        total      += labels.size(0)
        #correct    += predict.eq(labels).cpu().sum().double()
        correct += (predict == labels).cpu().sum().item()
        
        # sys.stdout.write('\r')
        # sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
        #                 % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        #                    epoch, num_epochs, idx, len(valid_dataset) // 80, 
        #                   valid_loss / (100 * (idx + 1)), correct / total))
        # sys.stdout.flush()
    return (100 * correct / total), valid_loss

def test(epoch, activation):
    if activation == "Sish":
        net = bestSishNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "Smish":
        net = bestSmishNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "SiLU":
        net = bestSiLUNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

SishTrainAccuracies = []
SishValidAccuracies = []
SishTrainRuntimes = []
SishValidRuntimes = []
SishTrainLoss = []
SishValidLoss = []
SishTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Sish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Sish")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestSishNet = copy.deepcopy(SishNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    SishTrainAccuracies.append(train_accuracy)
    SishValidAccuracies.append(valid_accuracy)
    SishTrainRuntimes.append(train_time)
    SishValidRuntimes.append(valid_time)
    SishTrainLoss.append(train_loss)
    SishValidLoss.append(valid_loss)
SishTestAccuracy.append(test(_epoch, "Sish"))
print("test_acc:", SishTestAccuracy)
print("Finished Sish")

GELUTrainAccuracies = []
GELUValidAccuracies = []
GELUTrainRuntimes = []
GELUValidRuntimes = []
GELUTrainLoss = []
GELUValidLoss = []
GELUTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"GELU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "GELU")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestGELUNet = copy.deepcopy(GELUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    GELUTrainAccuracies.append(train_accuracy)
    GELUValidAccuracies.append(valid_accuracy)
    GELUTrainRuntimes.append(train_time)
    GELUValidRuntimes.append(valid_time)
    GELUTrainLoss.append(train_loss)
    GELUValidLoss.append(valid_loss)
GELUTestAccuracy.append(test(_epoch, "GELU"))
print("test_acc:", GELUTestAccuracy)
print("Finished GELU")

SmishTrainAccuracies = []
SmishValidAccuracies = []
SmishTrainRuntimes = []
SmishValidRuntimes = []
SmishTrainLoss = []
SmishValidLoss = []
SmishTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Smish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Smish")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestSmishNet = copy.deepcopy(SmishNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    SmishTrainAccuracies.append(train_accuracy)
    SmishValidAccuracies.append(valid_accuracy)
    SmishTrainRuntimes.append(train_time)
    SmishValidRuntimes.append(valid_time)
    SmishTrainLoss.append(train_loss)
    SmishValidLoss.append(valid_loss)
SmishTestAccuracy.append(test(_epoch, "Smish"))
print("test_acc:", SmishTestAccuracy)
print("Finished Smish")

LogishTrainAccuracies = []
LogishValidAccuracies = []
LogishTrainRuntimes = []
LogishValidRuntimes = []
LogishTrainLoss = []
LogishValidLoss = []
LogishTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Logish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Logish")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestLogishNet = copy.deepcopy(LogishNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    LogishTrainAccuracies.append(train_accuracy)
    LogishValidAccuracies.append(valid_accuracy)
    LogishTrainRuntimes.append(train_time)
    LogishValidRuntimes.append(valid_time)
    LogishTrainLoss.append(train_loss)
    LogishValidLoss.append(valid_loss)
LogishTestAccuracy.append(test(_epoch, "Logish"))
print("test_acc:", LogishTestAccuracy)
print("Finished Logish")

ReLUTrainAccuracies = []
ReLUValidAccuracies = []
ReLUTrainRuntimes = []
ReLUValidRuntimes = []
ReLUTrainLoss = []
ReLUValidLoss = []
ReLUTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"ReLU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "ReLU")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestReLUNet = copy.deepcopy(ReLUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    ReLUTrainAccuracies.append(train_accuracy)
    ReLUValidAccuracies.append(valid_accuracy)
    ReLUTrainRuntimes.append(train_time)
    ReLUValidRuntimes.append(valid_time)
    ReLUTrainLoss.append(train_loss)
    ReLUValidLoss.append(valid_loss)
ReLUTestAccuracy.append(test(_epoch, "ReLU"))
print("test_acc:", ReLUTestAccuracy)
print("Finished ReLU")

MishTrainAccuracies = []
MishValidAccuracies = []
MishTrainRuntimes = []
MishValidRuntimes = []
MishTrainLoss = []
MishValidLoss = []
MishTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Mish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Mish")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestMishNet = copy.deepcopy(MishNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    MishTrainAccuracies.append(train_accuracy)
    MishValidAccuracies.append(valid_accuracy)
    MishTrainRuntimes.append(train_time)
    MishValidRuntimes.append(valid_time)
    MishTrainLoss.append(train_loss)
    MishValidLoss.append(valid_loss)
MishTestAccuracy.append(test(_epoch, "Mish"))
print("test_acc:", MishTestAccuracy)
print("Finished Mish")

SiLUTrainAccuracies = []
SiLUValidAccuracies = []
SiLUTrainRuntimes = []
SiLUValidRuntimes = []
SiLUTrainLoss = []
SiLUValidLoss = []
SiLUTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"SiLU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "SiLU")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestSiLUNet = copy.deepcopy(SiLUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    SiLUTrainAccuracies.append(train_accuracy)
    SiLUValidAccuracies.append(valid_accuracy)
    SiLUTrainRuntimes.append(train_time)
    SiLUValidRuntimes.append(valid_time)
    SiLUTrainLoss.append(train_loss)
    SiLUValidLoss.append(valid_loss)
SiLUTestAccuracy.append(test(_epoch, "SiLU"))
print("test_acc:", SiLUTestAccuracy)
print("Finished SiLU")

with open('ResNet_CIFAR10_Trial7.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator="\n")
    wr.writerow(SishTrainAccuracies)
    wr.writerow(GELUTrainAccuracies)
    wr.writerow(SmishTrainAccuracies)
    wr.writerow(LogishTrainAccuracies)
    wr.writerow(MishTrainAccuracies)
    wr.writerow(SiLUTrainAccuracies)
    wr.writerow(ReLUTrainAccuracies)

    wr.writerow(SishValidAccuracies)
    wr.writerow(GELUValidAccuracies)
    wr.writerow(SmishValidAccuracies)
    wr.writerow(LogishValidAccuracies)
    wr.writerow(MishValidAccuracies)
    wr.writerow(SiLUValidAccuracies)
    wr.writerow(ReLUValidAccuracies)

    wr.writerow(SishTrainLoss)
    wr.writerow(GELUTrainLoss)
    wr.writerow(SmishTrainLoss)
    wr.writerow(LogishTrainLoss)
    wr.writerow(MishTrainLoss)
    wr.writerow(SiLUTrainLoss)
    wr.writerow(ReLUTrainLoss)

    wr.writerow(SishValidLoss)
    wr.writerow(GELUValidLoss)
    wr.writerow(SmishValidLoss)
    wr.writerow(LogishValidLoss)
    wr.writerow(MishValidLoss)
    wr.writerow(SiLUValidLoss)
    wr.writerow(ReLUValidLoss)

    wr.writerow(SishTrainRuntimes)
    wr.writerow(GELUTrainRuntimes)
    wr.writerow(SmishTrainRuntimes)
    wr.writerow(LogishTrainRuntimes)
    wr.writerow(MishTrainRuntimes)
    wr.writerow(SiLUTrainRuntimes)
    wr.writerow(ReLUTrainRuntimes)

    wr.writerow(SishValidRuntimes)
    wr.writerow(GELUValidRuntimes)
    wr.writerow(SmishValidRuntimes)
    wr.writerow(LogishValidRuntimes)
    wr.writerow(MishValidRuntimes)
    wr.writerow(SiLUValidRuntimes)
    wr.writerow(ReLUValidRuntimes)

    wr.writerow(SishTestAccuracy)
    wr.writerow(GELUTestAccuracy)
    wr.writerow(SmishTestAccuracy)
    wr.writerow(LogishTestAccuracy)
    wr.writerow(ReLUTestAccuracy)
    wr.writerow(MishTestAccuracy)
    wr.writerow(SiLUTestAccuracy)

print("SqueezeNet CIFAR10 trial complete")
