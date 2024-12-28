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


torch.manual_seed(0)
lr           = 0.05 #0.005
start_epoch  = 1
num_epochs   = 100
batch_size   = 256
width = 128

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

# Data Preprocess
transform_train = transforms.Compose([
    #transforms.RandomCrop(28, padding=2),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

transform_valid  = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test0  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.)
    #AddGaussianNoise(0.,1.)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test1  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.02)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test2  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.04)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test3  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.06)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test4  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.08)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test5  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.1)
    #AddGaussianNoise(0.,1.)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test6  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.12)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test7  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.14)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test8  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.16)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_test9  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.18)
    #transforms.Normalize((0.5,), (0.5,))
])

transform_testx  = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.,0.2)
    #transforms.Normalize((0.5,), (0.5,))
])

train_dataset   = torchvision.datasets.FashionMNIST(root='./data', transform=transform_train, train=True, download=True)
train_dataset = torch.utils.data.Subset(train_dataset, range(50000))
print(len(train_dataset))
valid_dataset   = torchvision.datasets.FashionMNIST(root='./data', transform=transform_valid, train=True, download=True)
valid_dataset = torch.utils.data.Subset(valid_dataset, range(50000,60000))
print(len(valid_dataset))

test_dataset0    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test0, train=False, download=True)
test_dataset1    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test1, train=False, download=True)
test_dataset2    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test2, train=False, download=True)
test_dataset3    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test3, train=False, download=True)
test_dataset4    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test4, train=False, download=True)
test_dataset5    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test5, train=False, download=True)
test_dataset6    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test6, train=False, download=True)
test_dataset7    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test7, train=False, download=True)
test_dataset8    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test8, train=False, download=True)
test_dataset9    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test9, train=False, download=True)
test_datasetx    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_testx, train=False, download=True)

train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader    = torch.utils.data.DataLoader(valid_dataset, batch_size=80, num_workers=1, shuffle=False)

test_loader0     = torch.utils.data.DataLoader(test_dataset0, batch_size=80, num_workers=1, shuffle=False)
test_loader1     = torch.utils.data.DataLoader(test_dataset1, batch_size=80, num_workers=1, shuffle=False)
test_loader2     = torch.utils.data.DataLoader(test_dataset2, batch_size=80, num_workers=1, shuffle=False)
test_loader3     = torch.utils.data.DataLoader(test_dataset3, batch_size=80, num_workers=1, shuffle=False)
test_loader4     = torch.utils.data.DataLoader(test_dataset4, batch_size=80, num_workers=1, shuffle=False)
test_loader5     = torch.utils.data.DataLoader(test_dataset5, batch_size=80, num_workers=1, shuffle=False)
test_loader6     = torch.utils.data.DataLoader(test_dataset6, batch_size=80, num_workers=1, shuffle=False)
test_loader7     = torch.utils.data.DataLoader(test_dataset7, batch_size=80, num_workers=1, shuffle=False)
test_loader8     = torch.utils.data.DataLoader(test_dataset8, batch_size=80, num_workers=1, shuffle=False)
test_loader9     = torch.utils.data.DataLoader(test_dataset9, batch_size=80, num_workers=1, shuffle=False)
test_loaderx     = torch.utils.data.DataLoader(test_datasetx, batch_size=80, num_workers=1, shuffle=False)


# def TeLU(input):
#     return input * torch.tanh(torch.exp(input))

def init_weights_bias(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class TeLU(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh( torch.exp(input) )

class SimpleMLP_TeLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            TeLU(),
            nn.Linear(width, width),#2
            TeLU(),     
            nn.Linear(width, width),#3
            TeLU(),  
            nn.Linear(width, width),#4
            TeLU(), 
            nn.Linear(width, width),#5
            TeLU(),
            nn.Linear(width, width),#6
            TeLU(),     
            nn.Linear(width, width),#7
            TeLU(),  
            nn.Linear(width, width),#8
            TeLU(), 
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
TeLUNet = SimpleMLP_TeLU()
TeLUNet.apply(init_weights_bias)
if is_use_cuda:
    TeLUNet.to(device)
    TeLUNet = nn.DataParallel(TeLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()



class SimpleMLP_Mish(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            nn.Mish(),
            nn.Linear(width, width),#2
            nn.Mish(),    
            nn.Linear(width, width),#3
            nn.Mish(),  
            nn.Linear(width, width),#4
            nn.Mish(), 
            nn.Linear(width, width),#5
            nn.Mish(),
            nn.Linear(width, width),#6
            nn.Mish(),     
            nn.Linear(width, width),#7
            nn.Mish(),  
            nn.Linear(width, width),#8
            nn.Mish(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
MishNet = SimpleMLP_Mish()
MishNet.apply(init_weights_bias)
if is_use_cuda:
    MishNet.to(device)
    MishNet = nn.DataParallel(MishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()
       


class SimpleMLP_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            nn.ReLU(),
            nn.Linear(width, width),#2
            nn.ReLU(),    
            nn.Linear(width, width),#3
            nn.ReLU(),  
            nn.Linear(width, width),#4
            nn.ReLU(), 
            nn.Linear(width, width),#5
            nn.ReLU(),
            nn.Linear(width, width),#6
            nn.ReLU(),     
            nn.Linear(width, width),#7
            nn.ReLU(),  
            nn.Linear(width, width),#8
            nn.ReLU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
ReLUNet = SimpleMLP_ReLU()
ReLUNet.apply(init_weights_bias)
if is_use_cuda:
    ReLUNet.to(device)
    ReLUNet = nn.DataParallel(ReLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class SimpleMLP_GELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            nn.GELU(),
            nn.Linear(width, width),#2
            nn.GELU(),    
            nn.Linear(width, width),#3
            nn.GELU(),  
            nn.Linear(width, width),#4
            nn.GELU(), 
            nn.Linear(width, width),#5
            nn.GELU(),
            nn.Linear(width, width),#6
            nn.GELU(),     
            nn.Linear(width, width),#7
            nn.GELU(),  
            nn.Linear(width, width),#8
            nn.GELU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
GELUNet = SimpleMLP_GELU()
GELUNet.apply(init_weights_bias)
if is_use_cuda:
    GELUNet.to(device)
    GELUNet = nn.DataParallel(GELUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class SimpleMLP_SiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            nn.SiLU(),
            nn.Linear(width, width),#2
            nn.SiLU(),    
            nn.Linear(width, width),#3
            nn.SiLU(),  
            nn.Linear(width, width),#4
            nn.SiLU(), 
            nn.Linear(width, width),#5
            nn.SiLU(),
            nn.Linear(width, width),#6
            nn.SiLU(),     
            nn.Linear(width, width),#7
            nn.SiLU(),  
            nn.Linear(width, width),#8
            nn.SiLU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
SiLUNet = SimpleMLP_SiLU()
SiLUNet.apply(init_weights_bias)
if is_use_cuda:
    SiLUNet.to(device)
    SiLUNet = nn.DataParallel(SiLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class SimpleMLP_ELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            nn.ELU(),
            nn.Linear(width, width),#2
            nn.ELU(),    
            nn.Linear(width, width),#3
            nn.ELU(),  
            nn.Linear(width, width),#4
            nn.ELU(), 
            nn.Linear(width, width),#5
            nn.ELU(),
            nn.Linear(width, width),#6
            nn.ELU(),     
            nn.Linear(width, width),#7
            nn.ELU(),  
            nn.Linear(width, width),#8
            nn.ELU(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
ELUNet = SimpleMLP_ELU()
ELUNet.apply(init_weights_bias)
if is_use_cuda:
    ELUNet.to(device)
    ELUNet = nn.DataParallel(ELUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class Logish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.log( 1 + torch.sigmoid(input) )

class SimpleMLP_Logish(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            Logish(),
            nn.Linear(width, width),#2
            Logish(),    
            nn.Linear(width, width),#3
            Logish(),  
            nn.Linear(width, width),#4
            Logish(), 
            nn.Linear(width, width),#5
            Logish(),
            nn.Linear(width, width),#6
            Logish(),     
            nn.Linear(width, width),#7
            Logish(),  
            nn.Linear(width, width),#8
            Logish(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
LogishNet = SimpleMLP_Logish()
LogishNet.apply(init_weights_bias)
if is_use_cuda:
    LogishNet.to(device)
    LogishNet = nn.DataParallel(LogishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class Smish(nn.Module):
    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * torch.tanh( torch.log( 1 + torch.sigmoid(input) ) )

class SimpleMLP_Smish(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, width),#1
            Smish(),
            nn.Linear(width, width),#2
            Smish(),    
            nn.Linear(width, width),#3
            Smish(),  
            nn.Linear(width, width),#4
            Smish(), 
            nn.Linear(width, width),#5
            Smish(),
            nn.Linear(width, width),#6
            Smish(),     
            nn.Linear(width, width),#7
            Smish(),  
            nn.Linear(width, width),#8
            Smish(),
            nn.Linear(width, 10)
        )

    def forward(self, x):
        return self.layers(x)
       
SmishNet = SimpleMLP_Smish()
SmishNet.apply(init_weights_bias)
if is_use_cuda:
    SmishNet.to(device)
    SmishNet = nn.DataParallel(SmishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


def lr_schedule(lr, epoch):
    optim_factor = 0
    # if epoch > 160:
    #     optim_factor = 3
    # elif epoch > 120:
    #     optim_factor = 2
    # elif epoch > 60:
    #     optim_factor = 1
        
    return lr * math.pow(0.2, optim_factor)

def train(epoch, activation):
    #torch.autograd.set_detect_anomaly(True)
    # if activation == "TeLU":
    #     net = TeLUNet
    # elif activation == "Mish":
    #     net = MishNet
    # elif activation == "ReLU":
    #     net = ReLUNet
    if activation == "TeLU":
        net = TeLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "SiLU":
        net = SiLUNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "ELU":
        net = ELUNet
    elif activation == "Logish":
        net = LogishNet
    elif activation == "Smish":
        net = SmishNet

    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.0, weight_decay=0.001) #0.0005
    
    # print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #print(inputs[0])
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
    # if activation == "TeLU":
    #     net = TeLUNet
    # elif activation == "Mish":
    #     net = MishNet
    # elif activation == "ReLU":
    #     net = ReLUNet
    if activation == "TeLU":
        net = TeLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "SiLU":
        net = SiLUNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "ELU":
        net = ELUNet
    elif activation == "Logish":
        net = LogishNet
    elif activation == "Smish":
        net = SmishNet
    
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

def test0(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader0):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test1(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader1):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test2(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader2):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test3(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader3):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test4(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader4):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test5(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader5):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test6(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader6):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test7(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader7):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test8(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader8):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def test9(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader9):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

def testx(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    elif activation == "Mish":
        net = bestMishNet
    elif activation == "ReLU":
        net = bestReLUNet
    elif activation == "SiLU":
        net = bestSiLUNet
    elif activation == "GELU":
        net = bestGELUNet
    elif activation == "ELU":
        net = bestELUNet
    elif activation == "Logish":
        net = bestLogishNet
    elif activation == "Smish":
        net = bestSmishNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loaderx):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predict == labels).cpu().sum().item()
        
    return (100 * correct / total)

TeLUTrainAccuracies = []
TeLUValidAccuracies = []
TeLUTrainRuntimes = []
TeLUValidRuntimes = []
TeLUTrainLoss = []
TeLUValidLoss = []
TeLUTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"TeLU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "TeLU")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestTeLUNet = copy.deepcopy(TeLUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    TeLUTrainAccuracies.append(train_accuracy)
    TeLUValidAccuracies.append(valid_accuracy)
    TeLUTrainRuntimes.append(train_time)
    TeLUValidRuntimes.append(valid_time)
    TeLUTrainLoss.append(train_loss)
    TeLUValidLoss.append(valid_loss)
TeLUTestAccuracy.append(test0(_epoch, "TeLU"))
TeLUTestAccuracy.append(test1(_epoch, "TeLU"))
TeLUTestAccuracy.append(test2(_epoch, "TeLU"))
TeLUTestAccuracy.append(test3(_epoch, "TeLU"))
TeLUTestAccuracy.append(test4(_epoch, "TeLU"))
TeLUTestAccuracy.append(test5(_epoch, "TeLU"))
TeLUTestAccuracy.append(test6(_epoch, "TeLU"))
TeLUTestAccuracy.append(test7(_epoch, "TeLU"))
TeLUTestAccuracy.append(test8(_epoch, "TeLU"))
TeLUTestAccuracy.append(test9(_epoch, "TeLU"))
TeLUTestAccuracy.append(testx(_epoch, "TeLU"))
print("test_acc:", TeLUTestAccuracy)
print("Finished TeLU")

'''MishTrainAccuracies = []
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
MishTestAccuracy.append(test0(_epoch, "Mish"))
MishTestAccuracy.append(test1(_epoch, "Mish"))
MishTestAccuracy.append(test2(_epoch, "Mish"))
MishTestAccuracy.append(test3(_epoch, "Mish"))
MishTestAccuracy.append(test4(_epoch, "Mish"))
MishTestAccuracy.append(test5(_epoch, "Mish"))
MishTestAccuracy.append(test6(_epoch, "Mish"))
MishTestAccuracy.append(test7(_epoch, "Mish"))
MishTestAccuracy.append(test8(_epoch, "Mish"))
MishTestAccuracy.append(test9(_epoch, "Mish"))
MishTestAccuracy.append(testx(_epoch, "Mish"))
print("test_acc:", MishTestAccuracy)
print("Finished Mish")
'''
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
ReLUTestAccuracy.append(test0(_epoch, "ReLU"))
ReLUTestAccuracy.append(test1(_epoch, "ReLU"))
ReLUTestAccuracy.append(test2(_epoch, "ReLU"))
ReLUTestAccuracy.append(test3(_epoch, "ReLU"))
ReLUTestAccuracy.append(test4(_epoch, "ReLU"))
ReLUTestAccuracy.append(test5(_epoch, "ReLU"))
ReLUTestAccuracy.append(test6(_epoch, "ReLU"))
ReLUTestAccuracy.append(test7(_epoch, "ReLU"))
ReLUTestAccuracy.append(test8(_epoch, "ReLU"))
ReLUTestAccuracy.append(test9(_epoch, "ReLU"))
ReLUTestAccuracy.append(testx(_epoch, "ReLU"))
print("test_acc:", ReLUTestAccuracy)
print("Finished ReLU")
'''
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
GELUTestAccuracy.append(test0(_epoch, "GELU"))
GELUTestAccuracy.append(test1(_epoch, "GELU"))
GELUTestAccuracy.append(test2(_epoch, "GELU"))
GELUTestAccuracy.append(test3(_epoch, "GELU"))
GELUTestAccuracy.append(test4(_epoch, "GELU"))
GELUTestAccuracy.append(test5(_epoch, "GELU"))
GELUTestAccuracy.append(test6(_epoch, "GELU"))
GELUTestAccuracy.append(test7(_epoch, "GELU"))
GELUTestAccuracy.append(test8(_epoch, "GELU"))
GELUTestAccuracy.append(test9(_epoch, "GELU"))
GELUTestAccuracy.append(testx(_epoch, "GELU"))
print("test_acc:", GELUTestAccuracy)
print("Finished GELU")

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
SiLUTestAccuracy.append(test0(_epoch, "SiLU"))
SiLUTestAccuracy.append(test1(_epoch, "SiLU"))
SiLUTestAccuracy.append(test2(_epoch, "SiLU"))
SiLUTestAccuracy.append(test3(_epoch, "SiLU"))
SiLUTestAccuracy.append(test4(_epoch, "SiLU"))
SiLUTestAccuracy.append(test5(_epoch, "SiLU"))
SiLUTestAccuracy.append(test6(_epoch, "SiLU"))
SiLUTestAccuracy.append(test7(_epoch, "SiLU"))
SiLUTestAccuracy.append(test8(_epoch, "SiLU"))
SiLUTestAccuracy.append(test9(_epoch, "SiLU"))
SiLUTestAccuracy.append(testx(_epoch, "SiLU"))
print("test_acc:", SiLUTestAccuracy)
print("Finished SiLU")

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
SmishTestAccuracy.append(test0(_epoch, "Smish"))
SmishTestAccuracy.append(test1(_epoch, "Smish"))
SmishTestAccuracy.append(test2(_epoch, "Smish"))
SmishTestAccuracy.append(test3(_epoch, "Smish"))
SmishTestAccuracy.append(test4(_epoch, "Smish"))
SmishTestAccuracy.append(test5(_epoch, "Smish"))
SmishTestAccuracy.append(test6(_epoch, "Smish"))
SmishTestAccuracy.append(test7(_epoch, "Smish"))
SmishTestAccuracy.append(test8(_epoch, "Smish"))
SmishTestAccuracy.append(test9(_epoch, "Smish"))
SmishTestAccuracy.append(testx(_epoch, "Smish"))
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
LogishTestAccuracy.append(test0(_epoch, "Logish"))
LogishTestAccuracy.append(test1(_epoch, "Logish"))
LogishTestAccuracy.append(test2(_epoch, "Logish"))
LogishTestAccuracy.append(test3(_epoch, "Logish"))
LogishTestAccuracy.append(test4(_epoch, "Logish"))
LogishTestAccuracy.append(test5(_epoch, "Logish"))
LogishTestAccuracy.append(test6(_epoch, "Logish"))
LogishTestAccuracy.append(test7(_epoch, "Logish"))
LogishTestAccuracy.append(test8(_epoch, "Logish"))
LogishTestAccuracy.append(test9(_epoch, "Logish"))
LogishTestAccuracy.append(testx(_epoch, "Logish"))
print("test_acc:", LogishTestAccuracy)
print("Finished Logish")
'''
ELUTrainAccuracies = []
ELUValidAccuracies = []
ELUTrainRuntimes = []
ELUValidRuntimes = []
ELUTrainLoss = []
ELUValidLoss = []
ELUTestAccuracy = []
TopValidAccuracy = 0
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"ELU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "ELU")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        bestELUNet = copy.deepcopy(ELUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    ELUTrainAccuracies.append(train_accuracy)
    ELUValidAccuracies.append(valid_accuracy)
    ELUTrainRuntimes.append(train_time)
    ELUValidRuntimes.append(valid_time)
    ELUTrainLoss.append(train_loss)
    ELUValidLoss.append(valid_loss)
ELUTestAccuracy.append(test0(_epoch, "ELU"))
ELUTestAccuracy.append(test1(_epoch, "ELU"))
ELUTestAccuracy.append(test2(_epoch, "ELU"))
ELUTestAccuracy.append(test3(_epoch, "ELU"))
ELUTestAccuracy.append(test4(_epoch, "ELU"))
ELUTestAccuracy.append(test5(_epoch, "ELU"))
ELUTestAccuracy.append(test6(_epoch, "ELU"))
ELUTestAccuracy.append(test7(_epoch, "ELU"))
ELUTestAccuracy.append(test8(_epoch, "ELU"))
ELUTestAccuracy.append(test9(_epoch, "ELU"))
ELUTestAccuracy.append(testx(_epoch, "ELU"))
print("test_acc:", ELUTestAccuracy)
print("Finished ELU")


with open('MLP_FMNIST_Trial_0.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator="\n")
    wr.writerow(TeLUTrainAccuracies)
    #wr.writerow(MishTrainAccuracies)
    wr.writerow(ReLUTrainAccuracies)
    #wr.writerow(SiLUTrainAccuracies)
    wr.writerow(ELUTrainAccuracies)
    #wr.writerow(GELUTrainAccuracies)
    #wr.writerow(LogishTrainAccuracies)
    #wr.writerow(SmishTrainAccuracies)

    wr.writerow(TeLUValidAccuracies)
    #wr.writerow(MishValidAccuracies)
    wr.writerow(ReLUValidAccuracies)
    #wr.writerow(SiLUValidAccuracies)
    wr.writerow(ELUValidAccuracies)
    #wr.writerow(GELUValidAccuracies)
    #wr.writerow(LogishValidAccuracies)
    #wr.writerow(SmishValidAccuracies)

    # wr.writerow(TeLUTrainLoss)
    # wr.writerow(MishTrainLoss)
    # wr.writerow(ReLUTrainLoss)

    # wr.writerow(TeLUValidLoss)
    # wr.writerow(MishValidLoss)
    # wr.writerow(ReLUValidLoss)

    # wr.writerow(TeLUTrainRuntimes)
    # wr.writerow(MishTrainRuntimes)
    # wr.writerow(ReLUTrainRuntimes)

    # wr.writerow(TeLUValidRuntimes)
    # wr.writerow(MishValidRuntimes)
    # wr.writerow(ReLUValidRuntimes)

    # wr.writerow(TeLUTestAccuracy)
    # wr.writerow(MishTestAccuracy)
    # wr.writerow(ReLUTestAccuracy)

    wr.writerow(TeLUTestAccuracy)
    #wr.writerow(MishTestAccuracy)
    wr.writerow(ReLUTestAccuracy)
    #wr.writerow(SiLUTestAccuracy)
    wr.writerow(ELUTestAccuracy)
    #wr.writerow(GELUTestAccuracy)
    #wr.writerow(LogishTestAccuracy)
    #wr.writerow(SmishTestAccuracy)

    wr.writerow(TeLUTrainRuntimes)
    #wr.writerow(MishTrainRuntimes)
    wr.writerow(ReLUTrainRuntimes)
    #wr.writerow(SiLUTrainRuntimes)
    wr.writerow(ELUTrainRuntimes)
    #wr.writerow(GELUTrainRuntimes)
    #wr.writerow(LogishTrainRuntimes)
    #wr.writerow(SmishTrainRuntimes)

print("MLP FMNIST trial complete")
