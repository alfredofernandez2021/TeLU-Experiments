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


lr           = 0.005
start_epoch  = 1
num_epochs   = 100
batch_size   = 128
width = 128

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
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

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,))
])

train_dataset   = torchvision.datasets.FashionMNIST(root='./data', transform=transform_train, train=True, download=True)
train_dataset = torch.utils.data.Subset(train_dataset, range(50000))
print(len(train_dataset))
valid_dataset   = torchvision.datasets.FashionMNIST(root='./data', transform=transform_valid, train=True, download=True)
valid_dataset = torch.utils.data.Subset(valid_dataset, range(50000,60000))
print(len(valid_dataset))
test_dataset    = torchvision.datasets.FashionMNIST(root='./data', transform=transform_test, train=False, download=True)
train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
valid_loader    = torch.utils.data.DataLoader(valid_dataset, batch_size=80, num_workers=1, shuffle=False)
test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=80, num_workers=1, shuffle=False)


# def telu(input):
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
    if activation == "TeLU":
        net = TeLUNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "SiLU":
        net = SiLUNet
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
    optimizer  = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.9, weight_decay=5e-3)
    
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
    if activation == "TeLU":
        net = TeLUNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "SiLU":
        net = SiLUNet
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

def test(epoch, activation):
    if activation == "TeLU":
        net = TeLUNet
    elif activation == "ReLU":
        net = ReLUNet
    elif activation == "Mish":
        net = MishNet
    elif activation == "GELU":
        net = GELUNet
    elif activation == "SiLU":
        net = SiLUNet
    elif activation == "ELU":
        net = ELUNet
    elif activation == "Logish":
        net = LogishNet
    elif activation == "Smish":
        net = SmishNet
    
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
TeLUTestAccuracy.append(test(_epoch, "TeLU"))
print("test_acc:", TeLUTestAccuracy)
print("Finished TeLU")

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
ELUTestAccuracy.append(test(_epoch, "ELU"))
print("test_acc:", ELUTestAccuracy)
print("Finished ELU")

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


with open('F_8_4.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator="\n")
    wr.writerow(ReLUTrainAccuracies)
    wr.writerow(TeLUTrainAccuracies)
    wr.writerow(ELUTrainAccuracies)
    wr.writerow(SiLUTrainAccuracies)
    wr.writerow(GELUTrainAccuracies)
    wr.writerow(MishTrainAccuracies)
    wr.writerow(LogishTrainAccuracies)
    wr.writerow(SmishTrainAccuracies)

    wr.writerow(ReLUValidAccuracies)
    wr.writerow(TeLUValidAccuracies)
    wr.writerow(ELUValidAccuracies)
    wr.writerow(SiLUValidAccuracies)
    wr.writerow(GELUValidAccuracies)
    wr.writerow(MishValidAccuracies)
    wr.writerow(LogishValidAccuracies)
    wr.writerow(SmishValidAccuracies)

    wr.writerow(ReLUTestAccuracy)
    wr.writerow(TeLUTestAccuracy)
    wr.writerow(ELUTestAccuracy)
    wr.writerow(SiLUTestAccuracy)
    wr.writerow(GELUTestAccuracy)
    wr.writerow(MishTestAccuracy)
    wr.writerow(LogishTestAccuracy)
    wr.writerow(SmishTestAccuracy)

print("MLP FMNIST trial complete")
