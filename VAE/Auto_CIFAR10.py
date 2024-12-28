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
    #transforms.RandomCrop(28, padding=2),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_valid  = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_dataset   = torchvision.datasets.CIFAR10(root='./data', transform=transform_train, train=True, download=True)
train_dataset = torch.utils.data.Subset(train_dataset, range(45000))
print(len(train_dataset))
valid_dataset   = torchvision.datasets.CIFAR10(root='./data', transform=transform_valid, train=True, download=True)
valid_dataset = torch.utils.data.Subset(valid_dataset, range(45000,50000))
print(len(valid_dataset))
test_dataset    = torchvision.datasets.CIFAR10(root='./data', transform=transform_test, train=False, download=True)
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
            nn.Linear(32*32, 256),
            TeLU(),
            nn.Linear(256, 64),
            TeLU(),
            nn.Linear(64, 16),
            TeLU(),
            nn.Linear(16, 64),
            TeLU(),
            nn.Linear(64, 256),
            TeLU(),
            nn.Linear(256, 32*32),
        )

    def forward(self, x):
        return self.layers(x)
       
TeLUNet = SimpleMLP_TeLU()
TeLUNet.apply(init_weights_bias)
if is_use_cuda:
    TeLUNet.to(device)
    TeLUNet = nn.DataParallel(TeLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.MSELoss()



# class SimpleMLP_GELU(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28, 50),
#             nn.GELU(),
#             nn.Linear(50, 50),
#             nn.GELU(),
#             nn.Linear(50, 10)
#         )

#     def forward(self, x):
#         return self.layers(x)
       
# GELUNet = SimpleMLP_GELU()
# GELUNet.apply(init_weights_bias)
# if is_use_cuda:
#     GELUNet.to(device)
#     GELUNet = nn.DataParallel(GELUNet, device_ids=range(torch.cuda.device_count()))
# criterion = nn.CrossEntropyLoss()
       


class SimpleMLP_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 32*32),
        )

    def forward(self, x):
        return self.layers(x)
       
ReLUNet = SimpleMLP_ReLU()
ReLUNet.apply(init_weights_bias)
if is_use_cuda:
    ReLUNet.to(device)
    ReLUNet = nn.DataParallel(ReLUNet, device_ids=range(torch.cuda.device_count()))
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
    # elif activation == "GELU":
    #     net = GELUNet
    elif activation == "ReLU":
        net = ReLUNet

    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.9, weight_decay=5e-4)
    
    # print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs = inputs.to(device)
        optimizer.zero_grad()
        #print(inputs[0])
        outputs        = net(inputs)
        loss           = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        # sys.stdout.write('\r')
        # sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
        #                 % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
        #                    epoch, num_epochs, idx, len(train_dataset) // batch_size, 
        #                   train_loss / (batch_size * (idx + 1)), correct / total))
        # sys.stdout.flush()
    return train_loss


def validate(epoch, activation):
    if activation == "TeLU":
        net = TeLUNet
    # elif activation == "GELU":
    #     net = GELUNet
    elif activation == "ReLU":
        net = ReLUNet
    
    net.eval()
    valid_loss = 0
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(valid_loader):
        if is_use_cuda:
            inputs = inputs.to(device)
        outputs        = net(inputs)
        loss           = criterion(outputs, inputs)
        
        valid_loss  += loss.item()
        
    return valid_loss


def test(epoch, activation):
    if activation == "TeLU":
        net = bestTeLUNet
    # elif activation == "GELU":
    #     net = bestGELUNet
    elif activation == "ReLU":
        net = bestReLUNet
    
    net.eval()
    test_loss = 0
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs = inputs.to(device)
        outputs        = net(inputs)
        loss           = criterion(outputs, inputs)
        
        test_loss  += loss.item()
        
    return test_loss



TeLUTrainRuntimes = []
TeLUValidRuntimes = []
TeLUTrainLoss = []
TeLUValidLoss = []
TeLUTestLoss = []
BestValidLoss = 999999
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_loss = train(_epoch,"TeLU")
    inter_time = time.time()
    valid_loss = validate(_epoch, "TeLU")
    valid_end_time = time.time()
    if valid_loss < BestValidLoss:
        BestValidLoss = valid_loss
        bestTeLUNet = copy.deepcopy(TeLUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    TeLUTrainRuntimes.append(train_time)
    TeLUValidRuntimes.append(valid_time)
    TeLUTrainLoss.append(train_loss)
    TeLUValidLoss.append(valid_loss)
TeLUTestLoss.append(test(_epoch, "TeLU"))
print("test_err:", TeLUTestLoss)
print("Finished TeLU")

# GELUTrainAccuracies = []
# GELUValidAccuracies = []
# GELUTrainRuntimes = []
# GELUValidRuntimes = []
# GELUTrainLoss = []
# GELUValidLoss = []
# GELUTestAccuracy = []
# TopValidAccuracy = 0
# for _epoch in range(start_epoch, start_epoch + num_epochs):
#     train_start_time = time.time()
#     train_erruracy, train_loss = train(_epoch,"GELU")
#     inter_time = time.time()
#     valid_erruracy, valid_loss = validate(_epoch, "GELU")
#     valid_end_time = time.time()
#     if valid_erruracy > TopValidAccuracy:
#         TopValidAccuracy = valid_erruracy
#         bestGELUNet = copy.deepcopy(GELUNet)
#     train_time = inter_time - train_start_time
#     valid_time = valid_end_time - inter_time
#     print(f'[{_epoch}] train_err: {train_erruracy:.3f}% -- valid_err: {valid_erruracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
#     GELUTrainAccuracies.append(train_erruracy)
#     GELUValidAccuracies.append(valid_erruracy)
#     GELUTrainRuntimes.append(train_time)
#     GELUValidRuntimes.append(valid_time)
#     GELUTrainLoss.append(train_loss)
#     GELUValidLoss.append(valid_loss)
# GELUTestAccuracy.append(test(_epoch, "GELU"))
# print("test_err:", GELUTestAccuracy)
# print("Finished GELU")

ReLUTrainRuntimes = []
ReLUValidRuntimes = []
ReLUTrainLoss = []
ReLUValidLoss = []
ReLUTestLoss = []
BestValidLoss = 999999
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_loss = train(_epoch,"ReLU")
    inter_time = time.time()
    valid_loss = validate(_epoch, "ReLU")
    valid_end_time = time.time()
    if valid_loss < BestValidLoss:
        BestValidLoss = valid_loss
        bestReLUNet = copy.deepcopy(ReLUNet)
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    ReLUTrainRuntimes.append(train_time)
    ReLUValidRuntimes.append(valid_time)
    ReLUTrainLoss.append(train_loss)
    ReLUValidLoss.append(valid_loss)
ReLUTestLoss.append(test(_epoch, "ReLU"))
print("test_err:", ReLUTestLoss)
print("Finished ReLU")


with open('MLP_FMNIST_Trial322.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator="\n")
    wr.writerow(TeLUTrainLoss)
    #wr.writerow(GELUTrainAccuracies)
    wr.writerow(ReLUTrainLoss)

    wr.writerow(TeLUValidLoss)
    #wr.writerow(GELUValidAccuracies)
    wr.writerow(ReLUValidLoss)

    # wr.writerow(TeLUTrainLoss)
    # #wr.writerow(GELUTrainLoss)
    # wr.writerow(ReLUTrainLoss)

    # wr.writerow(TeLUValidLoss)
    # #wr.writerow(GELUValidLoss)
    # wr.writerow(ReLUValidLoss)

    # wr.writerow(TeLUTrainRuntimes)
    # #wr.writerow(GELUTrainRuntimes)
    # wr.writerow(ReLUTrainRuntimes)

    # wr.writerow(TeLUValidRuntimes)
    # #wr.writerow(GELUValidRuntimes)
    # wr.writerow(ReLUValidRuntimes)

    wr.writerow(TeLUTestLoss)
    #wr.writerow(GELUTestAccuracy)
    wr.writerow(ReLUTestLoss)

print("AUTO CIFAR10 trial complete")
