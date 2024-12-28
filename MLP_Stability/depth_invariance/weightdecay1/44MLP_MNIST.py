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

train_dataset   = torchvision.datasets.MNIST(root='./data', transform=transform_train, train=True, download=True)
train_dataset = torch.utils.data.Subset(train_dataset, range(50000))
print(len(train_dataset))
valid_dataset   = torchvision.datasets.MNIST(root='./data', transform=transform_valid, train=True, download=True)
valid_dataset = torch.utils.data.Subset(valid_dataset, range(50000,60000))
print(len(valid_dataset))
test_dataset    = torchvision.datasets.MNIST(root='./data', transform=transform_test, train=False, download=True)
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
            nn.Linear(width, width),#9
            TeLU(),
            nn.Linear(width, width),#10
            TeLU(),     
            nn.Linear(width, width),#11
            TeLU(),  
            nn.Linear(width, width),#12
            TeLU(), 
            nn.Linear(width, width),#13
            TeLU(),
            nn.Linear(width, width),#14
            TeLU(),     
            nn.Linear(width, width),#15
            TeLU(),  
            nn.Linear(width, width),#16
            TeLU(), 
            nn.Linear(width, width),#17
            TeLU(),
            nn.Linear(width, width),#18
            TeLU(),     
            nn.Linear(width, width),#19
            TeLU(),  
            nn.Linear(width, width),#20
            TeLU(), 
            nn.Linear(width, width),#21
            TeLU(),
            nn.Linear(width, width),#22
            TeLU(),     
            nn.Linear(width, width),#23
            TeLU(),  
            nn.Linear(width, width),#24
            TeLU(), 
            nn.Linear(width, width),#25
            TeLU(),
            nn.Linear(width, width),#26
            TeLU(),     
            nn.Linear(width, width),#27
            TeLU(),  
            nn.Linear(width, width),#28
            TeLU(), 
            nn.Linear(width, width),#29
            TeLU(),
            nn.Linear(width, width),#30
            TeLU(),     
            nn.Linear(width, width),#31
            TeLU(),  
            nn.Linear(width, width),#32
            TeLU(), 
            nn.Linear(width, width),#33
            TeLU(),
            nn.Linear(width, width),#34
            TeLU(),     
            nn.Linear(width, width),#35
            TeLU(),  
            nn.Linear(width, width),#36
            TeLU(), 
            nn.Linear(width, width),#37
            TeLU(),
            nn.Linear(width, width),#38
            TeLU(),     
            nn.Linear(width, width),#39
            TeLU(),  
            nn.Linear(width, width),#40
            TeLU(),
            nn.Linear(width, width),#41
            TeLU(),
            nn.Linear(width, width),#42
            TeLU(),
            nn.Linear(width, width),#43
            TeLU(),
            nn.Linear(width, width),#44
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
            nn.Linear(width, width),#9
            nn.Mish(),
            nn.Linear(width, width),#10
            nn.Mish(),     
            nn.Linear(width, width),#11
            nn.Mish(),  
            nn.Linear(width, width),#12
            nn.Mish(), 
            nn.Linear(width, width),#13
            nn.Mish(),
            nn.Linear(width, width),#14
            nn.Mish(),     
            nn.Linear(width, width),#15
            nn.Mish(),  
            nn.Linear(width, width),#16
            nn.Mish(), 
            nn.Linear(width, width),#17
            nn.Mish(),
            nn.Linear(width, width),#18
            nn.Mish(),     
            nn.Linear(width, width),#19
            nn.Mish(),  
            nn.Linear(width, width),#20
            nn.Mish(), 
            nn.Linear(width, width),#21
            nn.Mish(),
            nn.Linear(width, width),#22
            nn.Mish(),     
            nn.Linear(width, width),#23
            nn.Mish(), 
            nn.Linear(width, width),#24
            nn.Mish(),       
            nn.Linear(width, width),#25
            nn.Mish(),
            nn.Linear(width, width),#26
            nn.Mish(),     
            nn.Linear(width, width),#27
            nn.Mish(),  
            nn.Linear(width, width),#28
            nn.Mish(), 
            nn.Linear(width, width),#29
            nn.Mish(),
            nn.Linear(width, width),#30
            nn.Mish(),     
            nn.Linear(width, width),#31
            nn.Mish(), 
            nn.Linear(width, width),#32
            nn.Mish(),        
            nn.Linear(width, width),#33
            nn.Mish(),
            nn.Linear(width, width),#34
            nn.Mish(),     
            nn.Linear(width, width),#35
            nn.Mish(),  
            nn.Linear(width, width),#36
            nn.Mish(), 
            nn.Linear(width, width),#37
            nn.Mish(),
            nn.Linear(width, width),#38
            nn.Mish(),     
            nn.Linear(width, width),#39
            nn.Mish(), 
            nn.Linear(width, width),#40
            nn.Mish(),
            nn.Linear(width, width),#41
            nn.Mish(),
            nn.Linear(width, width),#42
            nn.Mish(),
            nn.Linear(width, width),#43
            nn.Mish(),
            nn.Linear(width, width),#44
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
            nn.Linear(width, width),#9
            nn.ReLU(),
            nn.Linear(width, width),#10
            nn.ReLU(),     
            nn.Linear(width, width),#11
            nn.ReLU(),  
            nn.Linear(width, width),#12
            nn.ReLU(), 
            nn.Linear(width, width),#13
            nn.ReLU(),
            nn.Linear(width, width),#14
            nn.ReLU(),     
            nn.Linear(width, width),#15
            nn.ReLU(),  
            nn.Linear(width, width),#16
            nn.ReLU(), 
            nn.Linear(width, width),#17
            nn.ReLU(),
            nn.Linear(width, width),#18
            nn.ReLU(),     
            nn.Linear(width, width),#19
            nn.ReLU(),  
            nn.Linear(width, width),#20
            nn.ReLU(), 
            nn.Linear(width, width),#21
            nn.ReLU(),
            nn.Linear(width, width),#22
            nn.ReLU(),     
            nn.Linear(width, width),#23
            nn.ReLU(), 
            nn.Linear(width, width),#24
            nn.ReLU(),           
            nn.Linear(width, width),#25
            nn.ReLU(),
            nn.Linear(width, width),#26
            nn.ReLU(),     
            nn.Linear(width, width),#27
            nn.ReLU(),  
            nn.Linear(width, width),#28
            nn.ReLU(), 
            nn.Linear(width, width),#29
            nn.ReLU(),
            nn.Linear(width, width),#30
            nn.ReLU(),     
            nn.Linear(width, width),#31
            nn.ReLU(), 
            nn.Linear(width, width),#32
            nn.ReLU(),              
            nn.Linear(width, width),#33
            nn.ReLU(),
            nn.Linear(width, width),#34
            nn.ReLU(),     
            nn.Linear(width, width),#35
            nn.ReLU(),  
            nn.Linear(width, width),#36
            nn.ReLU(), 
            nn.Linear(width, width),#37
            nn.ReLU(),
            nn.Linear(width, width),#38
            nn.ReLU(),     
            nn.Linear(width, width),#39
            nn.ReLU(), 
            nn.Linear(width, width),#40
            nn.ReLU(),
            nn.Linear(width, width),#41
            nn.ReLU(),
            nn.Linear(width, width),#42
            nn.ReLU(),
            nn.Linear(width, width),#43
            nn.ReLU(),
            nn.Linear(width, width),#44
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
    elif activation == "Mish":
        net = MishNet
    elif activation == "ReLU":
        net = ReLUNet

    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum=0.9, weight_decay=1e-3)
    
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
    elif activation == "Mish":
        net = MishNet
    elif activation == "ReLU":
        net = ReLUNet
    
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
    elif activation == "Mish":
        net = MishNet
    elif activation == "ReLU":
        net = ReLUNet
    
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


with open('MLP_MNIST_Trial_44_5.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator="\n")
    wr.writerow(TeLUTrainAccuracies)
    wr.writerow(MishTrainAccuracies)
    wr.writerow(ReLUTrainAccuracies)

    wr.writerow(TeLUValidAccuracies)
    wr.writerow(MishValidAccuracies)
    wr.writerow(ReLUValidAccuracies)

    wr.writerow(TeLUTrainLoss)
    wr.writerow(MishTrainLoss)
    wr.writerow(ReLUTrainLoss)

    wr.writerow(TeLUValidLoss)
    wr.writerow(MishValidLoss)
    wr.writerow(ReLUValidLoss)

    wr.writerow(TeLUTrainRuntimes)
    wr.writerow(MishTrainRuntimes)
    wr.writerow(ReLUTrainRuntimes)

    wr.writerow(TeLUValidRuntimes)
    wr.writerow(MishValidRuntimes)
    wr.writerow(ReLUValidRuntimes)

    wr.writerow(TeLUTestAccuracy)
    wr.writerow(MishTestAccuracy)
    wr.writerow(ReLUTestAccuracy)

print("MLP FMNIST trial complete")
