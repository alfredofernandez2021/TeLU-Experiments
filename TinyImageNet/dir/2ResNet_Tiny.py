import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
import numpy as np
from collections import defaultdict
#import imageio.v2 as imageio
#from tqdm.autonotebook import tqdm
from PIL import Image as im 
#from mixmo.utils import torchutils

import math
import sys
import time
import os
import csv


lr           = 0.05
start_epoch  = 1
num_epochs   = 100
batch_size   = 128
BATCHNORM_MOMENTUM_PREACT = 0.1

def _calculate_fan_in_and_fan_out(tensor):
    """
    Compute the minimal input and output sizes for the weight tensor
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode):
    """
    Return the minimal input or output sizes for the weight tensor depending on which is needed
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def truncated_normal_(tensor, mean=0, std=1):
    """
    Initialization function
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def kaiming_normal_truncated(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    
    fan = _calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    std = std / .87962566103423978
    with torch.no_grad():
        return truncated_normal_(tensor, 0, std)

def weights_init_hetruncatednormal(m, dense_gaussian=False):
    """
    Simple init function
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        kaiming_normal_truncated(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant(m.bias, 0)
    elif classname.find('Linear') != -1:
        if dense_gaussian:
            nn.init.normal_(m.weight.data, mean=0, std=0.01)
        else:
            kaiming_normal_truncated(
                m.weight.data, a=0, mode='fan_in', nonlinearity='relu'
            )
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def download_and_unzip(URL, root_dir):
  error_message = "Not implemented"
  raise NotImplementedError(error_message.format(URL))

def _add_channels(img, total_channels=3):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

class TinyImageNetPaths:
  def __init__(self, root_dir, download=False):
    if download:
      download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                         root_dir)
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': [],  # img_path
    }

    # Get the test paths
    test_path_images = os.path.join(test_path, "images")
    self.paths['test'] = list(map(lambda x: os.path.join(test_path_images, x), os.listdir(test_path_images)))
    # self.paths['test'] = []
    # test_files = os.listdir(test_path_images)
    # for testfile in test_files:
    #     self.paths['test'].append(os.path.join(test_path, 'images', testfile))

    # self.paths['test']
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths and labels
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))

class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir = "tiny-imagenet-200/", mode='train', preload=True, load_transform=None,
               transform=None, download=False, max_samples=None):
    tinp = TinyImageNetPaths(root_dir, download)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.preload = preload
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []
    #self.instances = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]
      #self.samples = []

    if self.preload:
      load_desc = "Preloading {} data...".format(mode)
      #self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32)
      #self.img_data = []
      #self.label_data = np.zeros((self.samples_num,), dtype=np.int32)
      #self.label_data = []
      #for idx in tqdm(range(self.samples_num), desc=load_desc):
      for idx in range(self.samples_num):
        s = self.samples[idx]
        ########################
        if(mode != 'test'):
            #img = imageio.imread(s[0])
            img = im.open(s[0]).convert('RGB')
        else:
            #img = imageio.imread(s)
            img = im.open(s).convert('RGB')
        self.img_data.append(img)
        #print(type(img))
        #print(type(self.img_data[idx]))
        if mode != 'test':
          self.label_data.append(s[self.label_idx])

      if load_transform:
        for lt in load_transform:
          result = lt(self.img_data, self.label_data)
          self.img_data, self.label_data = result[:2]
          if len(result) > 2:
            self.transform_results.update(result[2])

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    if self.preload:
      img = self.img_data[idx]
      lbl = None if self.mode == 'test' else self.label_data[idx]
    else:
      s = self.samples[idx]
      ###########################
    #   img = imageio.imread(s[0])
    #   img = _add_channels(img)
    #   img = im.fromarray(np.uint8(img)).convert('RGB')
      img = im.open(s[0]).convert('RGB')
      lbl = None if self.mode == 'test' else s[self.label_idx]
    #sample = {'image': img, 'label': lbl}
    sample = (img, lbl)

    if self.transform:
    #   print(sample)
    #   print(len(sample))
    #   print(len(sample[0]))
      (image, label) = sample
      transformedImage = self.transform(image)
      sample = (transformedImage, label)

    return sample

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")
# Data Preprocess
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_valid  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# train_dataset   = TinyImageNetDataset(mode="train", transform=transform_train)
# print(len(train_dataset))
# valid_dataset   = TinyImageNetDataset(mode="val", transform=transform_valid)
# print(len(valid_dataset))
# test_dataset    = TinyImageNetDataset(mode="test", transform=transform_test)
# print(len(test_dataset))

#since test labels are not provided for TinyImageNet, validation set is used at test set and train set is partitioned to train/valid
dataset = TinyImageNetDataset(mode="train", transform=transform_train)
train_dataset = torch.utils.data.subset(dataset, range(90000))
print(len(train_dataset))
valid_dataset   = torch.utils.data.subset(dataset, range(90000,100000))
print(len(valid_dataset))
test_dataset    = TinyImageNetDataset(mode="val", transform=transform_test)
print(len(test_dataset))

train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
valid_loader    = torch.utils.data.DataLoader(valid_dataset, batch_size=80, num_workers=2, shuffle=False)
test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=80, num_workers=2, shuffle=False)


def sish(input):
    return input * torch.tanh(torch.exp(input))

def smish(input):
    return input * torch.tanh(torch.log( 1+torch.sigmoid(input) ))

def logish(input):
    return input * torch.log( 1+torch.sigmoid(input) )

tinyargs = {
            "num_classes": 200,
            "dataset_name": "tinyimagenet",
        }

res18config = {
        "classifier": "resnet",
        "depth": 18,
        "widen_factor": 1,
        "num_members": 1,
    }

class PreActBlock_Sish(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_Sish, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = sish(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(sish(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_Sish(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_Sish,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_Sish():
    return PreActResNet_Sish(tinyargs, res18config)
       
SishNet = ResNet18_Sish()

if is_use_cuda:
    SishNet.to(device)
    SishNet = nn.DataParallel(SishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class PreActBlock_GELU(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_GELU, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.gelu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_GELU(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_GELU,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_GELU():
    return PreActResNet_GELU(tinyargs, res18config)
    
GELUNet = ResNet18_GELU()

if is_use_cuda:
    GELUNet.to(device)
    GELUNet = nn.DataParallel(GELUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()

'''
class PreActBlock_Smish(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_Smish, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = smish(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(smish(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_Smish(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_Smish,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_Smish():
    return PreActResNet_Smish(tinyargs, res18config)

SmishNet = ResNet18_Smish()

if is_use_cuda:
    SmishNet.to(device)
    SmishNet = nn.DataParallel(SmishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class PreActBlock_Logish(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_Logish, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = logish(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(logish(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_Logish(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_Logish,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_Logish():
    return PreActResNet_Logish(tinyargs, res18config)
       
LogishNet = ResNet18_Logish()

if is_use_cuda:
    LogishNet.to(device)
    LogishNet = nn.DataParallel(LogishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()
'''

class PreActBlock_ReLU(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_ReLU, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_ReLU(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_ReLU,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_ReLU():
    return PreActResNet_ReLU(tinyargs, res18config)
    
ReLUNet = ResNet18_ReLU()

if is_use_cuda:
    ReLUNet.to(device)
    ReLUNet = nn.DataParallel(ReLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()

'''
class PreActBlock_Mish(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_Mish, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.mish(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.mish(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_Mish(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_Mish,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_Mish():
    return PreActResNet_Mish(tinyargs, res18config)
    
MishNet = ResNet18_Mish()
if is_use_cuda:
    MishNet.to(device)
    MishNet = nn.DataParallel(MishNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()


class PreActBlock_SiLU(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, **kwargs):
        super(PreActBlock_SiLU, self).__init__()
        final_planes = planes * self.expansion
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BATCHNORM_MOMENTUM_PREACT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BATCHNORM_MOMENTUM_PREACT)

        if stride != 1 or inplanes != final_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, final_planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.silu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.silu(self.bn2(out)))
        out += shortcut
        return out

class PreActResNet_SiLU(nn.Module):
    """
    Pre-activated ResNet network
    """

    def __init__(self, config_network, config_args):
        nn.Module.__init__(self)
        self.config_network = config_network
        self.config_args = config_args
        self._define_config()
        self._init_first_layer()
        self._init_core_network()
        self._init_final_classifier()
        self._init_weights_resnet()

    def _define_config(self):
        """
        Initialize network parameters from specified config
        """
        # network config
        self.num_classes = 200
        self.depth = 18
        self._init_block(widen_factor=3)

    def _init_block(self, widen_factor):
        """
        Build list of residual blocks for networks on the CIFAR datasets
        Network type specifies number of layers for CIFAR network
        """
        blocks = {
            18: PreActBlock_SiLU,
        }
        layers = {
            18: [2, 2, 2, 2],
        }
        assert layers[
            self.depth
        ], 'invalid depth for ResNet (self.depth should be one of 18, 34, 50, 101, 152, and 200)'

        self._layers = layers[self.depth]
        self._block = blocks[self.depth]
        assert widen_factor in [1., 2., 3.]
        self._nChannels = [
            64,
            64 * widen_factor, 128 * widen_factor,
            256 * widen_factor, 512 * widen_factor
        ]

    def _init_first_layer(self):
        #assert self.config_args["num_members"] == 1
        self.conv1 = self._make_conv1(nb_input_channel=3)

    def _init_core_network(self, max_layer=4):
        """
        Build the core of the Residual network (residual blocks)
        """

        self.inplanes = self._nChannels[0]

        self.layer1 = self._make_layer(self._block, planes=self._nChannels[1],
                                       blocks=self._layers[0], stride=1)
        self.layer2 = self._make_layer(self._block, planes=self._nChannels[2],
                                       blocks=self._layers[1], stride=2)
        self.layer3 = self._make_layer(self._block, planes=self._nChannels[3],
                                       blocks=self._layers[2], stride=2)

        if max_layer == 4:
            self.layer4 = self._make_layer(self._block, self._nChannels[4], blocks=self._layers[3], stride=2)

        self.features_dim = self._nChannels[-1] * self._block.expansion

    def _make_conv1(self, nb_input_channel):
        conv1 = nn.Conv2d(
            nb_input_channel, self._nChannels[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        return conv1

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
    ):
        """
        Build a layer of successive (residual) blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            inplanes=self.inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample)
                      )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,))

        return nn.Sequential(*layers)

    def _init_final_classifier(self):
        """
        Build linear classification head
        """
        self.fc = nn.Linear(self.features_dim, self.num_classes)

    dense_gaussian = True
    def _init_weights_resnet(self):
        """
        Apply specified random initializations to all modules of the network
        """
        for m in self.modules():
            weights_init_hetruncatednormal(m, dense_gaussian=self.dense_gaussian)

    def forward(self, x):
        if isinstance(x, dict):
            metadata = x["metadata"] or {}
            pixels = x["pixels"]
        else:
            metadata = {"mode": "inference"}
            pixels = x

        merged_representation = self._forward_first_layer(pixels, metadata)
        extracted_features = self._forward_core_network(merged_representation)
        dict_output = self._forward_final_classifier(extracted_features)
        return dict_output

    def _forward_first_layer(self, pixels, metadata=None):
        return self.conv1(pixels)

    def _forward_core_network(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x_avg = F.avg_pool2d(x, 4)
        return x_avg.view(x_avg.size(0), -1)

    def _forward_final_classifier(self, extracted_features):
        x = self.fc(extracted_features)
        dict_output = {"logits": x, "logits_0": x}
        return dict_output

def ResNet18_SiLU():
    return PreActResNet_SiLU(tinyargs, res18config)
    
SiLUNet = ResNet18_SiLU()

if is_use_cuda:
    SiLUNet.to(device)
    SiLUNet = nn.DataParallel(SiLUNet, device_ids=range(torch.cuda.device_count()))
criterion = nn.CrossEntropyLoss()
'''

def lr_schedule(lr, epoch):
    optim_factor = 0
    if epoch > 80:
        optim_factor = 3
    elif epoch > 60:
        optim_factor = 2
    elif epoch > 30:
        optim_factor = 1
        
    return lr * math.pow(0.1, optim_factor)

def train(epoch, activation):
    #torch.autograd.set_detect_anomaly(True)
    if activation == "Sish":
        net = SishNet
    elif activation == "GELU":
        net = GELUNet
    # elif activation == "Smish":
    #     net = SmishNet
    # elif activation == "Logish":
    #     net = LogishNet
    elif activation == "ReLU":
        net = ReLUNet
    # elif activation == "Mish":
    #     net = MishNet
    # elif activation == "SiLU":
    #     net = SiLUNet
    net.train()
    train_loss = 0
    correct    = 0
    total      = 0
    optimizer  = optim.SGD(net.parameters(), lr=lr_schedule(lr, epoch), momentum = 0.9, weight_decay=0.002)
    
    # print('Training Epoch: #%d, LR: %.4f'%(epoch, lr_schedule(lr, epoch)))
    for idx, (inputs, labels) in enumerate(train_loader):
    #for idx, sample in enumerate(train_loader):#
    #    inputs = sample['image']#
    #    labels = sample['label']#
        if is_use_cuda:
            #print(labels)
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs        = net(inputs)['logits']
        #print(outputs)
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
    # elif activation == "Smish":
    #     net = SmishNet
    # elif activation == "Logish":
    #     net = LogishNet
    elif activation == "ReLU":
        net = ReLUNet
    # elif activation == "Mish":
    #     net = MishNet
    # elif activation == "SiLU":
    #     net = SiLUNet
    
    net.eval()
    valid_loss = 0
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(valid_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs        = net(inputs)['logits']
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
        net = SishNet
    elif activation == "GELU":
        net = GELUNet
    # elif activation == "Smish":
    #     net = SmishNet
    # elif activation == "Logish":
    #     net = LogishNet
    elif activation == "ReLU":
        net = ReLUNet
    # elif activation == "Mish":
    #     net = MishNet
    # elif activation == "SiLU":
    #     net = SiLUNet
    
    net.eval()
    correct   = 0
    total     = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)['logits']
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
BestLearnedParameters = SishNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Sish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Sish")
    valid_end_time = time.time()
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = SishNet.state_dict()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    SishTrainAccuracies.append(train_accuracy)
    SishValidAccuracies.append(valid_accuracy)
    SishTrainRuntimes.append(train_time)
    SishValidRuntimes.append(valid_time)
    SishTrainLoss.append(train_loss)
    SishValidLoss.append(valid_loss)
SishNet.load_state_dict(BestLearnedParameters)
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
BestLearnedParameters = GELUNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"GELU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "GELU")
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = GELUNet.state_dict()
    valid_end_time = time.time()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    GELUTrainAccuracies.append(train_accuracy)
    GELUValidAccuracies.append(valid_accuracy)
    GELUTrainRuntimes.append(train_time)
    GELUValidRuntimes.append(valid_time)
    GELUTrainLoss.append(train_loss)
    GELUValidLoss.append(valid_loss)
GELUNet.load_state_dict(BestLearnedParameters)
GELUTestAccuracy.append(test(_epoch, "GELU"))
print("test_acc:", GELUTestAccuracy)
print("Finished GELU")

'''
SmishTrainAccuracies = []
SmishValidAccuracies = []
SmishTrainRuntimes = []
SmishValidRuntimes = []
SmishTrainLoss = []
SmishValidLoss = []
SmishTestAccuracy = []
TopValidAccuracy = 0
BestLearnedParameters = SmishNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Smish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Smish")
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = SmishNet.state_dict()
    valid_end_time = time.time()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    SmishTrainAccuracies.append(train_accuracy)
    SmishValidAccuracies.append(valid_accuracy)
    SmishTrainRuntimes.append(train_time)
    SmishValidRuntimes.append(valid_time)
    SmishTrainLoss.append(train_loss)
    SmishValidLoss.append(valid_loss)
SmishNet.load_state_dict(BestLearnedParameters)
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
BestLearnedParameters = LogishNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Logish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Logish")
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = LogishNet.state_dict()
    valid_end_time = time.time()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    LogishTrainAccuracies.append(train_accuracy)
    LogishValidAccuracies.append(valid_accuracy)
    LogishTrainRuntimes.append(train_time)
    LogishValidRuntimes.append(valid_time)
    LogishTrainLoss.append(train_loss)
    LogishValidLoss.append(valid_loss)
LogishNet.load_state_dict(BestLearnedParameters)
LogishTestAccuracy.append(test(_epoch, "Logish"))
print("test_acc:", LogishTestAccuracy)
print("Finished Logish")
'''

ReLUTrainAccuracies = []
ReLUValidAccuracies = []
ReLUTrainRuntimes = []
ReLUValidRuntimes = []
ReLUTrainLoss = []
ReLUValidLoss = []
ReLUTestAccuracy = []
TopValidAccuracy = 0
BestLearnedParameters = ReLUNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"ReLU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "ReLU")
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = ReLUNet.state_dict()
    valid_end_time = time.time()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    ReLUTrainAccuracies.append(train_accuracy)
    ReLUValidAccuracies.append(valid_accuracy)
    ReLUTrainRuntimes.append(train_time)
    ReLUValidRuntimes.append(valid_time)
    ReLUTrainLoss.append(train_loss)
    ReLUValidLoss.append(valid_loss)
ReLUNet.load_state_dict(BestLearnedParameters)
ReLUTestAccuracy.append(test(_epoch, "ReLU"))
print("test_acc:", ReLUTestAccuracy)
print("Finished ReLU")

'''
MishTrainAccuracies = []
MishValidAccuracies = []
MishTrainRuntimes = []
MishValidRuntimes = []
MishTrainLoss = []
MishValidLoss = []
MishTestAccuracy = []
TopValidAccuracy = 0
BestLearnedParameters = MishNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"Mish")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "Mish")
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = MishNet.state_dict()
    valid_end_time = time.time()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    MishTrainAccuracies.append(train_accuracy)
    MishValidAccuracies.append(valid_accuracy)
    MishTrainRuntimes.append(train_time)
    MishValidRuntimes.append(valid_time)
    MishTrainLoss.append(train_loss)
    MishValidLoss.append(valid_loss)
MishNet.load_state_dict(BestLearnedParameters)
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
BestLearnedParameters = SiLUNet.state_dict()
for _epoch in range(start_epoch, start_epoch + num_epochs):
    train_start_time = time.time()
    train_accuracy, train_loss = train(_epoch,"SiLU")
    inter_time = time.time()
    valid_accuracy, valid_loss = validate(_epoch, "SiLU")
    if valid_accuracy > TopValidAccuracy:
        TopValidAccuracy = valid_accuracy
        BestLearnedParameters = SiLUNet.state_dict()
    valid_end_time = time.time()
    train_time = inter_time - train_start_time
    valid_time = valid_end_time - inter_time
    print(f'[{_epoch}] train_acc: {train_accuracy:.3f}% -- valid_acc: {valid_accuracy:.3f}% -- train_time: {train_time:.3f}s -- valid_time: {valid_time:.3f}s -- train_loss: {train_loss:.3f} -- valid_loss: {valid_loss:.3f}')
    SiLUTrainAccuracies.append(train_accuracy)
    SiLUValidAccuracies.append(valid_accuracy)
    SiLUTrainRuntimes.append(train_time)
    SiLUValidRuntimes.append(valid_time)
    SiLUTrainLoss.append(train_loss)
    SiLUValidLoss.append(valid_loss)
SiLUNet.load_state_dict(BestLearnedParameters)
SiLUTestAccuracy.append(test(_epoch, "SiLU"))
print("test_acc:", SiLUTestAccuracy)
print("Finished SiLU")
'''
with open('ResNet_Tiny_Trial2.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, lineterminator="\n")
    wr.writerow(SishTrainAccuracies)
    wr.writerow(GELUTrainAccuracies)
    # wr.writerow(SmishTrainAccuracies)
    # wr.writerow(LogishTrainAccuracies)
    # wr.writerow(MishTrainAccuracies)
    # wr.writerow(SiLUTrainAccuracies)
    wr.writerow(ReLUTrainAccuracies)

    wr.writerow(SishValidAccuracies)
    wr.writerow(GELUValidAccuracies)
    # wr.writerow(SmishValidAccuracies)
    # wr.writerow(LogishValidAccuracies)
    # wr.writerow(MishValidAccuracies)
    # wr.writerow(SiLUValidAccuracies)
    wr.writerow(ReLUValidAccuracies)

    wr.writerow(SishTrainLoss)
    wr.writerow(GELUTrainLoss)
    # wr.writerow(SmishTrainLoss)
    # wr.writerow(LogishTrainLoss)
    # wr.writerow(MishTrainLoss)
    # wr.writerow(SiLUTrainLoss)
    wr.writerow(ReLUTrainLoss)

    wr.writerow(SishValidLoss)
    wr.writerow(GELUValidLoss)
    # wr.writerow(SmishValidLoss)
    # wr.writerow(LogishValidLoss)
    # wr.writerow(MishValidLoss)
    # wr.writerow(SiLUValidLoss)
    wr.writerow(ReLUValidLoss)

    wr.writerow(SishTrainRuntimes)
    wr.writerow(GELUTrainRuntimes)
    # wr.writerow(SmishTrainRuntimes)
    # wr.writerow(LogishTrainRuntimes)
    # wr.writerow(MishTrainRuntimes)
    # wr.writerow(SiLUTrainRuntimes)
    wr.writerow(ReLUTrainRuntimes)

    wr.writerow(SishValidRuntimes)
    wr.writerow(GELUValidRuntimes)
    # wr.writerow(SmishValidRuntimes)
    # wr.writerow(LogishValidRuntimes)
    # wr.writerow(MishValidRuntimes)
    # wr.writerow(SiLUValidRuntimes)
    wr.writerow(ReLUValidRuntimes)

    wr.writerow(SishTestAccuracy)
    wr.writerow(GELUTestAccuracy)
    # wr.writerow(SmishTestAccuracy)
    # wr.writerow(LogishTestAccuracy)
    wr.writerow(ReLUTestAccuracy)
    # wr.writerow(MishTestAccuracy)
    # wr.writerow(SiLUTestAccuracy)

print("SqueezeNet CIFAR10 trial complete")
