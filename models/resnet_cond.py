# coding=utf-8
from . import utils, layers, layerspp, normalization
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
get_act = layers.get_act

class resnet_cond(nn.Module):
  """Classifier model for conditional sampling"""
  def __init__(self, config, model_type='resnet18_cond'):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.nf = nf = config.classifier.nf
    modules = []
    modules.append(layerspp.GaussianFourierProjection(embedding_size=nf, scale=config.classifier.fourier_scale))
    modules.append(nn.Linear(nf * 2, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(nf * 4, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    self.condition_modules = nn.ModuleList(modules)
    if model_type == 'resnet18_cond':
        self.classifier = ResNet_cond(BasicBlock, [2, 2, 2, 2], num_classes=config.classifier.classes, 
                                        in_channels=config.data.num_channels, temb_dim=nf * 4)
    elif model_type == 'resnet34_cond':
        self.classifier = ResNet_cond(BasicBlock, [3, 4, 6, 3], num_classes=config.classifier.classes, 
                                    in_channels=config.data.num_channels, temb_dim=nf * 4)

  def forward(self, x, time_cond):
    # Generate time emb.
    modules = self.condition_modules
    m_idx = 0
    used_sigmas = time_cond
    temb = modules[m_idx](torch.log(used_sigmas))
    m_idx += 1
    temb = modules[m_idx](temb)
    m_idx += 1
    temb = modules[m_idx](self.act(temb))
    m_idx += 1
    
    out = self.classifier(x, temb)
    return out


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
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


class ResNet_cond(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels, temb_dim):
        super(ResNet_cond, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.dense_1 = nn.Linear(temb_dim, 64)
        self.dense_2 = nn.Linear(temb_dim, 128)
        self.dense_3 = nn.Linear(temb_dim, 256)
        self.dense_4 = nn.Linear(temb_dim, 512)
        self.act = nn.SiLU()
        
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

    def forward(self, x, temb):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        temb_1 = self.dense_1(self.act(temb))[:, :, None, None]
        out += temb_1
        out = self.layer2(out)
        temb_2 = self.dense_2(self.act(temb))[:, :, None, None]
        out += temb_2
        out = self.layer3(out)
        temb_3 = self.dense_3(self.act(temb))[:, :, None, None]
        out += temb_3
        out = self.layer4(out)
        temb_4 = self.dense_4(self.act(temb))[:, :, None, None]
        out += temb_4
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out