# coding=utf-8
from . import utils
import torch.nn as nn
import torch
from torchvision.models import resnet50

class classifier_cas(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.classifier = resnet50(num_classes=config.classifier.classes)
    self.classifier.conv1 = nn.Conv2d(config.data.num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  def forward(self, x):
    x = self.classifier(x)
    return x