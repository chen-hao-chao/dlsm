# coding=utf-8
from . import utils, resnet_cond
import torch.nn as nn

@utils.register_model(name='classifier')
class Classifier(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.classifier = resnet_cond.resnet_cond(config, model_type=config.classifier.model)
  def forward(self, x, cond):
    x = self.classifier(x, cond)
    return x