# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn_new import _fasterRCNN
from model.faster_rcnn.rgc import *
import pdb

class gcn(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
      self.fc_1 = nn.Linear(512 * 7 * 7, 4096, bias=True)
      self.fc_2 = nn.Linear(4096, 4096, bias=True)
      self.rgc1 = RelationGraphConvolution()
      self.rgc2 = RelationGraphConvolution()
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

  def _head_to_tail(self, pool5, batch_size):

    loss_gl = 0
    pool5_flat = pool5.view(pool5.size(0), -1)  #[bs*128, 512*7*7]
    out = self.fc_1(pool5_flat)  #[bs*128, 1024]
    out, loss_gl1 = self.rgc1(out, batch_size=batch_size)  #
    out = F.relu(out, inplace=True)
    loss_gl += loss_gl1

    out = self.fc_2(out)  # [128, 1024]
    out,loss_gl2 = self.rgc2(out, batch_size=batch_size)  #
    out = F.relu(out, inplace=True)
    loss_gl += loss_gl2

    return out, loss_gl


