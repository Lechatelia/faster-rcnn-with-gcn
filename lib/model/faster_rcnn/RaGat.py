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
from model.faster_rcnn.faster_rcnn_RaGAT import _fasterRCNN
from model.faster_rcnn.ReGCN import Relation_Encoder, prepare_graph_variables
import pdb

class RaGat(_fasterRCNN):
  def __init__(self, classes, nongt_dim=128, pretrained=False, class_agnostic=False):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.nongt_dim = nongt_dim

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
    dim = 4096
    self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, dim, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5)
                                   )
    self.gat1 = Relation_Encoder(v_dim=dim, out_dim=dim,
                                 nongt_dim=self.nongt_dim,  dir_num=1, pos_emb_dim=64, num_steps=1)
    self.fc2 = nn.Sequential(nn.Linear(dim, dim, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout(p=0.5)
                                   )
    self.gat2 = Relation_Encoder(v_dim=dim, out_dim=dim,
                                 nongt_dim=self.nongt_dim, dir_num=1, pos_emb_dim=64, num_steps=1)

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    # self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(dim, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(dim, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(dim, 4 * self.n_classes)

  def _head_to_tail(self, pool5, rois):
    
    pool5_flat = pool5.view(pool5.size(0),pool5.size(1), -1)
    pos_emb = prepare_graph_variables(rois, nongt_dim=self.nongt_dim, device=rois.device)
    x = self.fc1(pool5_flat)
    x = self.gat1(x, pos_emb)
    x = self.fc2(x)
    x = self.gat2(x, pos_emb)
    return x

