import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN

from model.roi_layers import ROIAlign, ROIPool

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg

from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        # self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        # self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data #[bs] 代表每张图片的目标个数

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data) # [bs, 512, w, h]

        # feed base feature map tp RPN to obtain rois
        # RPN网络计算得到rois 和 对应的损失函数 rois [x1 y1 x2 y2]
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        # rois =  [batch_size, post_nms_topN=2000, 5]
        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # 将rois送进rcnn 并且为roi分配分类标签和回归目标
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            # rois [bs, 256, 5] rois 最后一个维度的最后一个数代表的是batch中图片序号
            # roi label [bs, 256] taget [bs, 256, 4]
            # rois_inside_ws  [bs, 256, 4]
            # rois_outside_ws [bs, 256, 4]

            rois_label = Variable(rois_label.view(-1).long()) #[bs*256]
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))   #[bs*256, 4]
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))  #[bs*256, 4]
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2))) #[bs*256, 4]
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois) #[bs, 256, 5]
        # do roi pooling based on predicted rois

        #每张图都根据那256个roi去在原特征图上面去找特征ROI pooling
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))
        # base_feat [bs, 512, h, w] rois [bs, 256, 5]
        # pooled_feat [bs*256, 512, 7, 7]
        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat) # [bs*256, 4096]

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)  # [bs*256, 4*(classes+1)=324]
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4) #[# [bs*256, classes+1, 4]]
            #根据相应的label去挑选相应的输出 哪一个类别的输出有效
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            #
            bbox_pred = bbox_pred_select.squeeze(1) #[bs*256, 4]

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat) #[bs*256, classes+1]
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        # rois [4, 256, 5]  cls_prob [4, 256, 81] bbox_pred[4, 256, 4] rois_label [4, 256]
        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
