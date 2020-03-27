from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride
        self._scales = scales
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        self._num_anchors = self._anchors.size(0)

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0] #[bs, self.nc_score_out=12*2, H, W ]
        gt_boxes = input[1] #[bs, 50, 5]
        im_info = input[2] # [H, W, scale]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3) #特征层的height and width

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        #构建二维坐标矩阵
        shift_x = np.arange(0, feat_width) * self._feat_stride # 每个点的x坐标 一维
        shift_y = np.arange(0, feat_height) * self._feat_stride #y坐标 一维
        shift_x, shift_y = np.meshgrid(shift_x, shift_y) #返回坐标矩阵 两个[feat_width, feat_height]的矩阵，分别代表x和y坐标
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose()) # 先将二维的坐标变为线性的（ravel），然后将其堆叠起来
        #构成anchor的shift偏移值，因为一个anchor需要2个点，四个坐标值，就是相当于 x y x y
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()  #[H*W, 4]

        A = self._num_anchors #12 某一个位置的anchor数量
        K = shifts.size(0) # [H*W] 这里的hw都是预测特征的长宽，是下采样16倍之后的的

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4) #[H*W, A=12, 4]
        all_anchors = all_anchors.view(K * A, 4) #[H*W*12, 4]

        total_anchors = int(K * A) #同一张特征图上面的anchor数量
        #通过位置的选择讲那些区域在图像外面的anchors给去掉
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1) #那些在特征图范围内的anchor的序号indicators

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :] #取出那些anchor [num_inside_anchor, 4]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1) #[bs, num_inside_anchors]
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()  #[bs, num_inside_anchors]
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()  #[bs, num_inside_anchors]

        #calculate the IOU [num_inside_anchor, 4] [bs, 50, 5]
        overlaps = bbox_overlaps_batch(anchors, gt_boxes) #[bs, num_anchors, num_gts]

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2) # 为每一个anchor计算找到最符合自己的gt，及其IOU [bs, num_anchors]
        gt_max_overlaps, _ = torch.max(overlaps, 1) #为每一个gt找到最符合自己的anchor，及其IOU [bs, num_gts]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5 #补位的gt算出来的IOU都是0，将置为非零的数，那么剩下的都是真实的gt
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2) #[bs, num_anchor]
        #利用每个gt与任一IOU的最大值，然后与这个gt的IOU等于这个最大值的anchors找出来 在gt维度上面求和 说明一个anchor可以同时是几个gt的最佳

        if torch.sum(keep) > 0:
            labels[keep>0] = 1 #最佳anchor无论IOU都标记为正样本

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1 #大于阈值的都标记为正样本

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 #小于阈值的标记为负样本

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE) # 规定的fg数量 128
        #label [bs, num_anchors]
        sum_fg = torch.sum((labels == 1).int(), 1) # positive for each images [bs]
        sum_bg = torch.sum((labels == 0).int(), 1) # negative [bs]

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg: #大于规定的数目
                fg_inds = torch.nonzero(labels[i] == 1).view(-1) #那些正样本的序号
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long() #随机排列
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]] #丢弃正样本的序号
                labels[i][disable_inds] = -1 #置为-1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            #如果正样本不够的128，就负样本多一些，共同为256
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many 一般来说这部分都会很多
            if sum_bg[i] > num_bg: #fg数目大于规定数目
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1 #丢弃的负样本置为-1

        offset = torch.arange(0, batch_size)*gt_boxes.size(1) #大小即为[bs]  data=[0, num_gt, 2*num_gt,...(bs-1)*num_gt]

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps) #相当于为给gt编号，区分不同bs的gt
        #第一个样本的gt为0~49
        #第二个为50~99

        #计算回归系数
        #这里应该有个问题，这样计算的回归系数总是按照每个anchor靠的最近的gt去计算回归系数，但是没有考虑到某些特殊情况下，
        #负责某个gt的anchor他的最大IOU所对应gt并不是他，但是应该问题不大，因为实际上就应该让anchor去预测离自己最近的gt
        #只不过那些没有靠的近的anchor的 gt 可以在低阈值上面选择anchor
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))

        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0) #正负样本总和一般是256
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights #1/256
        bbox_outside_weights[labels == 0] = negative_weights #1/256


        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1) #[bs,num_inside_anchor]
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0) #[bs,num_inside,anchor,4]
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0) #[bs,num_inside,anchor,4]
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0) #[bs,num_inside,anchor]

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)
        #return label targets inside_weights outside_weights
        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
