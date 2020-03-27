
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    self._roidb = roidb
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)

    # given the ratio_list, we want to make the ratio same for each batch.
    # 注意这部分的调整使得最终网络输入的每个batch数据的ratio是一致的
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_() #最终一个batch中的ratio如何取
    num_batch = int(np.ceil(len(ratio_index) / batch_size)) #batch的数量 注意向上取整，这样能够包括最后那几个落单的
    for i in range(num_batch): #对每一个batch的图片取最终的ratio
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        if ratio_list[right_idx] < 1:
            # 如果在宽高比例列表的升序排列中，最大的宽高比例值都小于1，说明这一个连续的batch size都小于1
            # 如果原始的输入图像是瘦瘦高高的，那就让它更加瘦瘦高高
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]


        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            # 如果原始的输入图像是矮矮胖胖的，那就让它更加矮矮胖胖
            # 如果在宽高比例列表的升序排列中，最小的宽高比例值都大于1，说明这一个连续的batch size都大于1
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1
        '''
        这样得到的torch.tensor中每连续batch_size个数值都是相同的，
        表示在同一个batch size中的图像设置怎样的宽高比（aspect  ratios）
        再反观Sampler迭代器，每次产生的列表都是随机，长度为bs的连续，
        且第一个数值必然是bs的整数倍
        就是说能够现在需要保证同一个batch size中的目标宽高比一致，而另外保证短边为600
        这样就保证了同一个batch的输入图像的分辨率是完全相同的
        '''
        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio #最终一个batch中规定的ratio


  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index]) #如果在训练中就按照长宽比的顺序依次取
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]] # 这里其实是只含有一个roidb字典的列表
    blobs = get_minibatch(minibatch_db, self._num_classes)
    #因为实际上输入的只是一个随机序号，所以这个minibatch的bs只是1
    #将roi转化为大小为1的batch数据
    # bolb是一个字典，依次为 图片数据、gt_boxes、图片信息（长宽和缩放scale尺寸-->使得最短边为600）、image——id
    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    if self.training:
        np.random.shuffle(blobs['gt_boxes']) # 将boxes打乱
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index] #注意取这个ratio能够保证最终的ratio一致

        #当图片本身的ratio超过了最大值和最小值时，需要先做一下ratio
        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio)) #这个ratio是batch中公用ed
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)
        #刚刚只是解决了需要crop的图片，图片还不一定满足ratio的要求

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            # 如果当前图像的width/height<1   则它的目标ratio会更小，说明要对高度进行padding
            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()
            # 对高度进行padding，将输入图像放在上面，不用对gt_boxes坐标变换，也就是补0是在图片最下面
            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            # 目标宽高比width/heigth>1 说明原始的输入图像是矮矮胖胖的，则它的目标ratio会更大
            # 为了让它变得更加矮矮胖胖，就填充宽度，将原始图像paste到左边，补0补在最右边
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size

        #这里需要注意的是无论是crop还是将ratio修改为batch中一致的ratio，这里都没有采用resize的方式
        #crop是直接从图中抠
        #修改成batch_ratio是直接添加0
        #所以图片的缩放因子这里都不变，所以整个过程只做了一次resize操作
        #保证同一个batch_size中的输入图像宽高比相同（batch——ratio），同时最短边等于600
        #这样就保证了同一个batch的输入图像的分辨率是完全相同的
        #就不用再重新书写collate_fn函数组件一个batch

        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_() #[50, 5]用来存放标记
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing [H, w, 3]->[3, H, W]
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)
        # 训练数据的返回  im_info含有最终经过crop和padding之后的长宽以及第一次resize的scale因子
        # （之后并没有缩放，只是采用的补0或者crop的方式）
        # gt_boxes_padding [50, 5] num_boxes 就是一个int
        return padding_data, im_info, gt_boxes_padding, num_boxes
    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

  def __len__(self):
    return len(self._roidb)
