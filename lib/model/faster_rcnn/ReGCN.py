# @Time : 2020/3/14 18:16 
# @Author : Jinguo Zhu
# @File : ReGCN.py 
# @Software: PyCharm
'''
                    .::::.
                  .::::::::.
                 :::::::::::  I && YOU
             ..:::::::::::'
           '::::::::::::'
             .::::::::::
        '::::::::::::::..
             ..::::::::::::.
           ``::::::::::::::::
            ::::``:::::::::'        .:::.
           ::::'   ':::::'       .::::::::.
         .::::'      ::::     .:::::::'::::.
        .:::'       :::::  .:::::::::' ':::::.
       .::'        :::::.:::::::::'      ':::::.
      .::'         ::::::::::::::'         ``::::.
  ...:::           ::::::::::::'              ``::.
 ````':.          ':::::::::'                  ::::..
                    '.:::::'                    ':'````..
 '''

import torch
from model.faster_rcnn.graph_att import GAttNet as GAT
import torch.nn as nn
from model.faster_rcnn.fc import  FCNet
from torch.autograd import Variable


def torch_extract_position_embedding(position_mat, feat_dim, wave_length=1000,
                                     device=torch.device("cuda")):
    # position_mat, [batch_size,num_rois, nongt_dim, 4]
    feat_range = torch.arange(0, feat_dim / 8)
    dim_mat = torch.pow(torch.ones((1,))*wave_length,
                        (8. / feat_dim) * feat_range)
    dim_mat = dim_mat.view(1, 1, 1, -1).to(device)
    position_mat = torch.unsqueeze(100.0 * position_mat, dim=4)
    div_mat = torch.div(position_mat.to(device), dim_mat)
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [batch_size,num_rois, nongt_dim, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], -1)
    # embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    embedding = embedding.view(embedding.shape[0], embedding.shape[1],
                               embedding.shape[2], feat_dim)
    return embedding


def torch_extract_position_matrix(bbox, nongt_dim=36):
    """ Extract position matrix

    Args:
        bbox: [batch_size, num_boxes, 4]

    Returns:
        position_matrix: [batch_size, num_boxes, nongt_dim, 4]
    """

    xmin, ymin, xmax, ymax = torch.split(bbox, 1, dim=-1)
    # [batch_size,num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [batch_size,num_boxes, num_boxes]
    delta_x = center_x-torch.transpose(center_x, 1, 2)
    delta_x = torch.div(delta_x, bbox_width)

    delta_x = torch.abs(delta_x)
    threshold = 1e-3
    delta_x[delta_x < threshold] = threshold
    delta_x = torch.log(delta_x)
    delta_y = center_y-torch.transpose(center_y, 1, 2)
    delta_y = torch.div(delta_y, bbox_height)
    delta_y = torch.abs(delta_y)
    delta_y[delta_y < threshold] = threshold
    delta_y = torch.log(delta_y)
    delta_width = torch.div(bbox_width, torch.transpose(bbox_width, 1, 2))
    delta_width = torch.log(delta_width)
    delta_height = torch.div(bbox_height, torch.transpose(bbox_height, 1, 2))
    delta_height = torch.log(delta_height)
    concat_list = [delta_x, delta_y, delta_width, delta_height]
    for idx, sym in enumerate(concat_list):
        # [batch_size, nongt_dim, num_boxes]
        sym = sym[:, :nongt_dim]
        concat_list[idx] = torch.unsqueeze(sym, dim=3)
    position_matrix = torch.cat(concat_list, 3)
    return position_matrix


class Relation_Encoder(nn.Module):
    def __init__(self, v_dim, out_dim, nongt_dim, dir_num=1, pos_emb_dim=64,
                 num_heads=16, num_steps=1,
                 residual_connection=True, label_bias=True):
        # nongt_dim number of objects consider relations per image
        super(Relation_Encoder, self).__init__()
        self.v_dim = v_dim # 特征维度
        self.out_dim = out_dim # 输出维度
        self.residual_connection = residual_connection #是否采用残差结构
        self.num_steps = num_steps #进行多少次GCN
        print("In ImplicitRelationEncoder, num of graph propogate steps:",
              "%d, residual_connection: %s" % (self.num_steps,
                                               self.residual_connection))

        if self.v_dim != self.out_dim:
            self.v_transform = FCNet([v_dim, out_dim]) #如果维度不一致，就先改编维度
        else:
            self.v_transform = None
        self.implicit_relation = GAT(dir_num, 1, out_dim, out_dim,
                                     nongt_dim=nongt_dim,
                                     label_bias=label_bias,
                                     num_heads=num_heads,
                                     pos_emb_dim=pos_emb_dim)

    def forward(self, v, position_embedding):
        """
        Args:
            v: [batch_size, num_rois, v_dim]  视觉特征
            position_embedding: [batch_size, num_rois, nongt_dim, emb_dim]

        Returns:
            output: [batch_size, num_rois, out_dim,3]
        """
        # [batch_size, num_rois, num_rois, 1]
        # imp_adj_mat = Variable(torch.ones(v.size(0), v.size(1), v.size(1), 1)).to(v.device)
        # 全1矩阵 [batch_szie, 256, 256, 1]     其实并没有什么意义 所以这里改成 by zjg 20200314
        imp_adj_mat = None
        
        imp_v = self.v_transform(v) if self.v_transform else v  #先检查特征维度

        for i in range(self.num_steps):
            imp_v_rel = self.implicit_relation.forward(imp_v,
                                                       imp_adj_mat,
                                                       position_embedding)
            if self.residual_connection:
                imp_v += imp_v_rel
            else:
                imp_v = imp_v_rel
        return imp_v

def prepare_graph_variables( bb, nongt_dim, device, pos_emb_dim=64):
    # bbox: [batch_size, num_boxes, 4]
    # pos_emd_dim position_embedding_dim:

    bb = bb.to(device) # [batch_size, num_boxes, 4]
    pos_mat = torch_extract_position_matrix(bb, nongt_dim=nongt_dim) # [batch_size, num_boxes, nongt_dim, 4]
    pos_emb = torch_extract_position_embedding( pos_mat, feat_dim=pos_emb_dim, device=device)
    # position embedding, [batch_size,num_rois, nongt_dim, feat_dim]
    pos_emb_var = Variable(pos_emb).to(device)
    return pos_emb_var


def test_GCN():
    print('CUDA available: {}'.format(torch.cuda.is_available()))
    print('the available CUDA number is : {}'.format(torch.cuda.device_count()))
    nongt_dim = 128
    rois = torch.randn(5, 256, 4).cuda()
    pooled_feat = torch.randn(5, 256, 49).cuda()
    relation_encode = Relation_Encoder(v_dim=49, out_dim=1024, nongt_dim=nongt_dim,  dir_num=1, pos_emb_dim=64, num_steps=3 )
    relation_encode = nn.DataParallel( relation_encode.cuda())
    pos_emb = prepare_graph_variables(rois, nongt_dim=nongt_dim, device=rois.device)
    realtion_feat = relation_encode(pooled_feat, pos_emb)
    print(realtion_feat.size())



if __name__ == "__main__":
    test_GCN()