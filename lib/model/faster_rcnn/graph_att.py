"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Relation-aware Graph Attention Network for Visual Question Answering
Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu
https://arxiv.org/abs/1903.12314

This code is written by Linjie Li.
"""
import torch
import torch.nn as nn
from model.faster_rcnn.fc import FCNet
from model.faster_rcnn.graph_att_layer import GraphSelfAttentionLayer


class GAttNet(nn.Module):
    def __init__(self, dir_num, label_num, in_feat_dim, out_feat_dim,
                 nongt_dim=20, dropout=0.5, label_bias=True,
                 num_heads=16, pos_emb_dim=-1):
        """ Attetion module with vectorized version

        Args:
            label_num: numer of edge labels
            dir_num: number of edge directions
            feat_dim: dimension of roi_feat
            pos_emb_dim: dimension of postion embedding for implicit relation, set as -1 for explicit relation

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GAttNet, self).__init__()
        assert dir_num <= 2, "Got more than two directions in a graph."
        self.dir_num = dir_num #图结构的方向数
        self.label_num = label_num # numer of edge labels 对spatial edge 和 semantic edge进行了类别的划分
        self.in_feat_dim = in_feat_dim  # dimension of input roi_feat
        self.out_feat_dim = out_feat_dim # dimension of output roi_feat
        self.dropout = nn.Dropout(dropout)
        self.self_weights = FCNet([in_feat_dim, out_feat_dim], '', dropout)
        self.label_num = label_num
        if label_num >1:
            self.bias = FCNet([label_num, 1], '', 0, label_bias)
        else:
            self.bias = None
        self.nongt_dim = nongt_dim
        self.pos_emb_dim = pos_emb_dim # position embedding 的维度大小
        neighbor_net = []
        for i in range(dir_num): # 对每一个方向， 建立一个不同的自注意力机制的图结构
            g_att_layer = GraphSelfAttentionLayer(
                                pos_emb_dim=pos_emb_dim,
                                num_heads=num_heads,
                                feat_dim=out_feat_dim,
                                nongt_dim=nongt_dim)
            neighbor_net.append(g_att_layer)
        self.neighbor_net = nn.ModuleList(neighbor_net)

    def forward(self, v_feat, adj_matrix=None, pos_emb=None):
        """
        Args:
            v_feat: [batch_size,num_rois, feat_dim]
            adj_matrix: [batch_size, num_rois, num_rois, num_labels]
            pos_emb: [batch_size, num_rois, pos_emb_dim]

        Returns:
            output: [batch_size, num_rois, feat_dim]
        """
        if self.pos_emb_dim > 0 and pos_emb is None:
            raise ValueError(
                f"position embedding is set to None "
                f"with pos_emb_dim {self.pos_emb_dim}")
        elif self.pos_emb_dim < 0 and pos_emb is not None:
            raise ValueError(
                f"position embedding is NOT None "
                f"with pos_emb_dim < 0")
        batch_size, num_rois, feat_dim = v_feat.shape
        nongt_dim = self.nongt_dim

        if adj_matrix is not None:
            adj_matrix = adj_matrix.float() #  [batch_size, num_rois, num_rois, 1]

            adj_matrix_list = [adj_matrix, adj_matrix.transpose(1, 2)] # 交换一下主次关系， 如果是有两个方向的图结构的话

        # Self - looping edges
        # [batch_size,num_rois, out_feat_dim]
        self_feat = self.self_weights(v_feat) # 相当于GCN当中的W矩阵

        output = self_feat
        neighbor_emb = [0] * self.dir_num
        for d in range(self.dir_num): #对于每个方向的图结构进行运算
            # [batch_size,num_rois, nongt_dim, label_num] --》[ba, 256, 128, 1]
            if adj_matrix is not None:
                input_adj_matrix = adj_matrix_list[d][:, :, :nongt_dim, :]
                if self.label_num > 1:
                    condensed_adj_matrix = torch.sum(input_adj_matrix, dim=-1) ## 在edge label维度上面进行求和 [batch_size,num_rois, nongt_dim]

                    # # [batch_size,num_rois, nongt_dim, label_num]--》 [batch_size,num_rois, nongt_dim]
                    v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1) #在label维度上面进行全连接， # [batch_size,num_rois, nongt_dim, 1]
                else:
                    condensed_adj_matrix = input_adj_matrix.squeeze(-1)
                    v_biases_neighbors = condensed_adj_matrix
                    # [batch_size,num_rois, out_feat_dim]
                neighbor_emb[d] = self.neighbor_net[d].forward(
                    self_feat, condensed_adj_matrix, pos_emb,v_biases_neighbors)

            else:
                neighbor_emb[d] = self.neighbor_net[d].forward(
                    self_feat, None, pos_emb, None)

            # [batch_size,num_rois, out_feat_dim]
            output = output + neighbor_emb[d] #聚合来的特征加上原来的特征

            #这样的图卷积公式相当于从 A*X*W  ----》》》 (1+A)*X*W
        output = self.dropout(output)
        output = nn.functional.relu(output)

        return output
