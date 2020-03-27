import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.nn import Parameter
def RoIPositionEmbedding_v1(box_a, box_b, img_size):

    A = box_a.size(0)
    B = box_b.size(0)
    cx_a = (box_a[:, 1] + box_a[:, 3]) * 0.5
    cy_a = (box_a[:, 0] + box_a[:, 2]) * 0.5
    cx_b = (box_b[:, 1] + box_b[:, 3]) * 0.5
    cy_b = (box_b[:, 0] + box_b[:, 2]) * 0.5

    w_a = torch.clamp((box_a[:, 3] - box_a[:, 1]),min=0.).unsqueeze(1).expand(A, B)
    h_a = torch.clamp((box_a[:, 2] - box_a[:, 0]),min=0.).unsqueeze(1).expand(A, B)

    w_b = torch.clamp((box_b[:, 3] - box_b[:, 1]),min=0.).unsqueeze(0).expand(A, B)
    h_b = torch.clamp((box_b[:, 2] - box_b[:, 0]),min=0.).unsqueeze(0).expand(A, B)


    cxa2b = (cx_a.unsqueeze(1).expand(A, B) - cx_b.unsqueeze(0).expand(A, B)) / (w_b + 1)
    cya2b = (cy_a.unsqueeze(1).expand(A, B) - cy_b.unsqueeze(0).expand(A, B)) / (h_b + 1)


    cxa2b = torch.log(torch.abs(cxa2b)+1e-9)
    cxa2b = torch.log(torch.abs(cxa2b)+1e-9)

    wa2b = torch.log((w_a+1) / (w_b+1))
    ha2b = torch.log((h_a+1) / (h_b+1))

    w_a /= (img_size[1] + 1)
    h_a /= (img_size[0] + 1)
    w_b /= (img_size[1] + 1)
    h_b /= (img_size[0] + 1)

    edge_boxes = torch.stack((w_a,h_a,w_b,h_b,cxa2b,cya2b,wa2b,ha2b),dim=-1)

    return edge_boxes

def RoIPositionEmbedding(roi, dim_g=64, wave_len=1000):

    #构造 relative geometric features
    x_min, y_min, x_max, y_max = torch.chunk(roi, 4, dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    #相对宽度 [128, 128, 1]
    delta_w = torch.log(w / w.view(1, -1))
    delta_h = torch.log(h / h.view(1, -1))
    size = delta_h.size()


    # 将其扩充为2维矩阵
    delta_x = delta_x.view(size[0], size[1], 1)
    delta_y = delta_y.view(size[0], size[1], 1)
    delta_w = delta_w.view(size[0], size[1], 1)
    delta_h = delta_h.view(size[0], size[1], 1)

    #[128, 128, 4]
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)

    feat_range = torch.arange(dim_g / 8).to(roi.device) #看要扩充多少倍
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(size[0], size[1], 4, -1)
    position_mat = 100. * position_mat # [128, 128, 4, 1]

    mul_mat = position_mat * dim_mat # [128, 128, 4, 8]
    sin_mat = torch.sin(mul_mat) # # [128, 128, 4, 8]
    cos_mat = torch.cos(mul_mat)
    embedding = torch.cat((sin_mat, cos_mat), -1) # [128, 128, 4, 16]

    return embedding.view(size[0], size[1], -1)  # [128, 128,64]

class StructureInferenceNet(nn.Module):
    def __init__(self, n_steps=2, n_inputs=4096, n_hidden_o=4096, n_hidden_e=4096):
        super(StructureInferenceNet, self).__init__()
        self.n_steps=n_steps
        self.n_inputs=n_inputs
        self.n_hidden_o=n_hidden_o
        self.n_hidden_e=n_hidden_e
        self.w_p = Parameter(torch.zeros(12, 1))
        self.w_v = Parameter(torch.zeros(n_inputs * 2, 1))
        torch.nn.init.xavier_uniform_(self.w_p.data)
        torch.nn.init.xavier_uniform_(self.w_v.data)
        self.e_cell = nn.GRUCell(n_inputs, n_hidden_e)
        self.o_cell = nn.GRUCell(n_inputs, n_hidden_o)

    def forward(self, ofe, ofo, ofs, n_boxes, iou):
        # n_boxes, train 128, test 256
        ofe[iou>0.6,:]=0

        fs = torch.cat(n_boxes * [ofs], 0)
        fs = fs.view(n_boxes, self.n_inputs)

        fe = ofe.view(n_boxes * n_boxes, 12)

        PE = F.relu(torch.matmul(fe, self.w_p).view(n_boxes, n_boxes))
        oinput = fs
        hi = ofo

        for t in range(self.n_steps):

            X = torch.cat(n_boxes * [hi], 0)
            X = X.view(n_boxes * n_boxes, self.n_inputs)
            Y = hi  # Y = fo: 128 * 4096

            # VE form 2:
            Y1 = torch.cat(n_boxes * [Y], 1)
            Y1 = Y1.view(n_boxes * n_boxes, self.n_inputs)

            Y2 = torch.cat(n_boxes * [Y], 0)
            Y2 = Y2.view(n_boxes * n_boxes, self.n_inputs)

            VE = torch.tanh((torch.matmul(torch.cat([Y1, Y2], 1), self.w_v).view(n_boxes, n_boxes)))

            E = PE * VE
            Z = F.softmax(E, dim=-1)  # edge relationships
            X = X.view(n_boxes, n_boxes, self.n_inputs)  # Nodes
            M = Z.view(n_boxes, n_boxes, 1) * X  # messages
            M,_ = torch.max(M, 1)  # intergated message

            einput  = M.view(n_boxes, self.n_inputs)

            hi1 = self.o_cell(input=oinput, hx=hi)

            hi2 = self.e_cell(input=einput, hx=hi)

            # meanpooling
            hi = torch.cat([hi1, hi2], 0)
            hi = hi.view(2, n_boxes, self.n_inputs)
            hi = torch.mean(hi, 0)

        return F.relu(hi,inplace=True)


class RelationGraphConvolution(nn.Module):
    def __init__(self, n_heads = 16, feat_dim=4096, geo_dim=64, fc_dim = 16, re_dim = 3, dim=(4096, 4096, 4096)):
        super(RelationGraphConvolution, self).__init__()
        self.fc_dim = fc_dim # 全连接的个数 等于 multi heads个数
        self.feat_dim = feat_dim # 特征维度
        self.group = n_heads # 有多少个multi heads
        self.dim = dim #(1024, 1024, 1024)
        self.geo_linear = nn.Linear(geo_dim, fc_dim)
        self.query = nn.Linear(feat_dim, dim[0])
        self.key = nn.Linear(feat_dim, dim[1])
        self.reduce_dim = nn.Linear(feat_dim, re_dim)
        #self.linear = nn.Linear(n_relations * self.feat_dim, dim[2], bias=False)
        for i in range(n_heads): # 有多少个 heads就有多少个最终的一维卷积形式
            self.add_module('conv_%d' % i, nn.Conv1d(self.feat_dim, dim[2] // n_heads, 1))

    def forward(self, roi_feat, batch_size, roi_pos=None):
        """ Attetion module with vectorized version
        Args:
            roi_feat: [num_rois, feat_dim]
            roi_pos: [num_rois, num_rois, emb_dim]
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
        Returns:
            output: [num_rois, output_dim]
        """

        dim_group = (self.dim[0] // self.group, self.dim[1] // self.group, self.dim[2] // self.group)
        # （1024,1024,1024）/16 =(64, 64, 64) # 每一个head分管多少个维度的channel

        if roi_pos is not None:
            # 位置信息的重组
            size = roi_pos.size()  # [num_rois, num_rois, emb_dim]
            # [num_rois * num_rois, emb_dim]
            roi_pos = roi_pos.view(size[0]*size[1], size[2])
            # geo_coef, [num_rois * num_rois, fc_dim]
            geo_coef = self.geo_linear(roi_pos)
            geo_coef = F.relu(geo_coef) # [128*128, 16]
            # aff_weight, [num_rois, num_rois, fc_dim]
            geo_coef = geo_coef.view(size[0], size[1], self.fc_dim) # [128, 128, 16]
            # aff_weight, [num_rois, fc_dim, num_rois]
            geo_coef = geo_coef.permute(0, 2, 1) # [128, 16, 128]

        # scale-dot self-attention
        # 将其分开用于不同的heads 进行attention
        q_data = self.query(roi_feat) #[bs*256, 1024]
        q_data_batch = q_data.view(batch_size, -1, self.group, dim_group[0]) # [bs, 256, 16, 64]
        q_data_batch = q_data_batch.permute(0, 2, 1, 3).contiguous().view(batch_size* self.group, -1, dim_group[0]) # [bs*16, 256, 64]
        k_data = self.key(roi_feat)
        k_data_batch = k_data.view(batch_size, -1, self.group, dim_group[0])
        k_data_batch = k_data_batch.permute(0, 2, 1, 3).contiguous().view(batch_size* self.group, -1, dim_group[0]) # [bs*16, 256, 64]

        v_data = roi_feat.clone() # 提取到的视觉特征[bs*256， 1024]

        # 进行矩阵的乘法操作
        #  [16, 128, 128] 得到视觉关联系数 不同proposal的特征之间进行点乘内积运算 其实可以化为矩阵运行，
        # 最终得到一个 [128, 128]大小的关联矩阵 [128*64]*[64, 128]
        vis_coef = torch.bmm(q_data_batch, k_data_batch.permute(0,2,1)) # [bs*16, 128, 128]

        # vis_coef, [group, num_rois, num_rois]
        vis_coef = (1.0 / math.sqrt(float(dim_group[1]))) * vis_coef
        vis_coef = vis_coef.view(batch_size, self.group,-1,vis_coef.size(-1)).permute(0, 2, 1, 3) # [bs, 128,16, 128]
        vis_coef = vis_coef.contiguous().view(-1, self.group,vis_coef.size(-1)) # [bs* 128,16, 128]

        assert self.fc_dim == self.group, 'fc_dim != group'

        # weighted_coef, [num_rois, fc_dim, num_rois]
        # 加权得到综合两种关系的关联系数
        if roi_pos is None:
            weighted_coef = vis_coef
        else:
            weighted_coef = torch.log(torch.clamp(geo_coef, min=1e-6)) + vis_coef
        # 进行softmax操作
        weighted_coef = F.softmax(weighted_coef, dim=2) # [bs* 128,16, 128]
        # [num_rois * fc_dim, num_rois]
        weighted_coef = weighted_coef.view(batch_size, weighted_coef.size(-1)*self.fc_dim, weighted_coef.size(-1)) #[bs*128*16, 128]
        # 大小维度为16的作用是，每一个head都需要一个不同的关系图
        # multiheads, [num_rois * fc_dim, feat_dim]
        # #相当于完成了邻接矩阵乘以特征的过程 对不同node的信息进行fuse 这样任意节点都会包括其他节点的信息
        # [bs,128*16, 128]*[bs, 128, 1024]
        multiheads = torch.bmm(weighted_coef, v_data.view(batch_size, -1, v_data.size(-1))) # [bs,16*128, 1024]
        # multiheads, [bs*128, 16 ,1024]
        multiheads = multiheads.view(-1, self.group, self.feat_dim)

        output = list()
        for i in range(self.group): # 每个抽头都有自己最终的W 采用1维卷积形式 一维卷积采用了stride方式 进行特征维度的缩减，便于多个head的结果拼接之后大小刚好为1024
            output.append(self._modules['conv_%d' % i](multiheads[:,i,:,None])) # add [128, 64, 1]

        # concatenate multi-head feature
        output = torch.cat(output, 1).squeeze_(2) # [bs*128, 1024]

        loss_gl = 0
        if self.training:
            loss_gl = torch.tensor(0.0).to(output.device)
        #     # f_data = v_data.view(batch_size, -1, v_data.size(-1)).detach()
        #     # f_data = f_data.unsqueeze(2).detach() #[bs, 128, 1024]
        #     # f_data_diff = f_data.view(batch_size, f_data.size(1), 1, f_data.size(2))- f_data.view(batch_size, 1, f_data.size(1), f_data.size(2))
        #     v_data = self.reduce_dim(v_data.detach()) #降维
        #     f_data_diff = v_data.view(batch_size, -1, 1, v_data.size(1)) - v_data.view(batch_size, 1, -1, v_data.size(1))
        #     f_data_diff = f_data_diff.norm(p=1,dim=3,keepdim=True)
        #     f_data_diff = f_data_diff/v_data.size(-1) # [bs, 128, 128, 1]
        #     # f_data_diff = f_data_diff.detach()
        #     weighted_coef = weighted_coef.view(batch_size, self.group, -1, weighted_coef.size(-1)).permute(0, 2,3,1) #[bs, 128, 128, 1]
        #     loss_gl = torch.mul(f_data_diff**2,weighted_coef).mean()
        #     loss_gl = loss_gl + 0.001*torch.norm(weighted_coef)


        # skip connection
        return roi_feat + output, loss_gl # 残差建构 skip connection



if __name__ == "__main__":
    NotImplemented