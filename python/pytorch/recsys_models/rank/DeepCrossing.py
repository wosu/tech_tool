'''
refer:https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf
'''

import torch
from torch import nn

from python.pytorch.layers.cross_layer import ResidualBlock


class DeepCrossing(nn.Module):
    def __init__(self,dis_feafield_size:list,feature_num,dis_fea_num,dnn_hidden_unit=[32,32],embedding_size=32):

        self.cate_feas_embedding_table = \
            nn.ModuleList([nn.Embedding(size, embedding_size) for size in dis_feafield_size])

        self.fea_num = feature_num
        self.dis_fea_num = dis_fea_num
        self.conti_fea_num = feature_num - dis_fea_num
        input_dims = self.dis_fea_num * embedding_size + self.conti_fea_num
        dnn_hidden_unit = [input_dims] + dnn_hidden_unit
        dnn_list = []
        for hidden_size in dnn_hidden_unit:
            dnn_list.append(ResidualBlock(input_dims,hidden_size))
        dnn_list.append(nn.Linear(input_dims,1))
        dnn_list.append(nn.Sigmoid())
        self.dnn = nn.Sequential(*dnn_list)

    def forward(self,X):
        cate_feas_embed_list = []
        for i in range(self.dis_fea_num):
            # 离散特征根据ont-hot id查找embedding表得到对于的embedding
            # fea_embeds:特征embedding,shape(batch_size,embedding_size)
            fea_embeds = self.cate_feas_embedding_table[i](X[:, i].long())
            cate_feas_embed_list.append(fea_embeds)
        inputs = torch.cat([torch.cat(cate_feas_embed_list, dim=1), X[:, self.dis_fea_num:]])
        self.dnn(inputs)