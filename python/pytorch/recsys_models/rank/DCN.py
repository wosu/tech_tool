'''
wide侧：交叉网络
deep侧：常规的DNN
DCN实现
交叉网络：
需要定义网络中交叉层的层数
每层的交叉层的输入包括
最开始输入的特征，会参与后面每一层的交叉，借鉴残差思想
上一层交叉层的输出
每层的输出维度相同维度大小相同，每层的参数量为2d,d为输入维度的大小
deep侧：正常的DNN结构
将deep侧和cross侧输出concat到一起，deep输出维度和cross侧输出维度大小可以不同
经过sigmoid激活输出，计算损失
loss：采用交叉熵损失，同时在损失中增加网络参数的正则
模型参数
优化器：adam
batch_normalization
使用梯度裁剪，torch.nn.utils.clip_grad_norm_  gradient clip norm
 was set at 100.
正则化：使用early-stop
超参优化：grid-search:grid search over
the number of hidden layers, hidden layer size, initial learning rate
 and number of cross layers.
e number of hidden layers ranged
 from 2 to 5, with hidden layer sizes from 32 to 1024
For DCN, the
 number of cross layers3 is from 1 to 6. e initial learning rate4
 was tuned from 0.0001 to 0.001 with increments of 0.0001. All experiments
 applied early stopping at training step 150,000, beyond
 which overfiing started to occur.
'''
import torch
from torch import nn

from python.pytorch.layers.cross_layer import DcnCrossLayer
from python.pytorch.layers.mlp import MLP


class DCN(nn.Module):
    def __init__(self,dis_feafield_size:list,feature_num,dis_fea_num,num_cross_num=6,
                 deep_out_dims=4,hidden_units=[32,32],embedding_size=32):
        # 创建离散特征域embedding table
        self.cate_feas_embedding_table = \
            nn.ModuleList([nn.Embedding(size,embedding_size) for size in dis_feafield_size])
        self.fea_num = feature_num
        self.dis_fea_num = dis_fea_num
        self.conti_fea_num = feature_num-dis_fea_num
        input_dims = self.dis_fea_num*embedding_size + self.conti_fea_num
        self.deep_part_net = MLP(input_dims,deep_out_dims,None,hidden_units)

        # 交叉层堆叠
        self.cross_net = nn.ModuleList([DcnCrossLayer(input_dims) for _ in range(num_cross_num)])
        final_dim = deep_out_dims + input_dims
        self.final_linear = nn.Linear(final_dim,1)


    def forward(self,X):
        cate_feas_embed_list = []
        for i in range(self.dis_fea_num):
            # 离散特征根据ont-hot id查找embedding表得到对于的embedding
            # fea_embeds:特征embedding,shape(batch_size,embedding_size)
            fea_embeds = self.cate_feas_embedding_table[i](X[:,i].long())
            cate_feas_embed_list.append(fea_embeds)
        inputs = torch.cat([torch.cat(cate_feas_embed_list,dim=1),X[:,self.dis_fea_num:]])
        # deep_out:(bs,deep_out_dims)
        deep_out = self.deep_part_net(inputs)
        # 交叉网络层计算
        X_i = inputs
        for cross_layer in self.cross_net:
            X_i = cross_layer(inputs,X_i)
        # 将cross侧和deep侧输出进行拼接，接入到linear变换，在输入到sigmoid
        z = self.final_linear(torch.cat([X_i,deep_out],dim=1))
        return torch.sigmoid(z)




# def add_regularization(self):
#     reg_loss = 0
#     if self._embedding_regularizer or self._net_regularizer:
#         emb_reg = get_regularizer(self._embedding_regularizer)
#         net_reg = get_regularizer(self._net_regularizer)
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 if "embedding_layer" in name:
#                     if self._embedding_regularizer:
#                         for emb_p, emb_lambda in emb_reg:
#                             reg_loss += (emb_lambda / emb_p) * torch.norm(param, emb_p) ** emb_p
#                 else:
#                     if self._net_regularizer:
#                         for net_p, net_lambda in net_reg:
#                             reg_loss += (net_lambda / net_p) * torch.norm(param, net_p) ** net_p
#     return reg_loss
