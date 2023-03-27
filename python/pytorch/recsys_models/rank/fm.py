import torch
from torch import nn

from python.pytorch.utils.model_util import model_visual


class FM(nn.Module):
    def __init__(self,dis_feafield_size:list,feature_num,dis_fea_num,embedding_size,if_cross_conti:bool=True,device='cpu'):
        super(FM,self).__init__()
        self.conti_feature_num = feature_num-dis_fea_num
        self.dis_feature_num = dis_fea_num

        # 按照特征域创建一阶特征的权重参数embedding_table
        self.linear_embeds_table = nn.ModuleList()
        for fea_size in dis_feafield_size:
            self.linear_embeds_table.append(nn.Embedding(fea_size, 1, scale_grad_by_freq=True))

        # 按照特征域创建二阶特征embedding矩阵
        self.fm_embeds_table = nn.ModuleList()
        # if not if_cross_conti:
        #     self.fm_embeds_table = [nn.Embedding(fea_size,embedding_size,scale_grad_by_freq=True) for fea_size in dis_feafield_size]
        # else:

        for fea_size in dis_feafield_size:
            self.fm_embeds_table.append(nn.Embedding(fea_size, embedding_size, scale_grad_by_freq=True))
        for i in range(self.conti_feature_num):
            self.fm_embeds_table.append(nn.Embedding(1, embedding_size, scale_grad_by_freq=True))

        self.activation = nn.Sigmoid()
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)

    def forward(self,X):
        # X shape:(batch_size,feature_num),默认开始是离散特征，后面是连续特征
        # 获取离散特征部分 dis_feas shape(batch_size,dis_feature_num)
        print("dis_feature_num",self.dis_feature_num)

        dis_feas = X[:,:self.dis_feature_num]
        # 获取连续特征部分 conti_feas shape(batch_size,conti_fea_num)
        conti_feas = X[:,self.dis_feature_num:]

        # 遍历所有离散特征域，得到样本中每个特征的index,
        # 查找embedding table，得到特征的embedding vector
        linear_weights = []
        fm_embed_vecs = []
        for i in range(self.dis_feature_num):
            # fea_batch_idx shape:(1,batch_size)
            fea_batch_idx = dis_feas[:,i].long()
            # 查找一阶embedding table,shape:(batch_size,1)
            linear_weights.append(self.linear_embeds_table[i].forward(fea_batch_idx))
            # 查找二阶embedding table shape:(batch_size,embed_size),是用特征指
            fm_embed_vecs.append(self.fm_embeds_table[i].
                                 forward(fea_batch_idx)*dis_feas[:,i].unsqueeze(dim=1) )
        print("fm_embed_vecs:",fm_embed_vecs)
        # 遍历所有的连续特征，获取连续特征二阶embedding向量
        for i in range(self.conti_feature_num):
            fm_embed_vecs.append(self.fm_embeds_table[i + self.dis_feature_num].
                                 forward(torch.zeros_like(conti_feas[:,i]).long()) * conti_feas[:,i].unsqueeze(dim=1) )
        # 一阶部分
        conti_feas = X[:,self.dis_feature_num:]
        linear_weights.append(conti_feas)
        # torch.cat(linear_weights,dim=1):(batch_size,fea_num)
        print("torch.cat(linear_weights,dim=1)",torch.cat(linear_weights,dim=1).shape)
        linea_part = torch.sum(torch.cat(linear_weights,dim=1),dim=1,keepdim=True)

        # 二阶部分,fm_embd_vecs shape:(batch_size,field_size,embedding_size)
        fm_embed_vecs = torch.stack(fm_embed_vecs,dim=1)
        # square_of_sum:(batch_size,1,embedding_size) sum_of_square:(batch_size,1,embedding_size)
        square_of_sum = torch.pow(torch.sum(fm_embed_vecs,dim=1,keepdim=True),2)
        sum_of_square = torch.sum(fm_embed_vecs*fm_embed_vecs,dim=1,keepdim=True)
        # cross_term:(batch_size,1)
        cross_term = torch.sum(square_of_sum - sum_of_square,dim=2,keepdim=False)*0.5
        z = linea_part + cross_term
        out = self.activation(z)
        return out

    def reset_parameters(self):
        def reset_param(m):
            if type(m) == nn.ModuleDict:
                for k, v in m.items():
                    if type(v) == nn.Embedding:
                        if "pretrained_emb" in self._feature_map.feature_specs[k]: # skip pretrained
                            continue
                        if self._embedding_initializer is not None:
                            try:
                                if v.padding_idx is not None:
                                    # the last index is padding_idx
                                    initializer = self._embedding_initializer.replace("(", "(v.weight[0:-1, :],")
                                else:
                                    initializer = self._embedding_initializer.replace("(", "(v.weight,")
                                eval(initializer)
                            except:
                                raise NotImplementedError("embedding_initializer={} is not supported."\
                                                          .format(self._embedding_initializer))
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(reset_param)


if __name__ == "__main__":
    dis_fea_num = 6
    conti_num = 2

    fea_num = dis_fea_num + conti_num
    dis_feas_size = [i+2 for i in range(dis_fea_num)]
    fm = FM(dis_feafield_size=dis_feas_size,feature_num=fea_num,dis_fea_num=dis_fea_num,embedding_size=5)
    X1 = [i+1 for i in range(dis_fea_num)]
    X2 = [i for i in range(dis_fea_num)]
    X1.extend([i+0.5 for i in range(conti_num)])
    X2.extend([i+0.1 for i in range(conti_num)])
    fm.forward(torch.tensor([X1,X2],requires_grad=True))
    Y = [[1,1]]
    model_visual(fm,"fm_model",torch.tensor([X1,X2]),"../model_result/")