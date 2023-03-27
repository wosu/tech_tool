import torch
from torch import nn

from python.pytorch.layers.mlp import MLP
from python.pytorch.recsys_models.rank.lr import LR


class WideDeep(nn.Module):
    def __init__(self,dis_feas_size:list,dis_fea_num:int,fea_num,embedding_size=10,
                 dnn_hidden_units=[64,64],activation='relu',dropout_rate=0.5):
        super(WideDeep,self).__init__()
        self.conti_fea_num = fea_num-dis_fea_num
        ## TODO(wide侧指定特征交叉待实现)
        self.wide = LR(dis_feas_size,feature_num=fea_num,dis_fea_num=dis_fea_num)
        self.input_dims = dis_feas_size*embedding_size + self.conti_fea_num
        self.dnn = MLP(input_dims=self.input_dims,out_dims=1,activation=activation,
                       output_activation="sigmoid",hidden_units=dnn_hidden_units,dropout_rate=dropout_rate)
        # TODO(只考虑最简单的二分类,其他待实现)
        self.out_activation = nn.Sigmoid()


    def forward(self,X):
        # wide_part:(batch_size,1) deep_part:(batch_size,1)
        wide_part = self.wide(X)
        deep_part = self.dnn(X)
        y_hat = self.out_activation(wide_part + deep_part)
        return y_hat

def train():
    pass
if __name__ == "__main__":
    pass

