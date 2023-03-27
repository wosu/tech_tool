import torch
from torch import nn

from python.pytorch.layers.mlp import MLP


class ESMM(nn.Module):
    def __init__(self,base_model:nn.Module,input_dims,task_weight=[1,1],dnn_hidden_units=[64,64]):
        self.ctr_net = base_model
        if self.ctr_net is None:
            self.ctr_net = MLP(input_dims,1,"sigmoid")

        self.cvr_net = base_model
        if self.cvr_net is None:
            self.cvr_net = MLP(input_dims, 1, "sigmoid")
        self.loss_func = nn.BCELoss()

        self.task_weight = task_weight

    def forward(self,X):
        # ctr_out:(batch_size,1)
        ctr_out = self.cvr_net(X)
        cvr_out = self.cvr_net(X)
        ctcvr_out = ctr_out * cvr_out
        return ctr_out,cvr_out,ctcvr_out

    def loss(self,Y_ctr,Y_ctrcvr,pred_ctr,pred_ctcvr):
        ctr_task_weight = self.task_weight[0]
        ctcvr_task_weight = self.task_weight[1]
        return self.loss_func(pred_ctr,Y_ctr)*ctr_task_weight + \
               self.loss_func(pred_ctcvr,Y_ctrcvr)*ctcvr_task_weight
