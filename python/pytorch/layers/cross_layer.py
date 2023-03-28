from torch import nn
import torch


class DcnCrossLayer(nn.Module):

    def __init__(self, input_dims):
        super(DcnCrossLayer, self).__init__()

        self.input_dims = input_dims
        self.last_out_weights = nn.Linear(input_dims, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(input_dims))

    def forward(self, X0, X_l):
        # (bs,1)
        return self.last_out_weights(X_l) * X0 + self.bias + X_l


class ResidualBlock(nn.Module):
    def __init__(self,input_dim,hidden_dim,if_batch_norm=True):
        self.linear_trans_1 = nn.Linear(input_dim,hidden_dim)
        self.linear_trans_2 = nn.Linear(hidden_dim,input_dim)
        self.if_batch_norm = if_batch_norm
        self.cross_layer = nn.Sequential(self.linear_trans_1,nn.ReLU(),self.linear_trans_2)
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.activation = nn.ReLU()

    def forward(self,X):
        z = self.cross_layer(X) + X
        if self.if_batch_norm:
            z = self.batch_norm()
        self.activation(z)
