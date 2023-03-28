import torch
from torch import nn

class Dice(nn.Module):
    def __init__(self,input_dim,eps=1e-8):
        super(Dice,self).__init__()
        self.eps = eps
        self.bn = nn.BatchNorm1d(input_dim,eps=eps)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self,X):
        '''

        :param X: (batch_size,1)
        :return:
        '''
        ps = torch.sigmoid(self.bn(X))
        return ps * X + (1-ps)*self.alpha*X
