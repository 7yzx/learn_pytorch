import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self,in_features,out_features):
        super(MyLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features,out_features))
        self.bias = nn.Parameter(torch.randn(out_features,))

    def forward(self,x):
        return torch.matmul(x,self.weight.data)+self.bias.data