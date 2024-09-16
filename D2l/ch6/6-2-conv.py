"""
卷积计算（互相关）
"""

import torch
from torch import nn
from d2l import torch as d2l

def corr2d(x, k):
    h,w = k.shape
    y = torch.zeros((x.shape[0] - h + 1),(x.shape[1] - w + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = (x[i:i+h, j:j+w] * k).sum()
    return y


class Conv2d(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2d, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        # self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        return corr2d(x, self.weight)


X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
result = corr2d(X, K)

print(result)


x = torch.ones((6, 8))
x[:,2:6] = 0
print(x)

k = torch.tensor([[1,-1]])

y1 = corr2d(x,k)
y2 = corr2d(x.t(),k)
# print(y1,'\n',y2)
con2d = nn.Conv2d(1,1,kernel_size=(1,2),bias=False)
# con2d = Conv2d(kernel_size=(1,2))
x = x.reshape(1,1,6,8)
y = y1.reshape(1,1,6,7)
# print(y)
lr = 3e-2
for i in range(10):
    y_hat = con2d(x)
    loss = (y-y_hat)**2
    con2d.zero_grad()
    loss.sum().backward()
    con2d.weight.data -= lr*con2d.weight.grad

    print("epoch {} , loss:{:.4f}".format(i+1,loss.sum()))

print(con2d.weight.data)

