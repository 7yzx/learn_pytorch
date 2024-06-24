import torch
from torch import nn
from torch.nn import functional as F


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x


net1 = MySequential (nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net2 = nn.Sequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256,10))
X = torch.rand(2, 20)
# Y = X.clone()
# print("x is y",X==Y)
prednet1 = net1(X)
# prednet2 = net2(Y)
# print("pred1:{}\npred2:{}".format(prednet1,prednet2))   # 初始化不一样，所以结果会不一样的

