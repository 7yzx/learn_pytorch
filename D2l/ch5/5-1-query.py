import torch
from torch import nn
from torch.nn import functional as F

# Q1
class MySequential_list(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = []
        for idx, module in enumerate(args):
            self.modules_list.append(module)

    def forward(self, x):
        for block in self.modules_list:
            x = block(x)
        return x


net_Q1 = MySequential_list([nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10)])
X = torch.rand(2, 20)
prednet_test = net_Q1(X)  # TypeError: 'list' object is not callable
print(prednet_test)
print(net_Q1)


# Q2
class ParallelBlock(nn.Module):
    def __init__(self, net1, net2):
        super(ParallelBlock, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        output1 = self.net1(x)
        output2 = self.net2(x)
        return torch.cat((output1, output2), dim=1)

# 创建两个子网络
net1 = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

net2 = nn.Sequential(
    nn.Linear(20, 128),
    nn.ReLU(),
    nn.Linear(128, 5)
)

# 创建平行块
parallel_block = ParallelBlock(net1, net2)

# 测试前向传播
x = torch.randn(2, 20)  # 创建输入张量
output = parallel_block(x)  # 前向传播
print(output)  # 输出形状为 (2, 15)，因为 net1 输出 10 维，net2 输出 5 维，串联后为 15 维
