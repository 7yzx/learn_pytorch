import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print(net[2].state_dict())
print(net[2].bias)
print(net[2].bias.data)
print(net.state_dict()['2.bias'].data)
print(net[2].weight)
print(*[(name, param.shape) for name, param in net.named_parameters()])


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net


rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)

# 参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)


def init_Xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])

#
