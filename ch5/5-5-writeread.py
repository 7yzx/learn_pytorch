import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(5, 10)
        self.output = nn.Linear(10, 1)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


# x = torch.arange(4)
# torch.save(x, 'x-file')
# print("write to x-file")
#
# x2 = torch.load('x-file')
# print(x2)


net = MLP()
X = torch.randn(size=(2, 5))
Y = net(X)

torch.save(net,"mlp_net")                 # 模型结构＋参数
torch.save(net.state_dict(),"mlp_params") # 只有参数，推荐

model1 = torch.load("mlp_net")
print(model1)

model2 = torch.load("mlp_params")
print(model2)

#加载
clone_mlp = MLP()
clone_mlp.load_state_dict(torch.load('mlp_params'))
print(clone_mlp)

