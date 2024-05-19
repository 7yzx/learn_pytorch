import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_data, test_data = d2l.load_data_fashion_mnist(batch_size)

num_input = 784
num_output = 10

W = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
b = torch.zeros(num_output,requires_grad=True)


def softmax(x):
    x = torch.exp(x)
    p = x.sum(1,keepdim=True)
    return x/p

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
