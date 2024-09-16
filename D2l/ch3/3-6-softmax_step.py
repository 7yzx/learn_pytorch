# import torch
# from IPython import display
# from d2l import torch as d2l
#
# batch_size = 256
# train_data, test_data = d2l.load_data_fashion_mnist(batch_size)
#
# num_input = 784
# num_output = 10
#
# W = torch.normal(0,0.01,size=(num_input,num_output),requires_grad=True)
# b = torch.zeros(num_output,requires_grad=True)
#
# def softmax(X):
#     x_exp = torch.exp(X)
#     p = x_exp.sum(1,keepdim=True)
#     return x_exp/p
#
# def net(X):
#     return softmax(torch.matmul(X.reshape(-1,W.shape[0]),W)+b)
#
# def cross_loss(y_hat,y):
#     return - torch.log(y_hat[range(len(y_hat)),y])
#
# def accuracy(y_hat, y ):
#     if len(y_hat.shape) >1 and y_hat.shape[1]>1:
#         y_hat = y_hat.argmax(axis=1)
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum())
#
#
# if __name__ == "__main__":
#
#
