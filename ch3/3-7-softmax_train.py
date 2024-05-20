import torch
from torch import nn
from d2l import torch as d2l
import time
import threading
from IPython import display
"""
和原来代码逻辑相同，添加了pytharm可以运行的Animator，还有支持GPU运算。
"""

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        if 'display' in globals():  # 非Notebook环境不调用display()
            display.display(self.fig)
            display.clear_output(wait=True)


def train_epoch_ch3(net, train_iter, loss, updater, device):
    """The training loop defined in Chapter 3."""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = d2l.Accumulator(3)
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum())*100, d2l.accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_accuracy(net, data_iter, device):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, loss, num_epochs, updater, device):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    def train_thread(device):
        for epoch in range(num_epochs):
            print(f"in the epoch {epoch}")
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater, device)
            test_acc = evaluate_accuracy(net, test_iter, device)
            animator.add(epoch + 1, train_metrics + (test_acc,))
            print(f'epoch {epoch}: train loss {train_metrics[0]*100:.2f}, train accuracy {train_metrics[1]:.2f}, test accuracy {test_acc:.2f}')
            d2l.plt.draw()
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

    th = threading.Thread(target=train_thread, args=(device,), name='training')
    th.start()
    d2l.plt.show(block=True)
    th.join()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    print("loading data ")
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    print("loaded ")
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    net.to(device)  # 将模型转移到设备
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    epochs = 10
    train(net, train_iter, test_iter, loss, epochs, trainer, device)
