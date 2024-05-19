import matplotlib.pyplot
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt
d2l.use_svg_display()

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=d2l.get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=d2l.get_dataloader_workers()))

timer = d2l.Timer()
resize = False
trans = [transforms.ToTensor()]
if resize:
    trans.insert(0, transforms.Resize(resize))
trans = transforms.Compose(trans)

train_data,test_data = load_data_fashion_mnist(16,64)
for i,sample in enumerate(train_data):
    img = sample[0]
    label = sample[1]

    d2l.show_images(img.reshape(16,64,64),2,8,titles=d2l.get_fashion_mnist_labels(label))
    if i==0:
        plt.show()
    print(f"{timer.stop():.2f} sec")