import torch
from d2l import torch as d2l
import matplotlib
import pandas as pd
import numpy as np
from torch import nn


def log_rmse(net, loss, features, labels):
    clipped_preds = torch.clamp(net(features),1,float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    loss = nn.MSELoss()
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, loss, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, loss, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k ,i ,X, y):
    assert k>1
    fold_size = X.shape[0]//k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  # silce做一个切片操作
        X_part, y_part = X[idx, :], y[idx]   # 分配第j折的数据和标签
        if j == i:
            X_valid, y_valid = X_part, y_part  # 选验证集的
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
        d2l.plt.show()
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.plot(np.arange(1,num_epochs+1),[train_ls],xlabel='epoch',
             ylabel='log rmse')
    print("训练 log rmse: {}".format(float(train_ls[-1])))
    preds = net(test_features).detach().cpu().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    submission.to_csv("submission.csv",index=False)

if "__main__" == __name__:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = pd.read_csv("../data/house-prices/train.csv")
    test_data = pd.read_csv("../data/house-prices/test.csv")

    print(train_data.shape)
    print(test_data.shape)

    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
    #--------数据预处理------------
    # 若无法获得测试数据，则可根据训练数据计算均值和标准差
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index  # 筛选出不是 'object' 类型的列（即数值列）
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))   # 这段代码对筛选出的数值特征进行标准化处理。标准化的目的是将每个特征的均值调整为0，标准差调整为1。
    # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0

    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    # 通过对数据框 all_features 使用 pd.get_dummies 并设置 dummy_na=True，
    # 将分类特征转换为二进制特征，同时为缺失值创建指示符特征。之后，通过 all_features.shape 查看数据框的形状，以了解数据框的大小和特征数量。
    all_features = pd.get_dummies(all_features,dummy_na=True)
    print(all_features.shape)
    #--------------------------------
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32, device=device)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32, device=device)
    train_labels = torch.tensor(
        train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32, device=device)

    in_features = train_features.shape[1]

    def get_net():
        net = nn.Sequential(nn.Linear(in_features, 1)).to(device)
        return net

    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    # train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
    #                           weight_decay, batch_size)
    # print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
    #       f'平均验证log rmse: {float(valid_l):f}')

    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)