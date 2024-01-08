# -*- coding: utf-8 -*-

"""
    训练数据：位于某一曲线上的 59 个二维坐标值
    模型功能：根据连续的 9 个坐标值，预测下一个坐标
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def create_dataset(data, n_predictions):
    train_X, train_Y = [], []
    for i in range(data.shape[0] - n_predictions):
        a = data[i:i + n_predictions, :]
        train_X.append(a)
        b = data[i + n_predictions, :]
        train_Y.append(b)
    train_X = np.array(train_X, dtype='float64')
    train_Y = np.array(train_Y, dtype='float64')

    return train_X, train_Y


def normalize_multi(data):
    x_min, x_max = 0, 100
    y_min, y_max = 0, 10
    normalized_data = np.zeros(data.shape, dtype=np.float32)
    normalized_data[:, 0] = (data[:, 0] - x_min) / (x_max - x_min)
    normalized_data[:, 1] = (data[:, 1] - y_min) / (y_max - y_min)
    return normalized_data


def get_train_data(raw_data_path, train_num=9):
    # 读入时间序列的文件数据
    data = pd.read_csv(raw_data_path).values
    print('data shape: ', data.shape)
    print("样本数：{0}，维度：{1}".format(data.shape[0], data.shape[1]))

    # 归一化
    # data, normalize = normalize_multi(data, set_range)
    data = normalize_multi(data)
    # 生成训练数据
    train_X, train_Y = create_dataset(data, train_num)

    return train_X, train_Y


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=6, num_layers=3, batch_first=True,
                            bidirectional=True)  # 特征是x,y两个坐标，此处修改为2
        self.fc = nn.Linear(in_features=12, out_features=2)  # 预测结果是x,y两个坐标，此处修改为2

    def forward(self, x):
        # x is input, size (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        # x is output, size (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]
        x = self.fc(x)
        x = x.view(-1, 1, 2)  # 此处修改为2
        return x


def train_step(model, features, labels):
    features, labels = features.to(device), labels.to(device)
    predictions = model(features)
    loss = loss_function(predictions, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def train_model(model, epochs):
    for epoch in range(1, epochs + 1):
        list_loss = []
        for features, labels in dl_train:
            lossi = train_step(model, features, labels)
            list_loss.append(lossi)
        loss = np.mean(list_loss)
        if epoch % 10 == 0:
            print('epoch={} | loss={} '.format(epoch, loss))
    print("finish training")
    save_path = './model.pth'
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_num = 9

    model = Net().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 数据说明：根据 9 个连续的坐标，预测下一个坐标，所以 input shape 为 （9, 2）
    train_X, train_Y = get_train_data("data.csv", train_num)  # train_X: shape(50, 9, 2), train_Y: shape(50, 2)

    # 创建用于训练的 dataset 和 dataloader
    X_train = torch.tensor(train_X.reshape((-1, train_num, 2)), dtype=torch.float)
    Y_train = torch.tensor(train_Y.reshape((-1, 1, 2)), dtype=torch.float)
    print('X_train.shape: ', X_train.shape, 'Y_train.shape: ', Y_train.shape)

    batch_size = 10
    ds_train = TensorDataset(X_train, Y_train)
    dl_train = DataLoader(ds_train, batch_size=batch_size, num_workers=0)

    # # 查看第一个batch
    # x, y = next(iter(dl_train))
    # print(x.shape)
    # print(y.shape)
    # 下面三行的作用：开启训练前，调通一个前向传播
    # features, labels = next(iter(dl_train))
    # loss = train_step(model, features, labels)
    # print(loss)

    # 开启训练
    train_model(model, 400)

    # 预测验证预览
    model = Net()
    model.load_state_dict(torch.load('model.pth', map_location="cpu"))
    model.to(device)
    y_true = Y_train.squeeze().cpu().numpy()
    print("y_true.shape:", y_true.shape)
    y_pred = model(X_train.to(device)).detach().squeeze().cpu().numpy()
    print("y_pred.shape:", y_pred.shape)

    # 画样本数据库
    plt.scatter(y_true[:, 0], y_true[:, 1], c='b', marker='o', label='y_true')
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='r', marker='o', label='y_pred')
    plt.legend(loc='upper left')
    plt.grid()
    # plt.show()
    plt.savefig("res.jpg")
