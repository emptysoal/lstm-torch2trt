import time

import matplotlib.pyplot as plt
import torch

from train import Net
from train import get_train_data

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_num = 9

# 获取数据
train_X, train_Y = get_train_data("./data.csv", train_num)  # 用于预测的数据

X_train = torch.tensor(train_X.reshape((-1, train_num, 2)), dtype=torch.float)
Y_train = torch.tensor(train_Y.reshape((-1, 1, 2)), dtype=torch.float)
print('X_train.shape： ', X_train.shape, 'Y_train.shape：', Y_train.shape)

# 加载模型
model = Net()
model.load_state_dict(torch.load('model.pth', map_location="cpu"))
model.to(device)

# 开始推理
start = time.time()

y_true = Y_train.squeeze().cpu().numpy()
print("y_true.shape:", y_true.shape)
y_pred = model(X_train.to(device)).detach().squeeze().cpu().numpy()
print("y_pred.shape:", y_pred.shape)

end = time.time()
print("Inference cost is: %.4f s" % (end - start))

# 绘制结果
plt.scatter(y_true[:, 0], y_true[:, 1], c='b', marker='o', label='y_true')
plt.scatter(y_pred[:, 0], y_pred[:, 1], c='r', marker='o', label='y_pred')
plt.legend(loc='upper left')
plt.grid()
# plt.show()
plt.savefig("res.jpg")
