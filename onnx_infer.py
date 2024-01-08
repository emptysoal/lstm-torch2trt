import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import onnxruntime

from train import Net
from train import get_train_data

pth_file = "./model.pth"
onnx_file = "./model.onnx"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def pth2onnx():
    """
        PyTorch模型转onnx模型
    """
    model = Net()
    model.load_state_dict(torch.load(pth_file, map_location="cpu"))
    model.to(device)

    torch.onnx.export(
        model,
        torch.randn((1, train_num, 2), device=device),
        onnx_file,
        input_names=["data"],
        output_names=["output"],
        do_constant_folding=True,
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=12,
        dynamic_axes={"data": {0: "nBatchSize"}, "output": {0: "nBatchSize"}}
    )
    print("Succeeded converting model into ONNX!")


def inference(data_input, onnx_session):
    inputs = {onnx_session.get_inputs()[0].name: data_input}
    outs = onnx_session.run(None, inputs)[0]
    outs = outs.reshape((-1, 2))
    return outs


if __name__ == '__main__':
    train_num = 9
    # 获取数据
    train_X, train_Y = get_train_data("data.csv", train_num)  # 用于预测的数据

    X_train = train_X.reshape((-1, train_num, 2)).astype(np.float32)
    Y_train = train_Y.reshape((-1, 1, 2)).astype(np.float32)
    print('X_train.shape： ', X_train.shape, 'Y_train.shape：', Y_train.shape)

    # onnx 模型导出及加载
    if not os.path.exists(onnx_file):  # 不存在onnx模型就使用pth模型导出
        pth2onnx()

    # onnx load
    session = onnxruntime.InferenceSession(
        onnx_file,
        providers=[
            # 'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            # 'CPUExecutionProvider'
        ]
    )

    # onnx模型推理
    start = time.time()

    y_pred = inference(X_train, session)
    print("y_pred.shape:", y_pred.shape)

    y_true = Y_train.reshape((-1, 2))
    print("y_true.shape:", y_true.shape)

    end = time.time()
    print("Inference cost is: %.4f s" % (end - start))

    # 绘制结果
    plt.scatter(y_true[:, 0], y_true[:, 1], c='b', marker='o', label='y_true')
    plt.scatter(y_pred[:, 0], y_pred[:, 1], c='r', marker='o', label='y_pred')
    plt.legend(loc='upper left')
    plt.grid()
    # plt.show()
    plt.savefig("res.jpg")
