import os
import time
import random

import numpy as np
import matplotlib.pyplot as plt

import tensorrt as trt
from cuda import cudart

from train import get_train_data

onnx_file = "./model.onnx"
trt_file = "./model.plan"

# for FP16 mode
use_fp16_mode = False


def get_engine():
    logger = trt.Logger(trt.Logger.ERROR)
    if os.path.exists(trt_file):
        with open(trt_file, "rb") as f:  # read .plan file if exists
            engine_string = f.read()
        if engine_string is None:
            print("Failed getting serialized engine!")
            return
        print("Succeeded getting serialized engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # set workspace for TensorRT
        if use_fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnx_file):
            print("Failed finding ONNX file!")
            return
        print("Succeeded finding ONNX file!")
        with open(onnx_file, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return
            print("Succeeded parsing .onnx file!")

        input_tensor = network.get_input(0)
        profile.set_shape(input_tensor.name, [1, train_num, 2], [10, train_num, 2], [MAX_BATCH, train_num, 2])
        config.add_optimization_profile(profile)

        engine_string = builder.build_serialized_network(network, config)
        if engine_string is None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trt_file, "wb") as f:
            f.write(engine_string)

    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_string)

    return engine


def inference(data_input, context, buffer_h, buffer_d):
    batch = data_input.shape[0]
    buffer_h[0][:batch, :, :] = np.ascontiguousarray(data_input)
    cudart.cudaMemcpy(buffer_d[0], buffer_h[0].ctypes.data, buffer_h[0].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(buffer_d)  # inference

    cudart.cudaMemcpy(buffer_h[1].ctypes.data, buffer_d[1], buffer_h[1].nbytes,
                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outs = buffer_h[1].reshape((-1, 2))[:batch, :]
    return outs


if __name__ == '__main__':
    train_num = 9
    MAX_BATCH = 50

    # tensorrt 导出及推理
    engine = get_engine()

    n_io = engine.num_bindings
    l_tensor_name = [engine.get_binding_name(i) for i in range(n_io)]
    n_input = np.sum([engine.binding_is_input(i) for i in range(n_io)])

    context = engine.create_execution_context()
    context.set_binding_shape(0, [MAX_BATCH, train_num, 2])  # 永远使用最大 batch size 作为输入
    for i in range(n_io):
        print("[%2d]%s->" % (i, "Input " if i < n_input else "Output"), engine.get_binding_dtype(i),
              engine.get_binding_shape(i), context.get_binding_shape(i), l_tensor_name[i])

    buffer_h = []
    for i in range(n_io):
        buffer_h.append(np.empty(context.get_binding_shape(i), dtype=trt.nptype(engine.get_binding_dtype(i))))
    buffer_d = []
    for i in range(n_io):
        buffer_d.append(cudart.cudaMalloc(buffer_h[i].nbytes)[1])

    # 获取数据
    train_X, train_Y = get_train_data("data.csv", train_num)  # 用于预测的数据
    X_train = train_X.reshape((-1, train_num, 2)).astype(np.float32)
    Y_train = train_Y.reshape((-1, 1, 2)).astype(np.float32)
    print('X_train.shape： ', X_train.shape, 'Y_train.shape：', Y_train.shape)

    for i in range(1, 6):  # 进行 5 次预测
        print("========= infer count: %d ========" % i)
        # 模拟多次进行 batch_size 不确定的推理
        start = time.time()

        rand_batch = random.randint(1, 50)  # 随机一个batch_size
        input_data = X_train[:rand_batch, :, :]  # 截取出部分数据用于推理
        y_pred = inference(input_data, context, buffer_h, buffer_d)
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
        plt.savefig("res_%d.jpg" % i)
        plt.close()

    for b in buffer_d:
        cudart.cudaFree(b)
