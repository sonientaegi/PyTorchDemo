import time

import onnx
import onnxruntime
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from network import NeuralNetwork

parallelize = True

if __name__ == '__main__':
    model = onnx.load("demo_model.onnx")
    onnx.checker.check_model(model)

    ort_session = onnxruntime.InferenceSession("demo_model.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    name = ort_session.get_inputs()[0].name

    test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    test_size = 1000 * 2
    hit = 0
    start = time.time()
    for i in range(test_size):
        x, y = test_data[i]
        ort_inputs = {name: to_numpy(x)}
        ort_outs = ort_session.run(None, ort_inputs)

        if y == ort_outs[0][0].argmax(0):
            hit += 1

    fin = time.time()
    hit_rate = hit / test_size * 100.0

    print(f"Hit rate {hit_rate:.2f}%")
    elapse = fin - start
    print(f"Inference takes {elapse:.2f} s")
