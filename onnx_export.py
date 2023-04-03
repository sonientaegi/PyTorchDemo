import time

import onnx
import onnxruntime
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from network import NeuralNetwork

parallelize = True
device_type = "mps"     # "cpu", "mps", "cuda"

if __name__ == '__main__':
    device = torch.device(device_type)
    model = NeuralNetwork()
    if parallelize:
        model = nn.DataParallel(model)
        # model = DataParallelModel(model)
    model.load_state_dict(torch.load("demo_model.pth", map_location=device))
    model.to(device)
    model.eval()

    test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    x = test_data[0][0]
    x = x.to(device)
    y = model(x)

    torch.onnx.export(model.module, x, "demo_model.onnx",
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])

    model = onnx.load("demo_model.onnx")
    onnx.checker.check_model(model)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_session = onnxruntime.InferenceSession("demo_model.onnx", providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(y)
    print(ort_outs)
