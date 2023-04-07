import time

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

parallelize = False
device_type = "cuda"     # "cpu", "mps", "cuda"

if device_type == "cuda":
    from network_cuda import NeuralNetwork
else:
    from network import NeuralNetwork


if __name__ == '__main__':
    device = torch.device(device_type)
    model = NeuralNetwork()
    if parallelize:
        model = nn.DataParallel(model)
        # model = DataParallelModel(model)
    model.load_state_dict(torch.load("demo_model.pth", map_location=device))
    if device_type != "cuda":
        model = model.to(device)
    model.eval()

    model = torch.jit.script(model)
    model.save("demo_model.pt")
