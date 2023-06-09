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

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

script_type = "trace" # script, trace

if __name__ == '__main__':
    device = torch.device(device_type)
    model = torch.jit.load(f'demo_model.{script_type}.pt')
    model.eval()

    test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_size = 1000 * 2
    hit = 0
    start = time.time()
    for i in range(test_size):
        x, y = test_data[i]
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.int)

        with torch.no_grad():
            pred = model(x)
            pred = pred.cpu()
            if y == pred[0].argmax(0):
                hit += 1

    fin = time.time()
    hit_rate = hit / test_size * 100.0

    print(f"Hit rate {hit_rate:.2f}%")
    elapse = fin - start
    print(f"Inference takes {elapse:.2f} s")

