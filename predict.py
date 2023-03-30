import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor

from network import NeuralNetwork
from parallel import DataParallelModel, DataParallelCriterion

parallelize = True

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

if __name__ == '__main__':
    model = NeuralNetwork()
    if parallelize:
        model = nn.DataParallel(model)
        # model = DataParallelModel(model)
    model.load_state_dict(torch.load("demo_model.pth"))

    test_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    model.eval()
    i = 7
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        pred = model(x)
        print(pred)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')



