import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

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

if __name__ == '__main__':
    model = NeuralNetwork()
    model.load_state_dict(torch.load("demo_model.pth"))

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
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



