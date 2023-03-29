import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from network import NeuralNetwork

if __name__ == '__main__':
    model = NeuralNetwork()
    model.load_state_dict(torch.load("demo_model.pth"))

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    batch_size = 64
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model.eval()
    with torch.no_grad():
        pred = model(test_data[0][0])
        print(pred)
        print(pred[0].argmax(0))



