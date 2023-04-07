from torch import nn

num_of_nodes = 128 * 128


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten().to("cuda:0")
        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28, num_of_nodes),
            nn.ReLU(),
            nn.Linear(num_of_nodes, num_of_nodes),
            nn.ReLU()
        ).to("cuda:1")
        self.layer2 = nn.Sequential(
            nn.Linear(num_of_nodes, num_of_nodes),
            nn.ReLU()
        ).to("cuda:2")
        self.layer_out = nn.Linear(
            num_of_nodes, 10
        ).to("cuda:0")

    def forward(self, x):
        out = self.flatten(x).to("cuda:1")
        out = self.layer1(out).to("cuda:2")
        out = self.layer2(out).to("cuda:0")
        logits = self.layer_out(out)
        return logits


