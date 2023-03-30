from torch import nn

num_of_nodes = 128 * 128


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, num_of_nodes),
            nn.ReLU(),
            nn.Linear(num_of_nodes, num_of_nodes),
            nn.ReLU(),
            nn.Linear(num_of_nodes, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


