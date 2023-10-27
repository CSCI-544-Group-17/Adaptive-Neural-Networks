import torch
import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import BinaryF1Score

class PytorchTopology(nn.Module):
    """
    Base class implementing Pytorch's nn.Module.
    Defines the structure(topology) of a neural network and the forward pass
    """

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Please implement the forward function")


class FNNTopology(PytorchTopology):
    """
    Defines the topology of a feedforward multi layer perceptron network
    1. 2 layers with 50 and 5 nodes each
    2. The input size can be provided, depending on the size of the word vectors
    """

    def __init__(self, name: str, input_size: int):
        super().__init__(name)
        self.__linear_0 = nn.Linear(input_size, 256)
        self.__relu_0 = nn.ReLU()
        self.__linear_1 = nn.Linear(256, 128)
        self.__relu_1 = nn.ReLU()
        self.__linear_2 = nn.Linear(128, 64)
        self.__relu_2 = nn.ReLU()
        self.__linear_3 = nn.Linear(64, 1)
        self.__sigmoid_out = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__relu_0(self.__linear_0(x))
        x = self.__relu_1(self.__linear_1(x))
        x = self.__relu_2(self.__linear_2(x))
        x = self.__sigmoid_out(self.__linear_3(x))
        return x


class Model:
    """
    Defines the model topology, training and evaluation functions
    """

    def __init__(self, topology: PytorchTopology, epochs: int, batch_size: int):
        self.__topology = topology
        self.__loss_fn = nn.BCELoss()
        self.__optimizer = optim.Adam(topology.parameters(), lr=0.0001)
        self.__epochs = epochs
        self.__batch_size = batch_size

    def train(self, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor):
        loss = None
        for epoch in range(self.__epochs):
            for i in range(0, len(X_train_tensor), self.__batch_size):
                X_batch = X_train_tensor[i:i + self.__batch_size]
                y_pred = self.__topology.forward(X_batch)
                y_batch = y_train_tensor[i:i + self.__batch_size]
                y_batch = y_batch.reshape(-1, 1)
                loss = self.__loss_fn(y_pred, y_batch)
                self.__optimizer.zero_grad()
                loss.backward()
                self.__optimizer.step()
            print("Finished epoch %d, latest loss %f" % (epoch, loss))

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor):
        with torch.no_grad():
            y_pred = self.__topology.forward(X_test)
            y_pred = (y_pred > 0.5).float()
            accuracy = torch.Tensor((y_pred.round() == y_test)).float().mean().mul(100)
            f1_metric = BinaryF1Score()
            return accuracy, f1_metric(y_pred, y_test)