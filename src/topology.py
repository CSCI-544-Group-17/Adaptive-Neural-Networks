import torch
from torch import nn as nn
from pnn import PNN, InitialColumnProgNN, ExtensibleColumnProgNN, train_column, test_column

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


class BinaryFNNTopology(PytorchTopology):
    def __init__(self, name: str, input_size: int):
        super().__init__(name)
        self.__linear_0 = nn.Linear(input_size, 512)
        self.__relu_0 = nn.ReLU()
        self.__linear_1 = nn.Linear(512, 256)
        self.__relu_1 = nn.ReLU()
        self.__linear_2 = nn.Linear(256, 128)
        self.__relu_2 = nn.ReLU()
        self.__linear_3 = nn.Linear(128, 64)
        self.__relu_3 = nn.ReLU()
        self.__linear_4 = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__relu_0(self.__linear_0(x))
        x = self.__relu_1(self.__linear_1(x))
        x = self.__relu_2(self.__linear_2(x))
        x = self.__relu_3(self.__linear_3(x))
        x = self.__linear_4(x)
        return x


class MulticlassFNNTopology(PytorchTopology):
    def __init__(self, name: str, input_size: int, num_classes: int):
        super().__init__(name)
        self.__linear_0 = nn.Linear(input_size, 1024)
        self.__relu_0 = nn.ReLU()
        self.__linear_1 = nn.Linear(1024, 512)
        self.__relu_1 = nn.ReLU()
        self.__linear_2 = nn.Linear(512, 256)
        self.__relu_2 = nn.ReLU()
        self.__linear_3 = nn.Linear(256, 64)
        self.__relu_3 = nn.ReLU()
        self.__linear_4 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__relu_0(self.__linear_0(x))
        x = self.__relu_1(self.__linear_1(x))
        x = self.__relu_2(self.__linear_2(x))
        x = self.__relu_3(self.__linear_3(x))
        x = self.__linear_4(x)
        return x


class MulticlassFNNTopologySmall(PytorchTopology):
    def __init__(self, name: str, input_size: int, num_classes: int):
        super().__init__(name)
        self.__linear_0 = nn.Linear(input_size, 128)
        self.__relu_0 = nn.ReLU()
        self.__linear_1 = nn.Linear(128, 64)
        self.__relu_1 = nn.ReLU()
        self.__linear_2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__relu_0(self.__linear_0(x))
        x = self.__relu_1(self.__linear_1(x))
        x = self.__linear_2(x)
        return x

'''PNN'''

class PNNTopology(PytorchTopology):
    def __init__(self, name: str):
        super().__init__(name)
        # Initialize the PNN model
        self.model = PNN()  # Assuming PNN() initializes with fixed input_size and num_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through your PNN
        return self.model(x)
    
    def get_all_parameters(self):
        # Delegate to the PNN model's method
        return self.model.get_all_parameters()
