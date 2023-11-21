import torch
from torch import nn as nn
from torch.nn.modules.loss import _Loss


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


class PNNTopology(PytorchTopology):
    def __init__(self, name):
        super().__init__(name)

    def get_criterion(self) -> _Loss:
        raise NotImplementedError("Please implement the 'get_criterion' function")


class InitialColumnProgNN(PNNTopology):
    def __init__(self, topology, activations, lr):
        super().__init__("Initial Column PNN")
        self.layers = nn.ModuleList()
        for i in range(len(topology) - 1):
            self.layers.append(nn.Linear(topology[i], topology[i + 1]))
        self.activations = activations
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.activations):
                x = self.activations[i](x)
        return x

    def get_criterion(self) -> _Loss:
        return self.criterion


class ExtensibleColumnProgNN(PNNTopology):
    def __init__(self, topology, activations, prev_columns, lr):
        super().__init__("Extensible PNN")
        self.layers = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        for i in range(len(topology) - 1):
            self.layers.append(nn.Linear(topology[i], topology[i + 1]))
            if i > 0:
                lateral = [nn.Linear(prev_column.layers[i - 1].out_features, topology[i + 1], bias=False) for
                           prev_column in prev_columns]
                self.lateral_connections.append(nn.ModuleList(lateral))
        self.activations = activations
        self.prev_columns = prev_columns
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x: torch.Tensor):
        prev_hs = [[[] for j in range(len(prev_col.layers))] for prev_col in self.prev_columns]
        for j in range(len(self.prev_columns)):
            x_copy = x.clone()
            for i, col_layer in enumerate(self.prev_columns[j].layers):
                x_copy = col_layer(x_copy)
                if i != len(self.prev_columns[j].activations):
                    x_copy = self.prev_columns[j].activations[i](x_copy)
                prev_hs[j][i] = x_copy.clone()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i > 0:
                for j, lateral in enumerate(self.lateral_connections[i - 1]):
                    x += lateral(prev_hs[j][i - 1])
            if i != len(self.activations):
                x = self.activations[i](x)
        return x

    def get_criterion(self) -> _Loss:
        return self.criterion


class ClassifierProgNN(PNNTopology):
    def __init__(self, input_size: int, num_classes: int, lr: float):
        super().__init__("Final Classifier Layer")
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.relu(x)
        return self.classifier(x)

    def get_criterion(self) -> _Loss:
        return self.criterion



