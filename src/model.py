import torch.nn as nn
import torch.optim as optim

from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
from ewc import *
from topology import PytorchTopology


class Model:
    """
    Defines the model topology, training and evaluation functions
    """

    def __init__(self, topology: PytorchTopology):
        self.__topology = topology
        self.__loss_fn = nn.BCELoss(reduction='none')
        self.__optimizer = optim.Adam(topology.parameters(), lr=0.0001)

    def get_topology(self):
        return self.__topology

    def get_loss_fn(self):
        return self.__loss_fn

    def train(self, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, epochs: int, batch_size: int, ewc: EWC = None, similarity: float = None):
        loss = None
        with tqdm(total=epochs) as bar:
            for epoch in range(epochs):
                for i in range(0, len(X_train_tensor), batch_size):
                    X_batch = X_train_tensor[i:i + batch_size]
                    y_pred = self.__topology.forward(X_batch)
                    y_batch = y_train_tensor[i:i + batch_size]
                    y_batch = y_batch.reshape(-1, 1)
                    loss = self.__loss_fn(y_pred, y_batch).mean()
                    if ewc is not None:
                        ewc_loss = ewc.penalty(self.__topology)
                        loss = loss + 2000 * ewc_loss * similarity
                    self.__optimizer.zero_grad()
                    loss.backward()
                    self.__optimizer.step()
                bar.set_description("Loss: %f" % loss)
                bar.update()

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor):
        with torch.no_grad():
            y_pred = self.__topology.forward(X_test)
            y_pred = (y_pred > 0.5).float()
            accuracy = torch.Tensor((y_pred.round() == y_test)).float().mean().mul(100)
            f1_metric = BinaryF1Score()
            return accuracy.item(), f1_metric(y_pred, y_test).item() * 100

    def get_loss(self, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> torch.Tensor:
        losses = torch.zeros(y.shape)
        for i in range(0, len(X), batch_size):
            with torch.no_grad():
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.__topology.forward(X_batch)
                loss = self.__loss_fn(y_pred, y_batch)
                losses[i:i + batch_size] = loss
        return losses
