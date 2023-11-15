import torch
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from tqdm import tqdm

from ewc import *
from topology import PytorchTopology


class Model:
    """
    Defines the model topology, training and evaluation functions
    """

    def __init__(self, topology: PytorchTopology, loss_fn: _Loss):
        self.__topology = topology
        self.__loss_fn = loss_fn
        self.__optimizer = optim.Adam(topology.parameters(), lr=0.0001)

    def get_topology(self):
        return self.__topology

    def get_loss_fn(self):
        return self.__loss_fn

    def train(self, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, epochs: int, batch_size: int,
              ewc: EWC = None, similarity: float = None):
        self.__topology.train()
        with tqdm(total=epochs) as bar:
            for epoch in range(epochs):
                for i in range(0, len(X_train_tensor), batch_size):
                    X_batch = X_train_tensor[i:i + batch_size]
                    y_pred = self.__topology.forward(X_batch)
                    y_batch = y_train_tensor[i:i + batch_size]
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
        self.__topology.eval()
        with torch.no_grad():
            y_pred = self.__topology.forward(X_test)
            num_classes = y_pred.shape[1]
            y_pred = torch.argmax(y_pred, dim=1)
            acc_metric = MulticlassAccuracy(num_classes=num_classes)
            accuracy = acc_metric(y_pred, y_test).mul(100)
            f1_metric = MulticlassF1Score(num_classes=num_classes, average=None)
            f1_score = f1_metric(y_pred, y_test).mul(100)
            return accuracy, f1_score

    def get_loss(self, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> torch.Tensor:
        self.__topology.eval()
        losses = torch.zeros(size=(y.shape[0], 1))
        for i in range(0, len(X), batch_size):
            with torch.no_grad():
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.__topology.forward(X_batch)
                loss = self.__loss_fn(y_pred, y_batch)
                loss = loss.reshape(-1, 1)
                losses[i:i + batch_size] = loss
        return losses
