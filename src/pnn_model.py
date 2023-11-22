from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from topology import InitialColumnProgNN, ExtensibleColumnProgNN, ClassifierProgNN
from topology import PNNTopology


class PNNModel:
    def __init__(self):
        self.num_classes = 1
        self.topology = [256, 100, 64, 25, 2]
        self.activations = [F.relu, F.relu, F.relu]
        self.lr = 0.001
        self.subnetworks: List[PNNTopology] = [InitialColumnProgNN(self.topology, self.activations, self.lr)]
        self.classifier = ClassifierProgNN(self.topology[-1] * self.num_classes, self.num_classes)

    def should_update(self, index: int) -> bool:
        return index >= self.num_classes

    def update_network(self):
        self.num_classes += 1
        self.subnetworks.append(ExtensibleColumnProgNN(self.topology, self.activations, self.subnetworks, self.lr))
        self.classifier.add_neuron()

    def get_subnetwork(self, index: int) -> PNNTopology:
        return self.subnetworks[index]

    def train_subnetwork(self, subnetwork_index: int, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int, batch_size: int, ewc=None, similarity=None):
        self.set_mode_train()
        column = self.get_subnetwork(subnetwork_index)
        self.__freeze_params(subnetwork_index)
        train_dataset = TensorDataset(X_train, y_train)
        with tqdm(total=epochs) as bar:
            for epoch in range(epochs):
                for X_batch, y_batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
                    y_pred = column(X_batch)
                    loss = column.get_criterion()(y_pred, y_batch).mean()
                    if ewc is not None:
                        ewc_loss = ewc.penalty(column)
                        loss += 2000 * ewc_loss * similarity
                    loss.backward()
                    column.optimizer.step()
                    column.optimizer.zero_grad()
                bar.set_description("Loss: %f" % loss.item())
                bar.update()
        self.__unfreeze_params()

    def train_classifier(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int, batch_size: int):
        self.__freeze_params()
        train_dataset = TensorDataset(X_train, y_train)
        with tqdm(total=epochs) as bar:
            for epochs in range(epochs):
                for X_batch, y_batch in DataLoader(train_dataset, batch_size=batch_size, shuffle=True):
                    y_preds = self.forward(X_batch)
                    loss = self.classifier.get_criterion()(y_preds, y_batch).mean()
                    loss.backward()
                    self.classifier.optimizer.step()
                    self.classifier.zero_grad()
                bar.set_description("Loss: %f" % loss.item())
                bar.update()
        self.__unfreeze_params()

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor):
        self.set_mode_eval()
        with torch.no_grad():
            y_pred = self.forward(X_test)
            _, predicted = torch.max(y_pred, 1)
            accuracy = accuracy_score(y_test, predicted)
            f1 = f1_score(y_test, predicted, average=None) * 100
            return accuracy, np.mean(f1), f1.tolist()

    def forward(self, x) -> torch.Tensor:
        subnetwork_outputs = [subnetwork(x) for subnetwork in self.subnetworks]
        # Concatenate the outputs from all subnetworks
        combined_output = torch.cat(subnetwork_outputs, dim=1)

        # Pass the combined output through the output layer
        final_output = self.classifier.forward(combined_output)
        return final_output

    def set_mode_train(self):
        for column in self.subnetworks:
            column.train()
        self.classifier.train()

    def set_mode_eval(self):
        for column in self.subnetworks:
            column.eval()
        self.classifier.eval()

    def get_loss(self, subnetwork_index: int, X: torch.Tensor, y: torch.Tensor):
        column = self.subnetworks[subnetwork_index]
        self.set_mode_eval()
        with torch.no_grad():
            y_pred = column(X)
            loss = column.get_criterion()(y_pred, y)
            return loss

    def __freeze_params(self, ignore_subnetwork: int = -1):
        for i in range(self.num_classes):
            if i == ignore_subnetwork:
                for param in self.get_subnetwork(i).parameters():
                    param.requires_grad = True
            else:
                for param in self.get_subnetwork(i).parameters():
                    param.requires_grad = False

    def __unfreeze_params(self):
        for i in range(self.num_classes):
            for param in self.subnetworks[i].parameters():
                param.requires_grad = True
