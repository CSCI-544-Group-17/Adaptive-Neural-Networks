import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.topology import ExtensibleColumnProgNN
from src.topology import InitialColumnProgNN
from src.topology import PNNTopology
from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


class PNNModel:
    def __init__(self):
        self.num_classes = 1
        self.topology = [256, 100, 64, 25, 2]
        self.activations = [F.relu, F.relu, F.relu]
        self.lr = 0.001
        self.batch_size = 100
        self.subnetworks: List[PNNTopology] = [InitialColumnProgNN(self.topology, self.activations, self.lr)]

    def should_add(self, index: int) -> bool:
        return index >= self.num_classes

    def add_network(self):
        self.num_classes += 1
        self.subnetworks.append(ExtensibleColumnProgNN(self.topology, self.activations, self.subnetworks, self.lr))

    def get_subnetwork(self, index: int) -> PNNTopology:
        return self.subnetworks[index]

    def get_all_parameters(self):
        # Aggregate parameters from all subnetworks
        all_params = [param for subnetwork in self.subnetworks for param in subnetwork.parameters()]
        return all_params

    def set_mode_train(self):
        for column in self.subnetworks:
            column.train()

    def set_mode_eval(self):
        for column in self.subnetworks:
            column.eval()

    def train(self, subnetwork_index: int, train_dataset: TensorDataset, epochs: int, ewc=None, similarity=None):
        self.set_mode_train()
        column = self.get_subnetwork(subnetwork_index)
        self.__freeze_params(subnetwork_index)
        with tqdm(total=epochs) as bar:
            for epoch in range(epochs):
                for X_batch, y_batch in DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True):
                    y_pred = column(X_batch)
                    loss = column.get_criterion()(y_pred, y_batch)
                    if ewc is not None:
                        ewc_loss = ewc.penalty(column)
                        loss += 2000 * ewc_loss * similarity
                    loss.backward()
                    column.optimizer.step()
                    column.optimizer.zero_grad()
                    bar.set_description("Loss: %f" % loss.item())
                    bar.update()
        self.__unfreeze_params()

    def evaluate(self, X_test: torch.Tensor, y_test: torch.Tensor):
        self.set_mode_eval()
        confidence_scores_class_arr = []
        predictions_class_arr = []
        for i in range(self.num_classes):
            logits, predicted = self.__predict(i, X_test)
            confidence_scores_class_arr.append(logits)
            predictions_class_arr.append(predicted)
        one_hot_predictions = []
        for j in range(len(y_test)):
            tmp = [0] * self.num_classes
            max_confidence = float('-inf')
            max_idx = -1
            for i in range(self.num_classes):
                if confidence_scores_class_arr[i][j] > max_confidence and predictions_class_arr[i][j] == 1:
                    max_confidence = confidence_scores_class_arr[i][j]
                    max_idx = i
            if max_idx != -1:
                tmp[max_idx] = 1
            one_hot_predictions.append(tmp)
        true_class_labels = y_test.tolist()
        predicted_class_labels = [pred.index(1) if 1 in pred else 0 for pred in one_hot_predictions]
        accuracy = accuracy_score(true_class_labels, predicted_class_labels)
        f1 = f1_score(true_class_labels, predicted_class_labels, average='macro')
        return accuracy, f1

    def __freeze_params(self, subnetwork_index: int):
        for i in range(self.num_classes):
            if i == subnetwork_index:
                # Unfreeze the parameters of the modules
                for param in self.get_subnetwork(i).parameters():
                    param.requires_grad = True
            else:
                # Freeze the parameters of the modules
                for param in self.get_subnetwork(i).parameters():
                    param.requires_grad = False

    def __unfreeze_params(self):
        # Unfreeze all parameters
        for i in range(self.num_classes):
            for param in self.subnetworks[i].parameters():
                param.requires_grad = True

    def __predict(self, index: int, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        column = self.subnetworks[index]
        column.eval()
        with torch.no_grad():
            test_outputs = column(embeddings)
            logits, predicted = torch.max(test_outputs, 1)
            return logits, predicted

    # Changed the name of train to train_PNN for it to work with REPEAT
    def train_PNN(self, subnetwork, embeddings, labels):
        for i in range(self.num_classes):
            if i == subnetwork:
                # Unfreeze the parameters of the modules
                for param in self.subnetworks[i].parameters():
                    param.requires_grad = True
            else:
                # Freeze the parameters of the modules
                for param in self.subnetworks[i].parameters():
                    param.requires_grad = False
        # Train the relevant PNN
        train_column(self.subnetworks[subnetwork], embeddings, labels, epochs=50)

        # Unfreeze all parameters
        for i in range(self.num_classes):
            for param in self.subnetworks[i].parameters():
                param.requires_grad = True

    def test_PNN(self, embeddings, labels, i):
        confidence_scores_class_arr = []
        predictions_class_arr = []
        for i in range(self.num_classes):
            confidence_scores, predictions = test_column(self.subnetworks[i], embeddings)
            confidence_scores_class_arr.append(confidence_scores)
            predictions_class_arr.append(predictions)
        one_hot_predictions = []
        for j in range(len(labels)):
            tmp = [0] * self.num_classes
            max_confidence = float('-inf')
            max_idx = -1
            for i in range(self.num_classes):
                if confidence_scores_class_arr[i][j] > max_confidence and predictions_class_arr[i][j] == 1:
                    max_confidence = confidence_scores_class_arr[i][j]
                    max_idx = i
            if max_idx != -1:
                tmp[max_idx] = 1
            one_hot_predictions.append(tmp)

        # Convert one-hot encoded ground truth and predictions to class labels
        true_class_labels = []
        predicted_class_labels = []
        for j in range(len(labels)):
            flag = True
            for i in range(self.num_classes):
                if labels[j][i] == 1:
                    flag = False
                    true_class_labels.append(i + 1)
                    break
            if flag:
                true_class_labels.append(0)

        for j in range(len(one_hot_predictions)):
            flag = True
            for i in range(self.num_classes):
                if one_hot_predictions[j][i] == 1:
                    flag = False
                    predicted_class_labels.append(i + 1)
                    break
            if flag:
                predicted_class_labels.append(0)

        print(true_class_labels)
        print(predicted_class_labels)

        # Compute metrics
        accuracy = accuracy_score(true_class_labels, predicted_class_labels)
        f1 = f1_score(true_class_labels, predicted_class_labels, average='macro')
        precision = precision_score(true_class_labels, predicted_class_labels, average='macro')
        recall = recall_score(true_class_labels, predicted_class_labels, average='macro')

        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')


# Added new function to just return loss for an epoch
def calculate_loss(pnn: PNNModel, subnetwork_index, output, target):
    loss = pnn.subnetworks[subnetwork_index].criterion(output, target)
    return loss


def freeze_params(pnn: PNNModel, subnetwork_index: int):
    for i in range(pnn.num_classes):
        if i == subnetwork_index:
            # Unfreeze the parameters of the modules
            for param in pnn.subnetworks[i].parameters():
                param.requires_grad = True
        else:
            # Freeze the parameters of the modules
            for param in pnn.subnetworks[i].parameters():
                param.requires_grad = False


def forward2(pnn: PNNModel, subnetwork_index: int, X: torch.Tensor, optimizer):
    optimizer.zero_grad()
    return pnn.subnetworks[subnetwork_index](X)


def forward(column, X: torch.Tensor):
    return column(X)


def unfreeze_params(pnn: PNNModel):
    # Unfreeze all parameters
    for i in range(pnn.num_classes):
        for param in pnn.subnetworks[i].parameters():
            param.requires_grad = True


# todo: add training batch size later
def train_column(column, data, target, epochs=50, batch_size=32):
    # Create a dataset and data loader for batching
    dataset = TensorDataset(data, target)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # saved h_values for training lateral connections
    for epoch in range(epochs):
        for batch_data, batch_target in data_loader:
            column.optimizer.zero_grad()

            # forward pass
            output = column(batch_data)

            loss = column.criterion(output, batch_target)
            loss.backward()
            column.optimizer.step()
        if epoch % 10 == 9:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')


def test_column(column, embeddings):
    with torch.no_grad():
        test_outputs = column(embeddings)
        logits, predicted = torch.max(test_outputs, 1)
        return (logits, predicted)

        # # Calculate accuracy
        # accuracy = accuracy_score(labels, predicted)

        # # Calculate F1 score
        # f1 = f1_score(labels, predicted)

        # # Calculate precision
        # precision = precision_score(labels, predicted)

        # # Calculate recall
        # recall = recall_score(labels, predicted)

        # print("Accuracy: {:.2f}".format(accuracy))
        # print("F1 Score: {:.2f}".format(f1))
        # print("Precision: {:.2f}".format(precision))
        # print("Recall: {:.2f}".format(recall))


""" model = PNN()

# train_column(model, embeddings0, labels0, epochs=50)

model.subnetworks

# Task 1
"""
"""
import json
import torch
embeddings = []
labels = []

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/dos_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embeddings'][0])
            labels.append(data['label'])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.train(0, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([data['label']])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.test(embeddings0, labels0, 0)

"""
"""
# Task 2

import json
import torch
embeddings = []
labels = []

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+info_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embeddings'][0])
            labels.append(data['label'])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.add_network()
model.train(1, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([data['label'], 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, data['label']])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

model.test(embeddings0, labels0, 1)

"""

"""

# Task 3


import json
import torch
embeddings = []
labels = []

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/bypass_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embeddings'][0])
            labels.append(data['label'])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.add_network()
model.train(2, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([data['label'], 0, 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, data['label'], 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/bypass_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, 0, data['label']])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

model.test(embeddings0, labels0, 2)

"""
"""# Task 4

import json
import torch
embeddings = []
labels = []

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+priv_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embeddings'][0])
            labels.append(data['label'])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.add_network()
model.train(3, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([data['label'], 0, 0, 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, data['label'], 0, 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/bypass_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, 0, data['label'], 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+priv_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, 0, 0, data['label']])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

model.test(embeddings0, labels0, 3)

"""  # Task 5

"""

import json
import torch
embeddings = []
labels = []

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/other_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embeddings'][0])
            labels.append(data['label'])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.add_network()
model.train(4, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([data['label'], 0, 0, 0, 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, data['label'], 0, 0, 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/bypass_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, 0, data['label'], 0, 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+priv_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, 0, 0, data['label'], 0])

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/other_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        labels.append([0, 0, 0, 0, data['label']])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

model.test(embeddings0, labels0, 4) 
"""
