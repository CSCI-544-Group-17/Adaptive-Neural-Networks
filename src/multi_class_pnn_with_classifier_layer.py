# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

class InitialColumnProgNN(nn.Module):
    def __init__(self, topology, activations, lr):
        super(InitialColumnProgNN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(topology) - 1):
            self.layers.append(nn.Linear(topology[i], topology[i+1]))
        self.activations = activations

        # optimizer and criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # removinf softmax from last layer because of crossentropyloss function - vishal
            if i != len(self.activations):
              x = self.activations[i](x)
        return x

class ExtensibleColumnProgNN(nn.Module):
    def __init__(self, topology, activations, prev_columns, lr):
        super(ExtensibleColumnProgNN, self).__init__()
        self.layers = nn.ModuleList()
        self.lateral_connections = nn.ModuleList()
        for i in range(len(topology) - 1):
            self.layers.append(nn.Linear(topology[i], topology[i+1]))
            if i > 0:
                lateral = [nn.Linear(prev_column.layers[i-1].out_features, topology[i+1], bias=False) for prev_column in prev_columns]
                self.lateral_connections.append(nn.ModuleList(lateral))
        self.activations = activations
        self.prev_columns = prev_columns

        # optimizer and criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        prev_hs = [[[] for j in range(len(prev_col.layers))] for prev_col in self.prev_columns]

        for j in range(len(self.prev_columns)):
            x_copy = x.clone()
            for i, col_layer in enumerate(self.prev_columns[j].layers):
                x_copy = col_layer(x_copy)
                # removinf softmax from last layer because of crossentropyloss function - vishal
                if i != len(self.prev_columns[j].activations):
                  x_copy = self.prev_columns[j].activations[i](x_copy)
                prev_hs[j][i] = x_copy.clone()


        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i > 0:
                for j, lateral in enumerate(self.lateral_connections[i-1]):
                    x += lateral(prev_hs[j][i - 1])
            # removinf softmax from last layer because of crossentropyloss function - vishal
            if i != len(self.activations):
              x = self.activations[i](x)
        return x

#todo: add training batch size later
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
        if epoch%10 == 9:
          print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

def test_column(column, embeddings, labels):
    column.eval()  # Set the model to evaluation mode todo: do we need to unset eval mode after testing??
    with torch.no_grad():
        test_outputs = column(embeddings)
        logits, predicted = torch.max(test_outputs, 1)
        return (logits, predicted)

def train_output_layer(model, data, target, epochs=50, batch_size=32, lr=0.01):
    # Freeze all parameters
    for subnetwork in model.subnetworks:
        for param in subnetwork.parameters():
            param.requires_grad = False

    # Create a dataset and data loader for batching
    dataset = TensorDataset(data, target)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Make sure the optimizer is now set up for the output layer only
    optimizer = torch.optim.Adam(model.output_layer.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    # saved h_values for training lateral connections
    for epoch in range(epochs):
        for batch_data, batch_target in data_loader:
            optimizer.zero_grad()

            # forward pass
            output = model(batch_data)

            loss = criterion(output, batch_target)
            loss.backward()
            optimizer.step()
        if epoch%10 == 9:
          print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    # Unfreeze all parameters
    for subnetwork in model.subnetworks:
        for param in subnetwork.parameters():
            param.requires_grad = True

def test_output_layer(model, embeddings, labels):
    # model.eval()  # Set the model to evaluation mode todo: do we need to unset eval mode after testing??
    print('Yass')
    with torch.no_grad():
        test_outputs = model(embeddings)
        _, predicted = torch.max(test_outputs, 1)

        # Calculate accuracy
        accuracy = accuracy_score(labels, predicted)

        # Calculate F1 score
        f1 = f1_score(labels, predicted, average='macro')

        # Calculate precision
        precision = precision_score(labels, predicted, average='macro')

        # Calculate recall
        recall = recall_score(labels, predicted, average='macro')

        print("Accuracy: {:.2f}".format(accuracy))
        print("F1 Score: {:.2f}".format(f1))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))

class PNN(nn.Module):
    def __init__(self):
        super(PNN, self).__init__()

        self.num_classes = 1

        self.topology = [256, 100, 64, 25, 2]
        self.activations = [F.relu, F.relu, F.relu]

        self.lr = 0.01

        # Instantiate the first module
        self.subnetworks = [InitialColumnProgNN(self.topology, self.activations, 0.001)]

        # Initialize the output layer with an arbitrary number of output features (e.g., number of classes)
        self.output_layer = nn.Linear(self.topology[-1] * self.num_classes, self.num_classes)

    def add_network(self):
        self.num_classes += 1
        self.subnetworks.append(ExtensibleColumnProgNN(self.topology, self.activations, self.subnetworks, self.lr))

        print(self.num_classes)

        # Adjust the output layer to match the new number of total output features
        self.output_layer = nn.Linear(self.topology[-1] * self.num_classes, self.num_classes)

    def forward(self, x):
        subnetwork_outputs = [subnetwork(x) for subnetwork in self.subnetworks]
        # Concatenate the outputs from all subnetworks
        combined_output = torch.cat(subnetwork_outputs, dim=1)

        # Pass the combined output through the output layer
        final_output = self.output_layer(combined_output)
        return final_output

    def train(self, subnetwork, embeddings, labels, epochs=50):
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
        train_column(self.subnetworks[subnetwork], embeddings, labels, epochs=epochs)

        #Unfreeze all parameters
        for i in range(self.num_classes):
            for param in self.subnetworks[i].parameters():
                    param.requires_grad = True

    def test(self, embeddings, labels, i):
        confidence_scores_class_arr = []
        predictions_class_arr = []
        for i in range(self.num_classes):
            # print('\nResults for class', i, '-')
            confidence_scores, predictions = test_column(self.subnetworks[i], embeddings, labels[i])
            confidence_scores_class_arr.append(confidence_scores)
            predictions_class_arr.append(predictions)

        one_hot_predictions = []
        for j in range(len(labels)):
            tmp = [0]*self.num_classes

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

        # print(true_class_labels)
        # print(predicted_class_labels)

        # Compute metrics
        accuracy = accuracy_score(true_class_labels, predicted_class_labels)
        f1 = f1_score(true_class_labels, predicted_class_labels, average='macro')
        precision = precision_score(true_class_labels, predicted_class_labels, average='macro')
        recall = recall_score(true_class_labels, predicted_class_labels, average='macro')

        print(f'Accuracy: {accuracy}')
        print(f'F1 Score: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')


model = PNN()

# train_column(model, embeddings0, labels0, epochs=50)

"""# Task 1"""

import json
import torch
embeddings = []
labels = []

for i in range(5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/train/train_{i}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings.append(data['embeddings'][0])
            if data['label'] == 1:
                labels.append(0)
            else:
                labels.append(1)

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.train(0, embeddings=embeddings0, labels=labels0, epochs=50)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/test/test_{i}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        embeddings.append(data['embeddings'][0])
        if data['label'] == 1:
            labels.append([0])
        else:
            labels.append([1])

# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

print(embeddings0.size())  # This will show the shape of the embeddings tensor
print(labels0.size())     # This will show the shape of the labels tensor

model.test(embeddings0, labels0, 0)

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

model.add_network()
model.train(1, embeddings=embeddings0, labels=labels0, epochs=25)

"""# Test Classification Layer"""

embeddings = []
labels = []


for i in range(5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/train/train_{i}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 0:
                embeddings.append(data['embeddings'][0])
                labels.append(0)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/dos_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(1)

# for i in range(1, 5):
#     with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+info_train_file_{i}.json', 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             embeddings.append(data['embeddings'][0])
#             if data['label'] == 1:
#                 labels.append(2)
#             else:
#                 labels.append(0)



# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

train_output_layer(model, embeddings0, labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/test/test_{i}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 0:
            embeddings.append(data['embeddings'][0])
            labels.append(0)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(1)

# with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         embeddings.append(data['embeddings'][0])
#         if data['label'] == 1:
#             labels.append(2)
#         else:
#             labels.append(0)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

test_output_layer(model, embeddings0, labels0)

"""# Task 2"""

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
model.train(2, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []


for i in range(5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/train/train_{i}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 0:
                embeddings.append(data['embeddings'][0])
                labels.append(0)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/dos_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(1)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+info_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(2)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

train_output_layer(model, embeddings0, labels0)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/test/test_{i}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 0:
            embeddings.append(data['embeddings'][0])
            labels.append(0)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(1)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(2)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

test_output_layer(model, embeddings0, labels0)

"""# Task 3"""

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
model.train(3, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []


for i in range(5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/train/train_{i}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 0:
                embeddings.append(data['embeddings'][0])
                labels.append(0)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/dos_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(1)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+info_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(2)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/bypass_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(3)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

train_output_layer(model, embeddings0, labels0, epochs=25)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/test/test_{i}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 0:
            embeddings.append(data['embeddings'][0])
            labels.append(0)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(1)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(2)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/bypass_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(3)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

test_output_layer(model, embeddings0, labels0)

"""# Task 4"""

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
model.train(4, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []


for i in range(5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/train/train_{i}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 0:
                embeddings.append(data['embeddings'][0])
                labels.append(0)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/dos_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(1)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+info_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(2)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/bypass_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(3)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+priv_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(4)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

train_output_layer(model, embeddings0, labels0, epochs=50)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/test/test_{i}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 0:
            embeddings.append(data['embeddings'][0])
            labels.append(0)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(1)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(2)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/bypass_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(3)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+priv_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(4)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

test_output_layer(model, embeddings0, labels0)

"""# Task 5"""

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
model.train(5, embeddings=embeddings0, labels=labels0)

embeddings = []
labels = []


for i in range(5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/train/train_{i}.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 0:
                embeddings.append(data['embeddings'][0])
                labels.append(0)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/dos_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(1)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+info_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(2)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/bypass_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(3)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/+priv_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(4)

for i in range(1, 5):
    with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/train_embeddings/other_train_file_{i}.json', 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['label'] == 1:
                embeddings.append(data['embeddings'][0])
                labels.append(5)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

train_output_layer(model, embeddings0, labels0, epochs=50)

embeddings = []
labels = []

with open(f'/content/drive/MyDrive/CSCI544-Project/t5p_small_embeddings/test/test_{i}.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 0:
            embeddings.append(data['embeddings'][0])
            labels.append(0)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/dos_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(1)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+info_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(2)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/bypass_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(3)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/+priv_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(4)

with open(f'/content/drive/MyDrive/CSCI544-Project/data/class_wise_embeddings/test_embeddings/other_test_file.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['label'] == 1:
            embeddings.append(data['embeddings'][0])
            labels.append(5)


# Convert lists to tensors
embeddings0 = torch.tensor(embeddings)
labels0 = torch.tensor(labels)

test_output_layer(model, embeddings0, labels0)