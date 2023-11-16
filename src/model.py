from torch.nn.modules.loss import _Loss
from torch.nn.modules.loss import _Loss
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
from tqdm import tqdm

from pnn import *
from topology import PNNTopology


class Model:
    """
    Defines the model topology, training and evaluation functions
    """

    def __init__(self, pnn: PNNTopology, loss_fn: _Loss):
        self.__pnn = pnn
        self.__loss_fn = loss_fn
        # self.__optimizer = optim.Adam(topology.parameters(), lr=0.0001)
        pnn_params = self.__pnn.get_all_parameters()
        self.__optimizer = torch.optim.Adam(pnn_params, lr=0.0001)

    def get_pnn(self):
        return self.__pnn

    def get_loss_fn(self):
        return self.__loss_fn

    """
    def train(self, X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, epochs: int, batch_size: int,
              ewc: EWC = None, similarity: float = None):

        # Need to modify the loop
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
                """

    def train(self, X_train_tensor, y_train_tensor, epochs, batch_size, ewc=None, similarity=None,
              subnetwork_index=None):
        self.__pnn.model.train()
        is_pnn = isinstance(self.__pnn, PNNTopology)  # Check if topology is PNN

        with tqdm(total=epochs) as bar:
            for epoch in range(epochs):

                for i in range(0, len(X_train_tensor), batch_size):
                    X_batch = X_train_tensor[i:i + batch_size]
                    y_batch = y_train_tensor[i:i + batch_size]

                    if is_pnn and subnetwork_index is not None:
                        # Special handling for PNN
                        column = self.__pnn.model.subnetworks[subnetwork_index]
                        freeze_params(self.__pnn.model, subnetwork_index)
                        y_pred = forward(column, X_batch)
                        loss = calculate_loss(self.__pnn.model, subnetwork_index, y_pred, y_batch)
                        if ewc is not None:
                            ewc_loss = ewc.penalty(self.__pnn.model)
                            loss += 2000 * ewc_loss * similarity
                        loss.backward()
                        column.optimizer.step()
                        column.optimizer.zero_grad()
                        unfreeze_params(self.__pnn.model)
                    else:
                        # Normal training for other topologies
                        y_pred = self.__pnn.model(X_batch)
                        loss = self.__loss_fn(y_pred, y_batch).mean()

                        if ewc is not None:
                            ewc_loss = ewc.penalty(self.__pnn.model)
                            loss += 2000 * ewc_loss * similarity

                        self.__optimizer.zero_grad()
                        loss.backward()
                        self.__optimizer.step()

                    bar.set_description("Loss: %f" % loss.item())
                    bar.update()

    """    
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
            """

    def evaluate(self, X_test, y_test):
        if isinstance(self.__pnn, PNNTopology):
            # Special handling for PNN
            return self.evaluate_pnn(X_test, y_test)
        else:
            # Normal evaluation for other topologies
            self.__pnn.eval()
            with torch.no_grad():
                y_pred = self.__pnn(X_test)
                num_classes = y_pred.shape[1]
                y_pred = torch.argmax(y_pred, dim=1)
                acc_metric = MulticlassAccuracy(num_classes=num_classes)
                accuracy = acc_metric(y_pred, y_test).mul(100)
                f1_metric = MulticlassF1Score(num_classes=num_classes, average=None)
                f1_score = f1_metric(y_pred, y_test).mul(100)
                return accuracy, f1_score

    def evaluate_pnn(self, X_test, y_test):
        # Initialize arrays to store scores and predictions
        self.__pnn.model.eval()
        confidence_scores_class_arr = []
        predictions_class_arr = []
        num_classes = self.__pnn.model.num_classes

        # Evaluate each subnetwork
        for i in range(num_classes):
            logits, predicted = test_column(self.__pnn.model.subnetworks[i], X_test, y_test)
            confidence_scores_class_arr.append(logits)
            predictions_class_arr.append(predicted)

        # Process predictions
        one_hot_predictions = []
        for j in range(len(y_test)):
            tmp = [0] * num_classes
            max_confidence = float('-inf')
            max_idx = -1
            for i in range(num_classes):
                if confidence_scores_class_arr[i][j] > max_confidence and predictions_class_arr[i][j] == 1:
                    max_confidence = confidence_scores_class_arr[i][j]
                    max_idx = i
            if max_idx != -1:
                tmp[max_idx] = 1
            one_hot_predictions.append(tmp)

        # Convert to class labels
        true_class_labels = y_test.tolist()
        predicted_class_labels = [pred.index(1) if 1 in pred else 0 for pred in one_hot_predictions]

        # Compute metrics
        accuracy = accuracy_score(true_class_labels, predicted_class_labels)
        f1 = f1_score(true_class_labels, predicted_class_labels, average='macro')
        # precision = precision_score(true_class_labels, predicted_class_labels, average='macro')
        # recall = recall_score(true_class_labels, predicted_class_labels, average='macro')

        return accuracy, f1

    def get_loss(self, X: torch.Tensor, y: torch.Tensor, batch_size: int, subnetwork_index: int) -> torch.Tensor:
        self.__pnn.model.eval()
        column = self.__pnn.model.subnetworks[subnetwork_index]
        losses = torch.zeros(size=(y.shape[0], 1))
        for i in range(0, len(X), batch_size):
            with torch.no_grad():
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = forward(column, X_batch)
                loss = calculate_loss(self.__pnn.model, subnetwork_index, y_pred, y_batch)
                loss = loss.reshape(-1, 1)
                losses[i:i + batch_size] = loss
        return losses
