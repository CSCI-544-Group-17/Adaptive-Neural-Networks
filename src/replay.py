import json
import os
from typing import Dict, List

import torch
from sklearn.cluster import KMeans

import utils
from fnn_model import FNNModel
from pnn_model import PNNModel
from utils import load_tensors


class Replayer:
    def load_exemplars(self):
        raise NotImplementedError("Please implement the load_exemplars function")

    def update_exemplars(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError("Please implement the update_exemplars function")

    def _get_loss(self, *args):
        raise NotImplementedError("Please implement the get_loss function")


class RepeatReplayer(Replayer):
    __EXEMPLARS_FILE_TEMPLATE = "exemplars_%d.jsonl"

    def __init__(self, model: FNNModel, exemplars_directory: str, task_id: int):
        self.__model = model
        self.__exemplars_directory = exemplars_directory
        self.__M = 100
        self.__task_id = task_id

    def load_exemplars(self):
        return load_tensors([self.__exemplars_directory])

    def update_exemplars(self, X: torch.Tensor, y: torch.Tensor):
        exemplars = []
        current_exemplars = self.__pick_current(X, y)
        old_exemplars = self.__pick_old()
        exemplars.extend(current_exemplars)
        exemplars.extend(old_exemplars)
        with open(os.path.join(self.__exemplars_directory, self.__EXEMPLARS_FILE_TEMPLATE % self.__task_id), "w") as f:
            for exemplar in exemplars:
                f.write(json.dumps(exemplar) + "\n")

    def _get_loss(self, X: torch.Tensor, y: torch.Tensor):
        return self.__model.get_loss(X, y, 100)

    @staticmethod
    def calculate_coefficient(new_data: torch.Tensor, replay_data: torch.Tensor):
        replay_feature = torch.mean(replay_data, dim=0)
        new_feature = torch.mean(new_data, dim=0)
        a = torch.dot(replay_feature, new_feature)
        b = torch.norm(replay_feature)
        c = torch.norm(new_feature)
        return a / (b * c)

    def __pick_current(self, X: torch.Tensor, y: torch.Tensor):
        kmeans = KMeans(n_clusters=10, init='k-means++', n_init='auto')
        labels = kmeans.fit_predict(X)
        losses = self.__model.get_loss(X, y, 100)
        exemplars = [Exemplar(i, labels[i], losses[i].item()) for i in range(len(labels))]
        classes: Dict[int, List[Exemplar]] = {}
        for exemplar in exemplars:
            if exemplar.label not in classes.keys():
                classes[exemplar.label] = []
            classes[exemplar.label].append(exemplar)
        current_exemplars = []
        m = 2 * self.__M // (self.__task_id + 1)
        for key in classes.keys():
            classes[key] = sorted(classes[key], key=lambda e: e.loss)
            current_exemplars.extend(classes[key][:m])
        results = []
        for exemplar in current_exemplars:
            results.append({
                utils.KEY_EMBEDDINGS: [X[exemplar.index].tolist()],
                utils.KEY_LABEL: self.__get_label(y[exemplar.index])
            })
        return results

    def __pick_old(self):
        results = []
        for task_id in range(0, self.__task_id):
            exemplars_file_path = os.path.join(self.__exemplars_directory, self.__EXEMPLARS_FILE_TEMPLATE % task_id)
            X, y = load_tensors([exemplars_file_path])
            losses = self.__model.get_loss(X, y, 100)
            exemplars = [Exemplar(i, 0, losses[i].item()) for i in range(len(losses))]
            exemplars = sorted(exemplars, key=lambda e: e.loss)
            M = self.__M
            t = self.__task_id + 2
            m = M * (t * t - t - 1) // (t * t - 1)
            exemplars = exemplars[:m]
            for exemplar in exemplars:
                results.append({
                    utils.KEY_EMBEDDINGS: [X[exemplar.index].tolist()],
                    utils.KEY_LABEL: self.__get_label(y[exemplar.index])
                })
        return results

    @staticmethod
    def __get_label(y_i: torch.Tensor):
        return int(y_i.item())

class PNNReplayer(Replayer):
    __EXEMPLARS_FILE_TEMPLATE = "exemplars_%d.jsonl"

    def __init__(self, model: PNNModel, exemplars_directory: str, subnetwork_index: int):
        self.__model = model
        self.__exemplars_directory = exemplars_directory
        self.__M = 100
        self.__subnetwork_index = subnetwork_index

    def load_exemplars(self):
        return load_tensors([self.__exemplars_directory])

    def update_exemplars(self, X: torch.Tensor, y: torch.Tensor):
        exemplars = []
        current_exemplars = self.__pick_current(X, y)
        exemplars.extend(current_exemplars)
        with open(os.path.join(self.__exemplars_directory, self.__EXEMPLARS_FILE_TEMPLATE % self.__subnetwork_index), "w") as f:
            for exemplar in exemplars:
                f.write(json.dumps(exemplar) + "\n")

    @staticmethod
    def calculate_coefficient(new_data: torch.Tensor, replay_data: torch.Tensor):
        replay_feature = torch.mean(replay_data, dim=0)
        new_feature = torch.mean(new_data, dim=0)
        a = torch.dot(replay_feature, new_feature)
        b = torch.norm(replay_feature)
        c = torch.norm(new_feature)
        return a / (b * c)

    def __pick_current(self, X: torch.Tensor, y: torch.Tensor):
        kmeans = KMeans(n_clusters=10, init='k-means++', n_init='auto')
        labels = kmeans.fit_predict(X)
        losses = self.__model.get_loss(self.__subnetwork_index, X, y)
        exemplars = [Exemplar(i, labels[i], losses[i].item()) for i in range(len(labels))]
        classes: Dict[int, List[Exemplar]] = {}
        for exemplar in exemplars:
            if exemplar.label not in classes.keys():
                classes[exemplar.label] = []
            classes[exemplar.label].append(exemplar)
        current_exemplars = []
        m = 2 * self.__M // 4
        for key in classes.keys():
            classes[key] = sorted(classes[key], key=lambda e: e.loss)
            current_exemplars.extend(classes[key][:m])
        results = []
        for exemplar in current_exemplars:
            results.append({
                utils.KEY_EMBEDDINGS: [X[exemplar.index].tolist()],
                utils.KEY_LABEL: self.__get_label(y[exemplar.index])
            })
        return results

    def __pick_old(self):
        results = []
        for task_id in range(0, self.__subnetwork_index):
            exemplars_file_path = os.path.join(self.__exemplars_directory, self.__EXEMPLARS_FILE_TEMPLATE % task_id)
            X, y = load_tensors([exemplars_file_path])
            losses = self.__model.get_loss(self.__subnetwork_index, X, y)
            exemplars = [Exemplar(i, 0, losses[i].item()) for i in range(len(losses))]
            exemplars = sorted(exemplars, key=lambda e: e.loss)
            M = self.__M
            t = self.__subnetwork_index + 2
            m = M * (t * t - t - 1) // (t * t - 1)
            exemplars = exemplars[:m]
            for exemplar in exemplars:
                results.append({
                    utils.KEY_EMBEDDINGS: [X[exemplar.index].tolist()],
                    utils.KEY_LABEL: self.__get_label(y[exemplar.index])
                })
        return results

    @staticmethod
    def __get_label(y_i: torch.Tensor):
        return int(y_i.item())


class Exemplar:
    def __init__(self, index: int, label: int, loss: float):
        self.index = index
        self.label = label
        self.loss = loss

    def __str__(self) -> str:
        return "(%s, %s, %s)" % (self.index, self.label, self.loss)

    def __repr__(self):
        return self.__str__()
