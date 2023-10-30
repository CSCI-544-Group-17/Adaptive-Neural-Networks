import json
import os
from typing import Dict, List

import torch
from sklearn.cluster import KMeans

import utils
from model import Model
from utils import load_tensors


class Replayer:
    def load_exemplars(self):
        raise NotImplementedError("Please implement the load_exemplars function")

    def update_exemplars(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError("Please implement the update_exemplars function")


class RepeatReplayer(Replayer):
    __EXEMPLARS_FILE_TEMPLATE = "exemplars_%d.jsonl"

    def __init__(self, model: Model, exemplars_directory: str, task_id: int):
        self.__model = model
        self.__exemplars_directory = exemplars_directory
        self.__M = 1000
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

    @staticmethod
    def calculate_coefficient(new_data: torch.Tensor, replay_data: torch.Tensor):
        replay_feature = torch.mean(replay_data, dim=0)
        new_feature = torch.mean(new_data, dim=0)
        a = torch.dot(replay_feature, new_feature)
        b = torch.norm(replay_feature)
        c = torch.norm(new_feature)
        return a / (b * c)

    def __pick_current(self, X: torch.Tensor, y: torch.Tensor):
        kmeans = KMeans(n_clusters=5, init='k-means++', n_init='auto')
        labels = kmeans.fit_predict(X)
        losses = self.__model.get_loss(X, y, 100)
        exemplars = [Exemplar(i, labels[i], losses[i].item()) for i in range(len(labels))]
        classes: Dict[int, List[Exemplar]] = {}
        for exemplar in exemplars:
            if exemplar.label not in classes.keys():
                classes[exemplar.label] = []
            classes[exemplar.label].append(exemplar)
        current_exemplars = []
        m = self.__M // (self.__task_id + 1)
        for key in classes.keys():
            classes[key] = sorted(classes[key], key=lambda e: e.loss)
            current_exemplars.extend(classes[key][:m])
        results = []
        for exemplar in current_exemplars:
            results.append({
                utils.KEY_EMBEDDINGS: [X[exemplar.index].tolist()],
                utils.KEY_LABEL: int(y[exemplar.index].item())
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
                    utils.KEY_LABEL: int(y[exemplar.index].item())
                })
        return results


class Exemplar:
    def __init__(self, index: int, label: int, loss: float):
        self.index = index
        self.label = label
        self.loss = loss

    def __str__(self) -> str:
        return "(%s, %s, %s)" % (self.index, self.label, self.loss)

    def __repr__(self):
        return self.__str__()