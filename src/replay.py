import json
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.cluster import KMeans

from model import Model
from utils import load_tensors

EMBEDDINGS_READ_PATH = "../t5p_small_embeddings/"


class Replayer:
    def load_exemplars(self):
        raise NotImplementedError("Please implement the load_exemplars function")

    def update_exemplars(self, X: torch.Tensor, y: torch.Tensor):
        raise NotImplementedError("Please implement the update_exemplars function")


class RepeatReplayer(Replayer):
    def __init__(self, model: Model, exemplars_directory: str, task_id: int):
        self.__model = model
        self.__exemplars_directory = exemplars_directory
        self.__cluster_pick = 0.2
        self.__M = 1000
        self.__task_id = task_id

    def load_exemplars(self):
        return load_tensors(self.__exemplars_directory)

    def update_exemplars(self, X: torch.Tensor, y: torch.Tensor):
        exemplars = []
        current_exemplars = self.__pick_current(X, y)
        exemplars.extend(current_exemplars)
        with open(os.path.join(self.__exemplars_directory, "exemplars_%d.jsonl" % self.__task_id), "w") as f:
            for exemplar in exemplars:
                f.write(json.dumps(exemplar) + "\n")

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
            results.append({"embeddings": [X[exemplar.index].tolist()], "label": int(y[exemplar.index].item())})
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
