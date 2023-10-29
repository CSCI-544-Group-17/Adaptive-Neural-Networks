import json
from typing import List

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tensors(file_path, exemplars_file_path=None):
    X = []
    y = []
    __load_and_append(file_path, X, y)
    if exemplars_file_path is not None:
        __load_and_append(exemplars_file_path, X, y)
    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X, dtype=torch.float32).clone().detach().to(DEVICE).reshape(-1, 256)
    y = torch.tensor(y, dtype=torch.float32).clone().detach().to(DEVICE).reshape(-1, 1)
    return X, y


def __load_and_append(file_path: str, X: List, y: List):
    with open(file_path) as f:
        line = f.readline()
        while True:
            if not line:
                break
            data = json.loads(line)
            X.append(data["embeddings"])
            y.append(data["label"])
            line = f.readline()
