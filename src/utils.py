import json
from typing import List

import numpy as np
import torch

KEY_LABEL = "label"
KEY_EMBEDDINGS = "embeddings"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tensors(files: List):
    X = []
    y = []
    for file in files:
        __load_and_append(file, X, y)
    X = np.array(X[:100])
    y = np.array(y[:100])
    X = torch.tensor(X, dtype=torch.float32).clone().detach().to(DEVICE).reshape(-1, 768)
    y = torch.tensor(y, dtype=torch.long).clone().detach().to(DEVICE).reshape((-1))
    return X, y


def __load_and_append(file_path: str, X: List, y: List):
    with open(file_path) as f:
        line = f.readline()
        while True:
            if not line:
                break
            data = json.loads(line)
            X.append(data[KEY_EMBEDDINGS])
            if not isinstance(data[KEY_LABEL], list):
                y.append([data[KEY_LABEL]])
            else:
                y.append(data[KEY_LABEL])
            line = f.readline()
