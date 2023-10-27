import json
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_tensors(file_path):
    X = []
    y = []
    with open(file_path) as f:
        line = f.readline()
        while True:
            if not line:
                break
            data = json.loads(line)
            X.append(data["embeddings"])
            y.append(data["label"])
            line = f.readline()
    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X, dtype=torch.float32).clone().detach().to(DEVICE).reshape(-1, 256)
    y = torch.tensor(y, dtype=torch.float32).clone().detach().to(DEVICE).reshape(-1, 1)
    return X, y
