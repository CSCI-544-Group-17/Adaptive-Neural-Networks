import os
import sys

import torch

from model import FNNTopology, Model
from replay import RepeatReplayer
from utils import load_tensors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(base_embeddings_path: str, base_exemplars_path: str):
    train_file_path = os.path.join(base_embeddings_path, "train/train.jsonl")
    X_train, y_train = load_tensors(train_file_path)
    model = Model(FNNTopology("FNN with averaged vectors", X_train.shape[1]))
    print("Initial model training...")
    model.train(X_train, y_train, 50, 100)
    scores = []
    for task_id in range(0, 5):
        train_file_path = os.path.join(base_embeddings_path, "train/train_%d.jsonl" % task_id)
        exemplars_file_path = os.path.join(base_exemplars_path, "exemplars_%d.jsonl" % (task_id - 1))
        test_file_path = os.path.join(base_embeddings_path, "test/test_%d.jsonl" % task_id)
        X_train, y_train = load_tensors(train_file_path, exemplars_file_path if task_id > 0 else None)
        X_test, y_test = load_tensors(test_file_path)
        print("Continual learning iteration %d..." % task_id)
        model.train(X_train, y_train, 50, 100)
        score = model.evaluate(X_test, y_test)
        scores.append(score)
        replayer = RepeatReplayer(model, base_exemplars_path, task_id)
        replayer.update_exemplars(X_train, y_train)
        print(score)


if __name__ == '__main__':
    base_embeddings_path = sys.argv[1]
    base_exemplars_path = sys.argv[2]
    main(base_embeddings_path, base_exemplars_path)
