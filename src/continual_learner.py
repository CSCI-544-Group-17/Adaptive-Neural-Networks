import json
import os
import time

import torch

from ewc import EWC
from model import Model
from replay import RepeatReplayer
from utils import load_tensors


class ContinualLearner:
    __KEY_TASK_ID = "task_id"
    __KEY_ACCURACY = "accuracy"
    __KEY_F1 = "f1"
    __RESULT_FILE_TEMPLATE = "result_%s.json"

    def __init__(self, model: Model,
                 base_embeddings_path: str,
                 base_exemplars_path: str,
                 results_directory: str,
                 epochs: int,
                 batch_size: 100,
                 repeat_enabled: bool,
                 tasks: int):
        self.__model = model
        self.__base_embeddings_path = base_embeddings_path
        self.__base_exemplars_path = base_exemplars_path
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__results_directory = results_directory
        self.__repeat_enabled = repeat_enabled
        self.__tasks = tasks

    def learn(self):
        start = time.time()
        if self.__repeat_enabled:
            print("REPEAT enabled!")
        train_file_path = os.path.join(self.__base_embeddings_path, "train/train.jsonl")
        X_train, y_train = load_tensors([train_file_path])
        print("Initial model training...")
        self.__model.train(X_train, y_train, self.__epochs, self.__batch_size)
        scores = []
        for task_id in range(0, self.__tasks):
            print("Task %d..." % task_id)
            train_file_path = os.path.join(self.__base_embeddings_path, "train/train_%d.jsonl" % task_id)
            X_train, y_train = load_tensors([train_file_path])
            ewc = None
            similarity = 0.1
            if self.__repeat_enabled and task_id > 0:
                exemplars_file_path = os.path.join(self.__base_exemplars_path, "exemplars_%d.jsonl" % (task_id - 1))
                X_exemplar, y_exemplar = load_tensors([exemplars_file_path])
                ewc = EWC(self.__model.get_topology(), self.__model.get_loss_fn(), X_exemplar, y_exemplar,
                          len(y_exemplar))
                similarity = RepeatReplayer.calculate_coefficient(X_train, X_exemplar)
                X_train = torch.cat((X_train, X_exemplar))
                y_train = torch.cat((y_train, y_exemplar))
            self.__model.train(X_train, y_train, self.__epochs, self.__batch_size, ewc, similarity)
            test_file_paths = self.__get_test_file_paths(task_id)
            X_test, y_test = load_tensors(test_file_paths)
            score = self.__model.evaluate(X_test, y_test)
            scores.append({self.__KEY_TASK_ID: task_id, self.__KEY_ACCURACY: score[0], self.__KEY_F1: score[1]})
            if self.__repeat_enabled:
                replayer = RepeatReplayer(self.__model, self.__base_exemplars_path, task_id)
                replayer.update_exemplars(X_train, y_train)
        enabled = "enabled" if self.__repeat_enabled else "disabled"
        with open(os.path.join(self.__results_directory, self.__RESULT_FILE_TEMPLATE % enabled), "w") as f:
            f.write(json.dumps(scores, indent=1))
        print("Time taken: %.2fs\n" % (time.time() - start))

    def __get_test_file_paths(self, task_id: int):
        test_file_paths = []
        for i in range(task_id + 1):
            test_file_paths.append(os.path.join(self.__base_embeddings_path, "test/test_%d.jsonl" % i))
        return test_file_paths