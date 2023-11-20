import json
import os
import time

import torch
from torch.utils.data import TensorDataset

from continual_learner import ContinualLearner
from ewc import EWC
from pnn_model import PNNModel
from replay import PNNReplayer
from utils import load_tensors


class PNNContinualLearner(ContinualLearner):
    CLASSES = ["nothing", "dos", "+info", "bypass", "+priv"]
    TRAIN_FILE_TEMPLATE = "%s_train_file_%d.json"
    EXEMPLAR_FILE_TEMPLATE = "%s_exemplar_file_%d.json"
    TEST_FILE_TEMPLATE = "%s_test_file.json"
    KEY_TASK_ID = "task_id"
    KEY_ACCURACY = "accuracy"
    KEY_F1 = "f1"
    RESULT_FILE_TEMPLATE = "result_%s.json"

    def __init__(self, model: PNNModel,
                 base_embeddings_path: str,
                 base_exemplars_path: str,
                 results_directory: str,
                 epochs: int,
                 batch_size: 100,
                 repeat_enabled: bool,
                 tasks: int):
        super().__init__("PNN Continual Learner")
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
        scores = []
        for subnetwork_index, clazz in enumerate(PNNContinualLearner.CLASSES):
            for task_id in range(1, self.__tasks + 1):
                print("Class %s Task %d..." % (clazz, task_id))
                X_train, y_train = self.load_train_data(subnetwork_index, task_id)
                if self.__model.should_add(subnetwork_index):
                    self.__model.add_network()
                ewc = None
                similarity = 0.1
                if self.__repeat_enabled and task_id > 1:
                    X_exemplar, y_exemplar = self.load_exemplars(subnetwork_index)
                    ewc = EWC(self.__model.get_subnetwork(subnetwork_index), self.__model.get_subnetwork(subnetwork_index).get_criterion(), X_exemplar, y_exemplar, len(y_exemplar))
                    similarity = PNNReplayer.calculate_coefficient(X_train, X_exemplar)
                    X_train = torch.cat((X_train, X_exemplar))
                    y_train = torch.cat((y_train, y_exemplar))
                train_dataset = TensorDataset(X_train, y_train)
                self.__model.train(subnetwork_index, train_dataset, self.__epochs, self.__batch_size, ewc, similarity)
                if self.__repeat_enabled:
                    self.update_exemplars(subnetwork_index, X_train, y_train)
            test_file_paths = self.__get_test_file_paths(subnetwork_index)
            X_test, y_test = load_tensors(test_file_paths)
            score = self.__model.evaluate(X_test, y_test)
            scores.append({self.KEY_TASK_ID: subnetwork_index, self.KEY_ACCURACY: self.__get_metric(score[0]),
                           self.KEY_F1: self.__get_metric(score[1])})
        enabled = "enabled" if self.__repeat_enabled else "disabled"
        with open(os.path.join(self.__results_directory, self.RESULT_FILE_TEMPLATE % enabled), "w") as f:
            f.write(json.dumps(scores, indent=1))
        print("Time taken: %.2fs\n" % (time.time() - start))

    def load_train_data(self, task_id: int, batch_index: int = 1):
        train_file_path = os.path.join(self.__base_embeddings_path, "train/%s_train_file_%d.json" % (PNNContinualLearner.CLASSES[task_id], batch_index))
        return load_tensors([train_file_path])

    def load_exemplars(self, task_id: int):
        exemplars_file_path = os.path.join(self.__base_exemplars_path, "exemplars_%d.jsonl" % task_id)
        return load_tensors([exemplars_file_path])

    def __get_test_file_paths(self, task_id: int):
        test_file_paths = []
        for i in range(task_id + 1):
            test_file_paths.append(os.path.join(self.__base_embeddings_path, "test/%s_test_file.json" % PNNContinualLearner.CLASSES[i]))
        print(test_file_paths)
        return test_file_paths

    def update_exemplars(self, subnetwork_index: int, X_train: torch.Tensor, y_train: torch.Tensor):
        replayer = PNNReplayer(self.__model, self.__base_exemplars_path, subnetwork_index)
        replayer.update_exemplars(X_train, y_train)

    @staticmethod
    def __get_metric(metric):
        if isinstance(metric, torch.Tensor):
            print(metric)
            return metric.tolist()
        return metric


def main():
    learner = PNNContinualLearner(PNNModel(), "../data/pnn", "../exemplars/pnn", "../results/pnn", 10, 32, True, 4)
    learner.learn()

    learner = PNNContinualLearner(PNNModel(), "../data/pnn", "../exemplars/pnn", "../results/pnn", 10, 32, False, 4)
    learner.learn()


if __name__ == '__main__':
    main()
