import os
import time
from os.path import join

import torch
from torch.utils.data import TensorDataset

from src.continual_learner import ContinualLearner
from src.ewc import EWC
from src.pnn_model import PNNModel
from src.replay import RepeatReplayer
from utils import load_tensors


class PNNContinualLearner(ContinualLearner):
    CLASSES = ["dos", "+info", "bypass", "+priv", "other"]
    TRAIN_FILE_TEMPLATE = "%s_train_file_%d.json"
    EXEMPLAR_FILE_TEMPLATE = "%s_exemplar_file_%d.json"
    TEST_FILE_TEMPLATE = "%s_test_file.json"

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
            print("Task %d Class %s..." % (subnetwork_index, clazz))
            if self.__model.should_add(subnetwork_index):
                self.__model.add_network()
            subnetwork = self.__model.get_subnetwork(subnetwork_index)
            X_train, y_train = self.load_train_data(clazz)
            ewc = None
            similarity = 0.1
            if self.__repeat_enabled and subnetwork_index > 1:
                X_exemplar, y_exemplar = self.load_exemplars(clazz, subnetwork_index)
                ewc = EWC(subnetwork, subnetwork.get_criterion(), X_exemplar, y_exemplar, len(y_exemplar))
                similarity = RepeatReplayer.calculate_coefficient(X_train, X_exemplar)
                X_train = torch.cat((X_train, X_exemplar))
                y_train = torch.cat((y_train, y_exemplar))
            train_dataset = TensorDataset(X_train, y_train)
            self.__model.train(subnetwork_index, train_dataset, self.__epochs, ewc, similarity)
            X_test, y_test = self.load_test_data(clazz)
            score = self.__model.evaluate(X_test, y_test)
            print(score)

    def load_train_data(self, clazz: str):
        train_file_paths = [
            join(self.__base_embeddings_path, "train", file_name) for file_name in
            os.listdir(os.path.join(self.__base_embeddings_path, "train"))
            if clazz in file_name
        ]
        return load_tensors(train_file_paths)

    def load_exemplars(self, clazz: str, subnetwork_id: int):
        exemplar_file_path = join(self.__base_exemplars_path, PNNContinualLearner.EXEMPLAR_FILE_TEMPLATE % (clazz, subnetwork_id))
        return load_tensors([exemplar_file_path])

    def load_test_data(self, clazz: str):
        test_file_path = join(self.__base_embeddings_path, "test", PNNContinualLearner.TEST_FILE_TEMPLATE % clazz)
        return load_tensors([test_file_path])

    def update_exemplars(self, *args):
        pass


def main():
    learner = PNNContinualLearner(PNNModel(), "../embeddings/pnn", "../exemplars/pnn", "../results/pnn", 10, 100, False,
                                  4)
    learner.learn()


if __name__ == '__main__':
    main()
