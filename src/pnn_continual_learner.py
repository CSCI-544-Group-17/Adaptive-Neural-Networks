import json
import os
import time
import torch

from src.continual_learner import ContinualLearner
from src.ewc import EWC
from src.pnn_model import PNNModel
from src.replay import RepeatReplayer
from src.utils import load_tensors


class PNNContinualLearner(ContinualLearner):
    CLASSES = ["dos", "+info", "bypass", "+priv", "other"]

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

        for task_id in range(1, self.__tasks + 1):
            for subnetwork_index, clazz in enumerate(PNNContinualLearner.CLASSES):
                print("Task %d Class %s..." % (task_id, clazz))



        # for task_id in range(0, self.__tasks):
        #     print("Task %d..." % task_id)
        #     X_train, y_train = self.load_train_data(task_id)
        #     ewc = None
        #     similarity = 0.1
        #     if self.__repeat_enabled and task_id > 0:
        #         X_exemplar, y_exemplar = self.load_exemplars(task_id)
        #         ewc = EWC(self.__model.get_topology(), self.__model.get_loss_fn(), X_exemplar, y_exemplar, len(y_exemplar))
        #         similarity = RepeatReplayer.calculate_coefficient(X_train, X_exemplar)
        #         X_train = torch.cat((X_train, X_exemplar))
        #         y_train = torch.cat((y_train, y_exemplar))
        #     self.__model.train(X_train, y_train, self.__epochs, self.__batch_size, ewc, similarity)
        #     test_file_paths = self.__get_test_file_paths(task_id)
        #     X_test, y_test = load_tensors(test_file_paths)
        #     score = self.__model.evaluate(X_test, y_test)
        #     scores.append({self.__KEY_TASK_ID: task_id, self.__KEY_ACCURACY: self.__get_metric(score[0]),
        #                    self.__KEY_F1: self.__get_metric(score[1])})
        #     if self.__repeat_enabled:
        #         self.update_exemplars(task_id, X_train, y_train)
        # enabled = "enabled" if self.__repeat_enabled else "disabled"
        # with open(os.path.join(self.__results_directory, self.__RESULT_FILE_TEMPLATE % enabled), "w") as f:
        #     f.write(json.dumps(scores, indent=1))
        # print("Time taken: %.2fs\n" % (time.time() - start))


def main():
    learner = PNNContinualLearner(PNNModel(), "../embeddings/pnn", "../exemplars/pnn", "../results/pnn", 10, 100, True,
                                  4)
    learner.learn()


if __name__ == '__main__':
    main()
