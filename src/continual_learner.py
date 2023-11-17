import torch


class ContinualLearner:
    def __init__(self, name: str):
        self.__name = name

    def load_train_data(self, task_id: int):
        raise NotImplementedError("'load_train_data' not implemented")

    def load_exemplars(self, task_id: int):
        raise NotImplementedError("'load_exemplars' not implemented")

    def learn(self):
        raise NotImplementedError("'learn' not implemented")

    def update_exemplars(self, task_id: int, X_train: torch.Tensor, y_train: torch.Tensor):
        raise NotImplementedError("'update_exemplars' not implemented")
