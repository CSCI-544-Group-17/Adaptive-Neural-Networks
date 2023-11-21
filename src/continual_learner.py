import torch


class ContinualLearner:
    def __init__(self, name: str):
        self.__name = name

    def load_train_data(self, *args):
        raise NotImplementedError("'load_train_data' not implemented")

    def load_exemplars(self, *args):
        raise NotImplementedError("'load_exemplars' not implemented")

    def load_test_data(self, *args):
        raise NotImplementedError("'load_test_data' not implemented")

    def learn(self):
        raise NotImplementedError("'learn' not implemented")

    def update_exemplars(self, *args):
        raise NotImplementedError("'update_exemplars' not implemented")
