import sys

from torch.nn import CrossEntropyLoss

from basic_continual_learner import BaseFNNContinualLearner
from fnn_model import FNNModel
from topology import BinaryFNNTopology


class BinaryBasicContinualLearner(BaseFNNContinualLearner):
    def __init__(self, model: FNNModel, base_embeddings_path: str, base_exemplars_path: str, results_directory: str, epochs: int,
                 batch_size: 100, repeat_enabled):
        super().__init__(model, base_embeddings_path, base_exemplars_path, results_directory, epochs, batch_size, repeat_enabled, 5)


if __name__ == '__main__':
    model = FNNModel(BinaryFNNTopology("FNN for binary classification", 256), CrossEntropyLoss())
    learner = BinaryBasicContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 50, 100, True)
    learner.learn()

    model = FNNModel(BinaryFNNTopology("FNN for binary classification", 256), CrossEntropyLoss())
    learner = BinaryBasicContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 50, 100, False)
    learner.learn()
