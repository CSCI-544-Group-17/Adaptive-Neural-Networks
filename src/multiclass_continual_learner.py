import sys

from torch.nn import CrossEntropyLoss

from continual_learner import ContinualLearner
from model import Model
from topology import MulticlassFNNTopologySmall


class MulticlassContinualLearner(ContinualLearner):
    def __init__(self, model: Model, base_embeddings_path: str, base_exemplars_path: str, results_directory: str, epochs: int, batch_size: 100, repeat_enabled: bool):
        super().__init__(model, base_embeddings_path, base_exemplars_path, results_directory, epochs, batch_size, repeat_enabled, 5)


def repeat_enabled():
    model = Model(MulticlassFNNTopologySmall("Multiclass FNN topology", 256, 5 + 1), CrossEntropyLoss())
    learner = MulticlassContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 150, 100, True)
    learner.learn()


def repeat_disabled():
    model = Model(MulticlassFNNTopologySmall("Multiclass FNN topology", 256, 5 + 1), CrossEntropyLoss())
    learner = MulticlassContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 150, 100, False)
    learner.learn()


if __name__ == '__main__':
    repeat_enabled()
    repeat_disabled()
