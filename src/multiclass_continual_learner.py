import sys

from torch.nn import CrossEntropyLoss

from continual_learner import ContinualLearner
from model import Model
from topology import MulticlassFNNTopology, MulticlassFNNTopologySmall


class MulticlassContinualLearner(ContinualLearner):
    def __init__(self, model: Model, base_embeddings_path: str, base_exemplars_path: str, results_directory: str, epochs: int, batch_size: 100, repeat_enabled: bool):
        super().__init__(model, base_embeddings_path, base_exemplars_path, results_directory, epochs, batch_size, repeat_enabled, 4)


def repeat_enabled():
    model = Model(MulticlassFNNTopology("Multiclass FNN topology", 256, 5 + 1), CrossEntropyLoss(reduction='none'))
    learner = MulticlassContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 50, 100, True)
    learner.learn()


def repeat_disabled():
    model = Model(MulticlassFNNTopology("Multiclass FNN topology", 256, 5 + 1), CrossEntropyLoss(reduction='none'))
    learner = MulticlassContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 50, 100, False)
    learner.learn()


if __name__ == '__main__':
    repeat_enabled()
    repeat_disabled()
