import sys

from torch.nn import CrossEntropyLoss

from basic_continual_learner import BasicContinualLearner
from basic_model import BasicModel
from topology import MulticlassFNNTopologySmall


class MulticlassBasicContinualLearner(BasicContinualLearner):
    def __init__(self, model: BasicModel, base_embeddings_path: str, base_exemplars_path: str, results_directory: str, epochs: int, batch_size: 100, repeat_enabled: bool):
        super().__init__(model, base_embeddings_path, base_exemplars_path, results_directory, epochs, batch_size, repeat_enabled, 5)


def repeat_enabled():
    model = BasicModel(MulticlassFNNTopologySmall("Multiclass FNN topology", 256, 5 + 1), CrossEntropyLoss())
    learner = MulticlassBasicContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 150, 100, True)
    learner.learn()


def repeat_disabled():
    model = BasicModel(MulticlassFNNTopologySmall("Multiclass FNN topology", 256, 5 + 1), CrossEntropyLoss())
    learner = MulticlassBasicContinualLearner(model, sys.argv[1], sys.argv[2], sys.argv[3], 150, 100, False)
    learner.learn()


if __name__ == '__main__':
    repeat_enabled()
    repeat_disabled()
