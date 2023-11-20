from torch.nn import CrossEntropyLoss

from basic_continual_learner import BaseFNNContinualLearner
from fnn_model import FNNModel
from topology import MulticlassFNNTopologySmall


class FNNContinualLearner(BaseFNNContinualLearner):
    def __init__(self, model: FNNModel, base_embeddings_path: str, base_exemplars_path: str, results_directory: str, epochs: int, batch_size: 100, repeat_enabled: bool):
        super().__init__(model, base_embeddings_path, base_exemplars_path, results_directory, epochs, batch_size, repeat_enabled, 5)


def repeat_enabled():
    model = FNNModel(MulticlassFNNTopologySmall("Multiclass FNN topology", 256, 5), CrossEntropyLoss(reduction='none'))
    learner = FNNContinualLearner(model, "../data/fnn", "../exemplars/fnn", "../results/fnn", 150, 16, True)
    learner.learn()


def repeat_disabled():
    model = FNNModel(MulticlassFNNTopologySmall("Multiclass FNN topology", 256, 5), CrossEntropyLoss(reduction='none'))
    learner = FNNContinualLearner(model, "../data/fnn", "../exemplars/fnn", "../results/fnn", 150, 16, False)
    learner.learn()


if __name__ == '__main__':
    repeat_enabled()
    repeat_disabled()
