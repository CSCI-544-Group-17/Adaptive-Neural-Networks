import copy

import torch
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

from topology import PytorchTopology, PNNTopology


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, pnn: PNNTopology, loss_fn: _Loss, X_exemplar: torch.Tensor, y_exemplar: torch.Tensor, length, subnetwork_index: int):
        self.pnn = pnn
        self.loss_fn = loss_fn
        self.X_exemplar = X_exemplar
        self.y_exemplar = y_exemplar
        self.length = length
        self.subnetwork_index = subnetwork_index
        self.column = self.pnn.model.subnetworks[subnetwork_index]
        self.params = {n: p for n, p in self.column.named_parameters() if p.requires_grad}
        self._means = {}  # previous parameters
        self._precision_matrices = self._diag_fisher()  # approximated diagnal fisher information matrix
        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        self.pnn.model.train()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        self.pnn.zero_grad()
        inputs = self.X_exemplar
        labels = self.y_exemplar
        y_pred = self.pnn.forward(inputs)
        loss = self.loss_fn(y_pred, labels).mean()
        loss.backward()
        for n, p in self.pnn.named_parameters():
            if p.grad is None:
                continue
            precision_matrices[n].data += p.grad.data ** 2 / self.length
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.pnn.zero_grad()
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
