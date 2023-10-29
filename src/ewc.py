import torch
import torch.nn as nn
from torch.autograd import Variable
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class EWC(object):
    def __init__(self, model, dataset, length):

        self.model = model
        self.dataset = dataset
        self.length = length

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {} # previous parameters
        self._precision_matrices = self._diag_fisher() # approximated diagnal fisher information matrix

        for n, p in copy.deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):

        self.model.train()
        precision_matrices = {}
        for n, p in copy.deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        for step,batch in enumerate(self.dataset):
            self.model.zero_grad()
            
            inputs = batch[0]       
            labels=batch[1]

            self.model.train()


            y_pred = self.__topology.forward(inputs)
            loss = self.__loss_fn(y_pred, labels).mean()

            # loss,logits = self.model(inputs,labels)

            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                precision_matrices[n].data += p.grad.data ** 2 / self.length

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        self.model.zero_grad()
        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()

        return loss
