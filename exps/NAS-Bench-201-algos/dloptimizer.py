# -*- coding: utf-8 -*-
# @Time    : 2021/9/15 2:18
# @Author  : LIU YI

import copy
from torch.optim import Optimizer
import torch


class dlOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda)
        super(dlOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, aggregated_models):
        loss = None
        aggregated_models = [aggregated_models]
        for group in self.param_groups:

            for i in range(len(group['params'])):
                p = group['params'][i]
                to_be_subtracted = copy.deepcopy(aggregated_models[0][i])

                for j in range(1, len(aggregated_models)):
                    to_be_subtracted += list(aggregated_models[j].values())[i]

                if p.grad!=None:
                    p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (len(aggregated_models) * p.data - to_be_subtracted))
                else:
                    p.data = p.data - group['lr'] * (group['lamda'] * (len(aggregated_models) * p.data - to_be_subtracted))


        return group['params'], loss