import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DP_layer(torch.nn.Module):
    def __init__(self, lbd=1.0):
        super(DP_layer, self).__init__()
        self.lbd = lbd
        self.C = 10

    def forward(self, h):
        l2_norm = h.norm(2)
        divisor = max(l2_norm/self.C, 1.0)
        h = h/divisor
        #noise = self.sigma/10 * torch.randn_like(h)
        noise = lap_noise(h, self.lbd)
        return h + noise


def lap_noise(h, lbd):
    loc = 0
    scale = lbd
    noise = torch.Tensor(np.random.laplace(loc, scale, h.shape))
    return noise


class perturb_adj(torch.nn.Module):
    def __init__(self, gamma=1.0):
        super(perturb_adj, self).__init__()
        self.gamma = gamma

    def forward(self, adj):
        Ne = torch.div(adj.sum(), 2, rounding_mode='floor')
        Nu = torch.div(adj.shape[0] * (adj.shape[0] - 1), 2)
        self.p1 = Nu/(np.exp(self.gamma) * Ne + Nu)
        self.p2 = Ne/(np.exp(self.gamma) * Ne + Nu)
        prob = adj * (1 - self.p1) + (1-adj) * (self.p2)
        adj_ptb = torch.bernoulli(prob)
        # modify adj
        torch.triu(adj_ptb).sum() + torch.triu(adj_ptb).T
        return adj_ptb
