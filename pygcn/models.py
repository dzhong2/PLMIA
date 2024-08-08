import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, DP_layer, perturb_adj
import torch
from utils import normalize_adj
import scipy.sparse as sp


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, device=torch.device("cuda:0")):
        super(GCN, self).__init__()
        self.device=device

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        adj_copy = adj.clone()
        adj_copy = normalize_adj(adj_copy.cpu() + torch.eye(adj.shape[1]).cpu())
        adj_copy = adj_copy.to(self.device)

        x = F.relu(self.gc1(x, adj_copy))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_copy)
        return F.softmax(x, dim=1)
