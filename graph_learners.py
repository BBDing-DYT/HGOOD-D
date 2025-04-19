# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Attentive
from utils import *
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv
from wgin_conv import WGINConv

class ATT_learner(nn.Module):
    def __init__(self, num_gc_layers, isize, hidden_dim, drop_ratio):
        super(ATT_learner, self).__init__()

        # self.i = i
        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = hidden_dim
        self.layers = nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(num_gc_layers):
            self.layers.append(Attentive(isize))
            bn = torch.nn.BatchNorm1d(isize)
            self.bns.append(bn)
        # self.k = k
        # self.knn_metric = knn_metric
        # self.non_linearity = 'relu'
        # self.sparse = sparse
        # self.mlp_act = mlp_act

    # def internal_forward(self, h):
    #     # for i, layer in enumerate(self.layers):
    #     for i in range(self.num_gc_layers):
    #         h = self.layers[i](h)
    #         h = self.bns[i](h)
    #         if i != (len(self.layers) - 1):
    #             if self.mlp_act == "relu":
    #                 h = F.relu(h)
    #             elif self.mlp_act == "tanh":
    #                 h = F.tanh(h)
    #     return h

    def forward(self, x, edge_index):
        for i in range(self.num_gc_layers):
            x = self.layers[i](x)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
            #     # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            # if i != (len(self.layers) - 1):
            #     if self.mlp_act == "relu":
            #         h = F.relu(h)
            #     elif self.mlp_act == "tanh":
            #         h = F.tanh(h)
        return x

class GNN_learner(nn.Module):
    def __init__(self, num_gc_layers, isize, hidden_dim, drop_ratio):
        super(GNN_learner, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = hidden_dim * num_gc_layers
        self.convs  = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                mlp = Sequential(Linear(isize, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            conv = GINConv(mlp)
            # conv = WGINConv(mlp)
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index):
        xs = []
        for i in range(self.num_gc_layers):
            # x = F.relu(self.convs[i](x, edge_index))
            # x = F.relu(self.convs[i](x, edge_index, None))

            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
                # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
            xs.append(x)
        return torch.cat(xs, dim=1)
    # def forward(self, x, edge_index):
    #     for i in range(self.num_gc_layers):
    #         # x = F.relu(self.convs[i](x, edge_index))
    #         # x = F.relu(self.convs[i](x, edge_index, None))
    #
    #         x = self.convs[i](x, edge_index)
    #         x = self.bns[i](x)
    #         if i == self.num_gc_layers - 1:
    #             # remove relu for the last layer
    #             x = F.dropout(x, self.drop_ratio, training=self.training)
    #         else:
    #             x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
    #     return x


class MLP_learner(nn.Module):
    def __init__(self, num_gc_layers, isize, hidden_dim, drop_ratio):
        super(MLP_learner, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.drop_ratio = drop_ratio
        self.out_node_dim = hidden_dim
        self.convs  = nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                mlp = Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            else:
                mlp = Sequential(Linear(isize, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            bn = torch.nn.BatchNorm1d(hidden_dim)
            self.convs.append(mlp)
            self.bns.append(bn)


    def forward(self, x, edge_index):
        for i in range(self.num_gc_layers):
            # x = F.relu(self.convs[i](x, edge_index))
            # x = F.relu(self.convs[i](x, edge_index, None))

            x = self.convs[i](x)
            x = self.bns[i](x)
            if i == self.num_gc_layers - 1:
            #     # remove relu for the last layer
                x = F.dropout(x, self.drop_ratio, training=self.training)
            else:
                x = F.dropout(F.relu(x), self.drop_ratio, training=self.training)
        return x