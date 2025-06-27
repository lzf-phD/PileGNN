import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, GATConv

from .layers import *

class Pile_GraphNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, aggr, edge_attr_dim,
                 gnn_act=True, gnn_dropout=True, dropout_p=0.1,
                 edge_out_dim = 1,device="cuda",
                 **kwargs):
        super(Pile_GraphNetwork, self).__init__()
        self.gnn_act = gnn_act #True
        self.gnn_dropout = gnn_dropout # True
        self.dropout_p = dropout_p #0.0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.SF_layer1 = GATConv(in_channels=input_dim, out_channels=16, edge_dim=edge_attr_dim)
        self.SF_layer2 = GATConv(in_channels=16, out_channels=32, edge_dim=edge_attr_dim)
        decoder = [nn.Linear(32 + edge_attr_dim , 32),nn.Linear(32, 32),
                   nn.BatchNorm1d(32),nn.ReLU(),nn.Linear(32, 16),
                   nn.Linear(16, 16),nn.BatchNorm1d(16),nn.ReLU(),
                   nn.Linear(16, 8),nn.Linear(8, edge_out_dim)]
        self.edge_decoder = nn.Sequential(*decoder)
        self.Batchnorm_16 = nn.BatchNorm1d(16)
        self.Batchnorm_32 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.1)
        self.edge_aggregation = Edge_aggregation_layer(32,edge_attr_dim,32)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr, P ,D ,K):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        x = self.SF_layer1(x, edge_index, edge_attr)
        x = self.Batchnorm_16(x)
        x = self.dropout(x)
        x = self.SF_layer2(x, edge_index, edge_attr)
        y = self.edge_aggregation(x, edge_index, edge_attr)
        edge_out = self.edge_decoder(torch.cat([y,edge_attr],dim=1))
        return  self.Sigmoid(edge_out)

class Edge_aggregation_layer(nn.Module):
    def __init__(self, node_dim, edge_attr_dim, out_dim):
        super().__init__()
        self.edge_mlp = MLP(2 * node_dim + edge_attr_dim,[out_dim],out_dim)

    def forward(self, x, edge_index, edge_attr):
        src, tgt = edge_index  # [edge_num]
        x_i = x[src]            # source node features
        x_j = x[tgt]            # target node features
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(edge_input)  # output edge features

