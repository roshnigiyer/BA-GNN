
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import math
import init


class BRGCNNode(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_nodes, num_bases=5,
                 negative_slope=0.2, dropout=0, root_weight=True, bias=False, **kwargs):
        super(BRGCNNode, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.num_bases = num_bases
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.basis_0 = Param(torch.Tensor(num_relations, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.gat_att = Param(torch.Tensor(self.num_relations, self.in_channels*self.out_channels))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels

        init.glorot(self.att)
        init.glorot(self.basis)
        init.glorot(self.basis_0)
        init.glorot(self.gat_att)
        init.glorot(self.root)
        init.zeros(self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):

        firstTime = True
        k = math.floor(self.num_relations*0.80)
        for relation in range(0, k):
            idx = (edge_type == relation).nonzero()
            idx = idx.view(-1)
            if firstTime:
                new_idx = idx
                firstTime = False
            else:
                new_idx = torch.cat((new_idx, idx), 0)

        edge_index_subset = edge_index[:, new_idx]
        edge_type_subset = edge_type[new_idx].long()

        return self.propagate(edge_index_subset, size=size, x=x, edge_type=edge_type_subset,
                              edge_norm=edge_norm)


    def message(self, x_i, x_j, edge_index_i, edge_index_j, edge_type, edge_norm, size_i):

        if self.num_bases > 0:
            w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        else:
            w = self.basis_0

        if x_j is None:
            w_xj = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out_j = torch.index_select(w_xj, 0, index)
            gatAtt = self.gat_att.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out_gat_j = torch.index_select(gatAtt, 0, index)

        else:
            w_xj = w.view(self.num_relations, self.in_channels, self.out_channels)
            w_xj = torch.index_select(w_xj, 0, edge_type)
            out_j = torch.bmm(x_j.unsqueeze(1), w_xj).squeeze(-2)
            gatAtt = self.gat_att.view(self.num_relations, self.in_channels, self.out_channels)
            gatAtt = torch.index_select(gatAtt, 0, edge_type)
            out_gat_j = torch.bmm(x_j.unsqueeze(1), gatAtt).squeeze(-2)

        if x_i is None:
            gatAtt = self.gat_att.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_i
            out_gat_i = torch.index_select(gatAtt, 0, index)

        else:
            gatAtt = self.gat_att.view(self.num_relations, self.in_channels, self.out_channels)
            gatAtt = torch.index_select(gatAtt, 0, edge_type)
            out_gat_i = torch.bmm(x_i.unsqueeze(1), gatAtt).squeeze(-2)

        if self.num_bases > 0:
            out_gat_j = out_gat_j.view(-1, self.num_bases, self.out_channels)
            out_gat_i = out_gat_i.view(-1, self.num_bases, self.out_channels)
        else:
            out_gat_j = out_gat_j.view(-1, 1, self.out_channels)
            out_gat_i = out_gat_i.view(-1, 1, self.out_channels)

        alpha = (torch.cat([out_gat_i, out_gat_j], dim=-1)).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.view(-1)
        alpha = softmax(alpha, edge_index_i, size_i)
        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out_j = out_j.view(-1, self.out_channels)
        return out_j if alpha is None else out_j * alpha.view(-1, 1)


    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out


    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)