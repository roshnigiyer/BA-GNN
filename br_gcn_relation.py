
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import torch_scatter.utils.gen as gen
import init
import math


class BRGCNRelation(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_bases, num_nodes, num_edges,
                 root_weight=True, bias=False, **kwargs):
        super(BRGCNRelation, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_nodes = num_nodes

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.basis_0 = Param(torch.Tensor(num_relations, in_channels, out_channels))
        self.relation_weight = Param(torch.Tensor(num_relations, 1))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


        def maybe_dim_size_custom(index, dim_size=None):
            if dim_size is not None:
                return dim_size
            return self.num_nodes if index.numel() > 0 else 0

        gen.maybe_dim_size = maybe_dim_size_custom


    def reset_parameters(self):
        if self.num_bases == 0:
            size = self.in_channels
        else:
            size = self.num_bases * self.in_channels

        init.glorot(self.basis)
        init.glorot(self.att)
        init.glorot(self.root)
        init.glorot(self.bias)
        init.glorot(self.basis_0)
        init.glorot(self.relation_weight)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):

        final_node_embeddings = torch.zeros(self.num_nodes, self.out_channels)
        k = math.floor(self.num_relations * 0.40)
        for relation in range(0, k):
            idx = (edge_type == relation).nonzero()
            idx = idx.view(-1)
            self.idx = idx
            edge_index_subset = edge_index[:, idx]
            edge_type_subset = edge_type[idx].long()

            aggr_for_edge_type = self.propagate(edge_index_subset, size=size, x=x, edge_type=edge_type_subset,
                                                edge_norm=edge_norm)

            rel_out = aggr_for_edge_type * self.relation_weight[relation]

            final_node_embeddings += rel_out

        return final_node_embeddings


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        if self.num_bases > 0:
            w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        else:
            w = self.basis_0

        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out_pre = x_j.unsqueeze(1)
            out_before = torch.bmm(x_j.unsqueeze(1), w)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        else:
            out = aggr_out
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)