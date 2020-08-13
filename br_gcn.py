
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_scatter.utils.gen as gen
import init
import collections
import heapq
import math
import numpy as np


class BRGCN(MessagePassing):

    def __init__(self, in_channels, out_channels, num_relations, num_nodes, num_bases=5,
                 negative_slope=0.2, dropout=0, root_weight=True, bias=False, **kwargs):
        super(BRGCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_nodes = num_nodes
        self.num_bases = num_bases
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.num_original_features = 16

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.basis_0 = Param(torch.Tensor(num_relations, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_relations, num_bases))
        self.gat_att = Param(torch.Tensor(self.num_relations, self.in_channels*self.out_channels))

        self.relation_weight_1 = Param(torch.Tensor(num_relations, self.num_nodes, self.num_nodes))
        self.relation_weight_2 = Param(torch.Tensor(num_relations, self.num_nodes, self.num_nodes))
        self.relation_weight_3 = Param(torch.Tensor(num_relations, self.num_nodes, self.num_nodes))

        if root_weight:
            self.root = Param(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Param(torch.Tensor(self.in_channels, self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        gen.maybe_dim_size = self.maybe_dim_size_rGATConv


    def maybe_dim_size_rGATConv(self, index, dim_size=None):
        if dim_size is not None:
            return self.num_nodes
        return self.num_nodes if index.numel() > 0 else 0


    def reset_parameters(self):
        if self.num_bases == 0:
            size = self.in_channels
        else:
            size = self.num_bases * self.in_channels

        init.uniform(size, self.att)
        init.glorot(self.basis)
        init.glorot(self.basis_0)
        init.glorot(self.gat_att)
        init.uniform(size, self.root)
        init.zeros(self.bias)
        init.ones(self.relation_weight_1)
        init.ones(self.relation_weight_2)
        init.ones(self.relation_weight_3)


    def getRelationFeatures(self, x, edge_index, index, edge_type, edge_norm=None, size=None):
        if len(index) > 0:
            self.idx = index
            edge_index_subset = edge_index[:, index]
            edge_type_subset = edge_type[index].long()

            aggr_for_edge_type = self.propagate(edge_index_subset, size=size, x=x, edge_type=edge_type_subset,
                                                  edge_norm=edge_norm)
            return aggr_for_edge_type
        else:
            return None

    def topKElems(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        count = collections.Counter(nums)
        return heapq.nlargest(k, count.keys(), key=count.get)

    def our_sparse_multiply(self, x, edge_index):
        edge_index = edge_index.t()
        y = torch.zeros(x.size())
        y[edge_index[0], edge_index[1]] = x[edge_index[0], edge_index[1]]
        return y

    ###########################
    # forward helper functions#
    ###########################

    def fullBatchedRelAttention(self, q_r_a, k_r_b, v_r_b, applySoftmax, edge_index):
        if (q_r_a is None):
            if (k_r_b is None):
                val = v_r_b
            else:
                val = torch.matmul(k_r_b, k_r_b.t())
                if applySoftmax:
                    val = F.log_softmax(val, dim=1)
                val = self.our_sparse_multiply(val, edge_index)
                val = torch.matmul(val, v_r_b)

        else:
            if (k_r_b is None):
                val = torch.matmul(q_r_a, q_r_a.t())
                if applySoftmax:
                    val = F.log_softmax(val, dim=1)
                val = self.our_sparse_multiply(val, edge_index)
                val = torch.matmul(val, v_r_b)

            else:
                val = torch.matmul(q_r_a, k_r_b.t())
                if applySoftmax:
                    val = F.log_softmax(val, dim=1)
                val = self.our_sparse_multiply(val, edge_index)
                val = torch.matmul(val, v_r_b)
        return val

    def partialBatchedRelAttention(self, q_r_a, k_r_b, v_r_b, applySoftmax, edge_index):
        if (q_r_a is None):
            if (k_r_b is None):
                val = v_r_b
            else:
                val = torch.matmul(v_r_b, k_r_b.t())
                if applySoftmax:
                    val = F.log_softmax(val, dim=1)

                val = self.our_sparse_multiply(val, edge_index)
                val = torch.matmul(val, k_r_b)

        else:
            if (k_r_b is None):
                val = torch.matmul(v_r_b, q_r_a.t())
                if applySoftmax:
                    val = F.log_softmax(val, dim=1)
                val = self.our_sparse_multiply(val, edge_index)
                val = torch.matmul(val, q_r_a)

            else:
                val = torch.matmul(v_r_b, q_r_a.t())
                if applySoftmax:
                    val = F.log_softmax(val, dim=1)
                val = self.our_sparse_multiply(val, edge_index)
                val = torch.matmul(val, k_r_b)
        return val

    def no_outer_relations(self, x, freq_relations_inner, edge_index, edge_type, num_rows, applySoftmax, final_node_embeddings):
        for r_b in freq_relations_inner:
            idx = torch.where(edge_type == r_b)[0]
            aggr_for_edge_type_b = self.getRelationFeatures(x, edge_index, idx, edge_type)

            if (aggr_for_edge_type_b is None):
                continue

            aggr_for_edge_type_b_avg = aggr_for_edge_type_b[0:num_rows, :]

            k_r_b = torch.matmul(self.relation_weight_2[r_b, :, :], aggr_for_edge_type_b_avg)
            v_r_b = torch.matmul(self.relation_weight_3[r_b, :, :], aggr_for_edge_type_b)

            aggr_for_edge_type_b = None
            aggr_for_edge_type_b_avg = None

            try:
                if (k_r_b is None):
                    val = v_r_b
                else:
                    val = torch.matmul(k_r_b, k_r_b.t())
                    if applySoftmax:
                        val = F.log_softmax(val, dim=1)
                    val = self.our_sparse_multiply(val, edge_index)
                    val = torch.matmul(val, v_r_b)

                final_node_embeddings += val

            except Exception as e:
                print('k_r_b.t(): ', k_r_b.t().size())
                print('v_r_b.t(): ', v_r_b.t().size())
                print('final_node_embeddings: ', final_node_embeddings.size())
                print(e)
                print("skipping over r_b: ", r_b)
        return final_node_embeddings

    #########################################
    #########################################


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):

        final_node_embeddings = torch.zeros(self.num_nodes, self.out_channels)

        ####################
        # user configurable#
        ####################
        applySoftmax = True  # attention normalization
        num_relations_outer = self.num_relations # max = self.num_relations
        num_relations_inner = self.num_relations # max = self.num_relations
        num_rows = self.num_nodes  # max = self.num_nodes
        ####################

        freq_relations_outer = self.topKElems(edge_type, num_relations_outer)
        freq_relations_inner = self.topKElems(edge_type, num_relations_inner)

        e0 = edge_index[0][edge_index[0] < num_rows]
        e1 = edge_index[1][edge_index[1] < num_rows]
        new_edge_index = torch.stack([e0, e1], dim=0)
        edge_index = new_edge_index

        if (len(freq_relations_outer) > 0):
            for r_a in freq_relations_outer:
                idx = (edge_type == r_a).nonzero()
                idx = idx[idx < num_rows]
                idx = idx.view(-1)

                aggr_for_edge_type_a = self.getRelationFeatures(x, edge_index, idx, edge_type)

                if aggr_for_edge_type_a is None:
                    continue

                aggr_for_edge_type_a = aggr_for_edge_type_a[0:num_rows, :]

                for r_b in freq_relations_inner:
                    idx = (edge_type == r_b).nonzero()
                    idx = idx[idx < num_rows]
                    idx = idx.view(-1)
                    aggr_for_edge_type_b = self.getRelationFeatures(x, edge_index, idx, edge_type)

                    if (aggr_for_edge_type_b is None):
                        continue

                    aggr_for_edge_type_b_avg = aggr_for_edge_type_b[0:num_rows, :]

                    q_r_a = torch.matmul(self.relation_weight_1[r_a, :, :], aggr_for_edge_type_a)
                    k_r_b = torch.matmul(self.relation_weight_2[r_b, :, :], aggr_for_edge_type_b_avg)
                    v_r_b = torch.matmul(self.relation_weight_3[r_b, :, :], aggr_for_edge_type_b)

                    aggr_for_edge_type_b = None
                    aggr_for_edge_type_b_avg = None

                    if (num_rows == self.num_nodes):
                        val = self.fullBatchedRelAttention(q_r_a, k_r_b, v_r_b, applySoftmax, edge_index)

                    # if using batched version
                    if (num_rows < self.num_nodes):
                        val = self.partialBatchedRelAttention(q_r_a, k_r_b, v_r_b, applySoftmax, edge_index)

                    final_node_embeddings += val

                aggr_for_edge_type_a = None
        else:
            final_node_embeddings = self.no_outer_relations(x, freq_relations_inner, edge_index, edge_type, num_rows, applySoftmax, final_node_embeddings)
        return final_node_embeddings




    def message(self, x_i, x_j, edge_index_i, edge_index_j, edge_type, edge_norm, size_i):

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
