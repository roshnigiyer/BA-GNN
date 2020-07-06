import os
import os.path as osp
import torch
import tensorflow as tf
import tensorboard as tb
import torch.nn.functional as F
from torch_geometric.datasets import Entities
import torch_geometric.transforms as T
from br_gcn_node import BRGCNNode
import statistics
import networkx as nx
import numpy as np
import shutil

dataType = 'Entities'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', dataType)

if dataType == 'Entities':
    mydataName = 'AIFB' #"MUTAG", "AIFB", "BGS", "AM"
    dataset = Entities(path, mydataName)
    data = dataset[0]

class Net(torch.nn.Module):
    def __init__(self, interim_channel=16, num_bases=0, dropout=0.4, negative_slope=0.4, num_nodes=data.num_nodes):
        super(Net, self).__init__()
        self.interim_channel = interim_channel
        self.num_bases = num_bases
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_nodes = num_nodes
        self.conv1 = BRGCNNode(
            data.num_nodes, self.interim_channel, dataset.num_relations, self.num_nodes, self.num_bases,
            self.dropout, self.negative_slope)
        self.conv2 = BRGCNNode(
            self.interim_channel, dataset.num_classes, dataset.num_relations, self.num_nodes, self.num_bases,
            self.dropout, self.negative_slope)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, edge_index, edge_type, edge_norm):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)

loss_op = F.nll_loss
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=0),
    dict(params=model.non_reg_params, weight_decay=0)
], lr=0.05)
l = model(data.edge_index, data.edge_type, data.edge_norm)


def split_data(train_percent):
    orig_train = l[data.train_idx]
    train_split = train_percent
    train_size = int(train_split * len(orig_train))
    val_size = len(orig_train) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(orig_train, [train_size, val_size])
    train_indices = torch.LongTensor(train_dataset.indices)
    val_indices = torch.LongTensor(val_dataset.indices)
    return train_indices, val_indices


def train(train_indices):
    model.train()
    optimizer.zero_grad()
    l = model(data.edge_index, data.edge_type, data.edge_norm)
    pred = l[data.train_idx][train_indices].max(1)[1]
    acc = pred.eq(data.train_y[train_indices]).sum().item() / data.train_y[train_indices].size(0)
    loss = loss_op(l[data.train_idx][train_indices], data.train_y[train_indices])
    loss.backward(retain_graph=True)
    optimizer.step()
    return loss, acc


def validation(val_indices):
    model.eval()
    logits = model(data.edge_index, data.edge_type, data.edge_norm)
    pred = logits[data.train_idx][val_indices].max(1)[1]
    acc = pred.eq(data.train_y[val_indices]).sum().item() / data.train_y[val_indices].size(0)
    return acc


def test():
    model.eval()
    logits = model(data.edge_index, data.edge_type, data.edge_norm)
    pred = logits[data.test_idx].max(1)[1]
    acc = pred.eq(data.test_y).sum().item() / data.test_y.size(0)
    return acc


num_epochs = 90
num_runs = 10
num_trials = 10
val_acc_list = []
test_acc_list = []
for trial in range(0, num_trials):
    for runs in range(0, num_runs):

        model, data = Net().to(device), data.to(device)
        loss_op = F.nll_loss
        optimizer = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0),
            dict(params=model.non_reg_params, weight_decay=0)
        ], lr=0.05)

        train_indices, val_indices = split_data(0.8)

        for epoch in range(0, num_epochs):
            train_loss, train_acc = train(train_indices)
            log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.4f}'
            print(log.format(epoch, train_loss, train_acc))
            val_acc = validation(val_indices)
            print('Val Acc: {:.4f}'.format(val_acc))

        val_acc = validation(val_indices)
        print('Val Acc: {:.4f}'.format(val_acc))
        val_acc_list.append(val_acc)
        test_acc = test()
        print('Test Acc: {:.4f}'.format(test_acc))
        test_acc_list.append(test_acc)

        print("##### Iteration ", str(runs), 'completed #####')

        shutil.rmtree(path)
        dataType = 'Entities'
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', dataType)

        if dataType == 'Entities':
            dataName = mydataName  # "MUTAG", "AIFB", "BGS", "AM"
            dataset = Entities(path, dataName)
            data = dataset[0]

    # val
    avg_val_acc = statistics.mean(val_acc_list)
    std_val_acc = statistics.stdev(val_acc_list)
    print("Avg Val Acc: ", str(avg_val_acc), " +/- ", str(std_val_acc))

    # test
    avg_test_acc = statistics.mean(test_acc_list)
    std_test_acc = statistics.stdev(test_acc_list)
    print("Avg Test Acc: ", str(avg_test_acc), " +/- ", str(std_test_acc))



