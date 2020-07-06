# README #

# Models #  
Models utilize the Pytorch Geometric Framework. See requirements.txt for environment details.

- BR-GCN-node: br_gcn_node.py

- BR-GCN-relation: br_gcn_relation.py

- BR-GCN: br_gcn.py

# Baseline Models #
HAN: https://github.com/Jhy1993/HAN

R-GCN: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/rgcn_conv.html#RGCNConv

GAT: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html#GATConv

GIN: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gin_conv.html#GINConv


# Experiments #

Node classification:

- To run BR-GCN-node, run: python datasets_br_gcn_node.py

- To run BR-GCN-relation, run: python datasets_br_gcn_relation.py

- To run BR-GCN, run: python datasets_br_gcn.py

Link Prediction: 

Link prediction experiments are conducted using the Deep Graph Library (DGL) Framework. 

Script for Link Prediction: https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/link_predict.py, such that 
R-GCN models are replaced with BR-GCN models (consult Section: Models above)

# Datasets #

Node classification datasets: https://github.com/tkipf/relational-gcn/tree/master/rgcn/data

Link prediction datasets: https://github.com/MichSchli/RelationPrediction/tree/master/data
