import dgl
from dgl import DGLHeteroGraph
from dgl.nn.pytorch.glob import SumPooling
import torch
import torch.nn as nn
import networkx as nx
import dgl.function as fn
import dgl
import torch
import numpy as np
import scipy.sparse as spp
import matplotlib.pyplot as plt

# sp = SumPooling()
# grapg = dgl.rand_graph(3, 4)
# node_feat = torch.randn((3,2))
# edge_feat = torch.randn((4,2))
# # print(node_feat)
# # print(edge_feat)
# grapg.ndata['h'] = node_feat
# grapg.edata['e'] = edge_feat
# grapg.update_all(fn.u_add_e('h', 'e', 'm'),fn.sum('m', 'h'))
# print(grapg.ndata)
# print(grapg.edata)
# print(grapg.update_all)

attn = torch.randn((3,4,5))
mask = torch.tensor([[[0],[1],[0],[1]],
                     [[0],[1],[0],[1]],
                     [[0],[1],[0],[1]]])
# mask.unsqueeze(1)
# print(mask.shape)
print(mask.shape)
attn.masked_fill_(mask.to(torch.bool), float('-inf'))
print(attn)

