import dgl
from dgl import DGLHeteroGraph
from dgl.nn.pytorch.glob import SumPooling
import torch
import torch.nn as nn
import networkx as nx
import dgl
import torch
import numpy as np
import scipy.sparse as spp
import matplotlib.pyplot as plt

u, v = torch.tensor([4,3,0,1,2,1,0,2]), torch.tensor([1,2,3,4,4,3,4,1])
graph = dgl.graph((u, v))
# nx.draw(g3.to_networkx(), with_labels=True)
# plt.show()
nodes = torch.ones(size=[graph.num_nodes(), 1])
sp = SumPooling()
norm = sp(graph, nodes)
print(norm)
# norm_1 = torch.sqrt(norm)
# norm_2 = torch.repeat_interleave(norm_1, graph.batch_num_nodes())
# norm_3 = norm_2.unsqueeze(-1)
# norm_4 = norm_3.repeat(1, 4)
# # print(nodes)
# feature = torch.randn((6, 4))
# res = norm_4/feature
# print(res.shape)