import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl import DGLHeteroGraph
from dgl.nn.pytorch.glob import SumPooling
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 更像是用来初始化？
class GraphNorm(nn.Module):
    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = SumPooling()

    def forward(self, graph:DGLHeteroGraph, feature):
        nodes = torch.ones(size=[graph.num_nodes(), 1]).to(my_device)
        norm = self.graph_pool(graph, nodes) # 1
        norm = torch.sqrt(norm) # 1
        norm = torch.repeat_interleave(norm, graph.batch_num_nodes()) # N
        norm = norm.unsqueeze(-1) # N*1
        norm = norm.repeat(1, int(feature.size()[1])) # N*feature.shape[1]
        return feature/norm



