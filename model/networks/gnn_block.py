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
# 有待验证！
class GraphNorm(nn.Module):
    def __init__(self):
        super(GraphNorm, self).__init__()
        self.graph_pool = SumPooling()

    def forward(self, graph:DGLHeteroGraph, feature): # feature:N*32
        nodes = torch.ones(size=[graph.num_nodes(), 1]).to(my_device)
        norm = self.graph_pool(graph, nodes) # 1
        norm = torch.sqrt(norm) # 1
        norm = torch.repeat_interleave(norm, graph.batch_num_nodes()) # N
        norm = norm.unsqueeze(-1) # N*1
        norm = norm.repeat(1, int(feature.size()[1])) # N*32
        return feature/norm

# N个节点，M个feature
# 返回M*1
class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.graph_pool = SumPooling()

    def forward(self, graph, node_feat):
        sum_pooled = self.graph_pool(graph, node_feat)
        ones_sum_pooled = self.graph_pool(
            graph,
            torch.ones_like(node_feat))
        pooled = sum_pooled / ones_sum_pooled
        return pooled

# 通过节点和边的特征得到节点的特征
class GIN(nn.Module):
    """
    Implementation of Graph Isomorphism Network (GIN) layer with edge features
    """
    def __init__(self, hidden_size):
        super(GIN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, graph : DGLHeteroGraph, node_feat, edge_feat):
        graph.ndata['h'] = node_feat
        graph.edata['e'] = edge_feat
        graph.update_all(fn.u_add_e('h', 'e', 'm'), fn.sum('m', 'h'))
        graph = graph.to(my_device)
        node_feat = graph.ndata['h']
        node_feat = self.mlp(node_feat)
        return node_feat

# 图注意力网络
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    # 初始化网络参数
    def reset_parameters(self):
        # gain用来调整权重的标准差
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    # 拼接目标节点和源节点的特征向量，计算原始的图注意力
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["z"], edges.dst["z"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    # 返回源节点和边的特征向量字典
    # 生成沿着边传递的信息
    def message_func(self, edges):
        return {"z": edges.src["z"], "e": edges.data["e"]}

    # 得到聚合后的节点特征向量
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["z"], dim=1)
        return {"h": h}

    # 得到更新后的结点特征
    def forward(self,g, h):
        z = self.fc(h)
        g.ndata["z"] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("h")
