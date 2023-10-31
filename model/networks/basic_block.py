"""
一些经常被使用的模块
包括如下：
（1）激活函数：可以选择relu或leakyrelu
（2）MLP层：输入总层数，输入、输出维度等
（3）RBF层：径向基函数层
"""


import torch
import torch.nn as nn

class Activation(nn.Module):
    def __init__(self, act_type, **params):
        super(Activation,self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(**params)
        else:
            raise ValueError(act_type)

    def forward(self, x):
        return self.act(x)

class MLP(nn.Module):
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout):
        super(MLP, self).__init__()

        layers = nn.ModuleList()
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(Activation(act))
            elif layer_id < layer_num-1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(Activation(act))
            else:
                layers.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class RBF(nn.Module):
    def __init__(self, centers, gamma, dtype=torch.float32):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers, dtype=dtype), [1, -1]).cuda()
        self.gamma = torch.tensor(gamma).cuda()

    def forward(self, x):
        x = torch.reshape(x, [1, -1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))
