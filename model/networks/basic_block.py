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