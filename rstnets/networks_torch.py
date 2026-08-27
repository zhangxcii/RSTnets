import torch.nn as nn


def FCNN(layer_sizes):
    layers = []
    for in_width, out_width in zip(layer_sizes[:-2], layer_sizes[1:-1]):
        linear = nn.Linear(in_width, out_width)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.Tanh())
    out_linear = nn.Linear(layer_sizes[-2], layer_sizes[-1])
    nn.init.xavier_normal_(out_linear.weight)
    nn.init.zeros_(out_linear.bias)
    layers.append(out_linear)
    return nn.Sequential(*layers)
