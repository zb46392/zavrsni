import torch
import torch.nn as nn

net = nn.Sequential()
net.add_module('fc_0', nn.Linear(in_features=10, out_features=64)
net.add_module('relu_0', nn.ReLU())
net.add_module('fc_1' nn.Linear(in_features=64, out_features=32)
net.add_module('relu_1', nn.ReLU())
net.add_module('fc_2', nn.Linear(in_features=32, out_features=2)
net.add_module('softmax_2', nn.Softmax(dim=1))

X = torch.rand(10)
y = net(X)

