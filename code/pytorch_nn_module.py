import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self._fc1 = nn.Linear(in_features=input_size, out_features=64)
        self._fc2 = nn.Linear(in_features=64, out_features=32)
        self._out = nn.Linear(in_features=32, out_features=output_size)
		
    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        t = in_tensor
        t = F.relu(self._fc1(t))
        t = F.relu(self._fc2(t))
        t = self._out(t)
        return t


net = MyNet(3, 2)

X = torch.rand(3)
y = net(X)

