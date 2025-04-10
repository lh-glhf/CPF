import torch
import torch.nn as nn


class MN(nn.Module):
    def __init__(self, args, dim=1, eps=1e-8):
        super(MN, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        x_min, _ = torch.min(x, dim=self.dim, keepdim=True)
        x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        x_range = (x_max - x_min).clamp(min=self.eps)
        x = (x - x_min) / x_range

        return x
