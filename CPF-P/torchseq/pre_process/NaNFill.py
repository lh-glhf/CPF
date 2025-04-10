import torch
import torch.nn as nn


class NaNFill(nn.Module):
    def __init__(self, args, fill_type='mean', dim=1):
        super(NaNFill, self).__init__()
        self.fill_type = fill_type
        self.dim = dim

    def forward(self, x):
        nan_mask = torch.isnan(x)

        if self.fill_type == 'mean':
            fill_value = torch.nanmean(x, dim=self.dim, keepdim=True)
        elif self.fill_type == 'zero':
            fill_value = torch.zeros_like(x)

        x = torch.where(nan_mask, fill_value, x)
        return x
