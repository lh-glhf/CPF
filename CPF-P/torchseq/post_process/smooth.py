import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverageSmoothing(nn.Module):
    def __init__(self, args, window_size=3):
        super(MovingAverageSmoothing, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        x = x.unsqueeze(1)
        kernel = torch.ones(1, 1, self.window_size) / self.window_size
        x_smoothed = F.conv1d(x, kernel.to(x.device), padding=self.window_size // 2)
        x_smoothed = x_smoothed.squeeze(1)

        return x_smoothed


class GaussianSmoothing(nn.Module):
    def __init__(self, args, window_size=3, sigma=1.0):
        super(GaussianSmoothing, self).__init__()
        self.window_size = window_size
        self.sigma = sigma

    def create_gaussian_kernel(self, window_size, sigma):

        x = torch.arange(window_size) - window_size // 2
        gauss_kernel = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        return gauss_kernel.view(1, 1, -1)  # 形状 (1, 1, window_size)

    def forward(self, x):

        x = x.unsqueeze(1)  # 转换形状为 (batch_size, 1, seq_len, n)
        x_smoothed = F.conv1d(x, self.kernel.to(x.device), padding=self.window_size // 2)
        x_smoothed = x_smoothed.squeeze(1)  # 恢复原始形状
        return x_smoothed


class ExponentialMovingAverage(nn.Module):
    def __init__(self, args, alpha=0.1):
        super(ExponentialMovingAverage, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        ema = x[:, 0, :].clone()
        ema_values = [ema]

        for t in range(1, x.size(1)):
            ema = self.alpha * x[:, t, :] + (1 - self.alpha) * ema
            ema_values.append(ema)

        return torch.stack(ema_values, dim=1)
