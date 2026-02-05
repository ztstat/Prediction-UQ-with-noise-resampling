# src/mmd.py

import torch

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Gaussian RBF kernel matrix between x and y.
    Size:
    x: (n, d)
    y: (m, d)
    returns: (n, m)
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)          # (n, m, d)
    dist_sq = torch.sum(diff ** 2, dim=2)           # (n, m)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def mmd_loss(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    MMD^2 estimate using full kernel means.
    x: (n, d) generated samples
    y: (m, d) real datas
    """
    kxx = gaussian_kernel(x, x, sigma)
    kyy = gaussian_kernel(y, y, sigma)
    kxy = gaussian_kernel(x, y, sigma)
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()