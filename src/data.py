# src/data.py
import torch
from sklearn.datasets import make_moons

def generate_two_moons(n_samples: int, noise: float) -> torch.Tensor:
    x, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.tensor(x, dtype=torch.float32)