# src/data.py

"""
Synthetic data generation utilities for controlled experiments.

This module provides simple, low-dimensional datasets used to
study prediction uncertainty in a controlled setting.
"""


import torch
from sklearn.datasets import make_moons

def generate_two_moons(n_samples: int, noise: float) -> torch.Tensor:
    """
        Generate a two-moons dataset as reference data.

        The two-moons distribution is used as a toy example where:
        - the data manifold is nonlinear but low-dimensional,
        - the ground-truth structure is visually interpretable,
        - predictive uncertainty can be analyzed qualitatively.

        Parameters
        ----------
        n_samples : int
            Number of data points to generate.
        noise : float
            Standard deviation of Gaussian noise added to the data.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_samples, 2) containing the dataset.
        """

    x, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.tensor(x, dtype=torch.float32)