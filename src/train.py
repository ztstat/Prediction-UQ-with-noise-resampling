"""
Training utilities for building ensembles of generative models
used in prediction-focused uncertainty quantification.
"""

import torch

from typing import Tuple
from torch.optim import Adam
from src.flows import PlanarFlowModel
from src.mmd import mmd_loss
from src.utils import Logger

def train_one_model(
    x_real: torch.Tensor,
    sample_size_n: int,
    num_flows: int,
    lr: float,
    sigma_pre: float,
    min_iters: int,
    max_iters: int,
    tolerance: float,
    patience: int,
    logging_interval: int,
    logger: Logger,
    exp_id: int,
) -> Tuple[PlanarFlowModel, float]:
    """
    Train a single generative model for the ensemble.

    Each call to this function produces one candidate predictive model.
    The goal is not aggressive optimization, but to obtain a diverse set
    of reasonably well-fitted models for downstream prediction uncertainty
    analysis.
    """

    # Initialize a flow-based generator.
    # Input dimension is fixed (2D toy example), depth controlled by num_flows.
    model = PlanarFlowModel(2, num_flows)
    opt = Adam(model.parameters(), lr=lr)

    # prev : loss value at previous iteration
    # pat  : counter for consecutive small-improvement steps
    # min_l: best (minimum) loss observed during training
    prev, pat, min_l = None, 0, float("inf")
    u_small = torch.randn(sample_size_n, 2)

    # prev : loss value at previous iteration
    # pat  : counter for consecutive small-improvement steps
    # min_l: best (minimum) loss observed during training
    for it in range(max_iters + 1):
        opt.zero_grad()

        # Forward pass: map latent samples to data space
        out = model(u_small)

        # Discrepancy between generated samples and real data (MMD)
        loss = mmd_loss(out, x_real, sigma_pre)
        lval = float(loss.item())
        min_l = min(min_l, lval)

        # Periodic logging for monitoring convergence behavior
        if it % logging_interval == 0:
            logger.log(f"[Two-sample exp {exp_id}] Iter {it}, Loss {lval:.6f}")

        loss.backward()
        opt.step()

        # Early stopping based on local loss stabilization
        # We start checking only after min_iters to avoid premature stopping.
        # If the absolute change between consecutive losses falls below
        # tolerance for 'patience' consecutive iterations, we stop training.
        if it >= min_iters and prev is not None:
            if abs(lval - prev) < tolerance:
                pat += 1
                if pat >= patience:
                    logger.log(f"[Two-sample exp {exp_id}] early stop at iter {it}")
                    break
            else:
                pat = 0
        prev = lval

    # Report the best discrepancy achieved by this model.
    # This value is later used for ranking models in the ensemble.
    logger.log(f"[Two-sample exp {exp_id}] min MMD = {min_l:.6f}")
    return model, min_l
