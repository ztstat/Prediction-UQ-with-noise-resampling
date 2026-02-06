# src/predict.py

"""
Prediction-stage uncertainty quantification via truncated ensembles.
"""


from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


def build_predictive_points(
    experiment_models: List,
    experiment_losses: List[float],
    percentiles: List[float],
    num_experiments: int,
    per_model_test_size: int,
    logger=None,
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """
    Truncation-based predictive ensemble:
    sort models by loss (discrepancy)
    keep top-k% models
    sample predictive points from each retained model
    Returns:
      sorted_idxs: np.ndarray of indices sorted by loss ascending
      pts_data: dict {percentile -> (N,2) numpy array of predictive points}
    """

    # Sort models by discrepancy (ascending: best models first)
    sorted_idxs = np.argsort(experiment_losses)

    # Mapping: truncation percentile -> predictive samples
    pts_data: Dict[float, np.ndarray] = {}

    for p in percentiles:
        # Number of retained models under truncation level p
        k = max(1, int(np.ceil(p * num_experiments)))
        idxs = sorted_idxs[:k]

        if logger is not None:
            logger.log(
                f"[Truncation Top {int(p*100)}%] Retained models: {k}, "
                f"Per-model test points: {per_model_test_size}, Total: {per_model_test_size * k}"
            )

        # Collect predictive samples from each retained model
        pts_list = []
        for i in idxs:
            # Fresh latent samples are drawn per model to reflect
            # predictive variability across model choices.
            u_test = torch.randn(per_model_test_size, 2)
            with torch.no_grad():
                out = experiment_models[i](u_test).cpu().numpy()
            pts_list.append(out)

        # Stack all predictive samples into a single array
        # Shape: (k * per_model_test_size, data_dim)
        pts_data[p] = np.vstack(pts_list)

    return sorted_idxs, pts_data

def compute_predictive_uncertainty(
    experiment_models: List,
    sorted_idxs: np.ndarray,
    percentiles: List[float],
    num_experiments: int,
    per_model_test_size: int,
) -> Tuple[List[float], List[int]]:
    """
    Prediction-focused uncertainty:
    quantify variability across retained models in predictive space.
    Current implementation matches the original script:
      for each retained model, draw per_model_test_size samples
      compute model-wise predictive mean
      use dispersion of these means as uncertainty
    """
    uncertainties: List[float] = []
    selection_counts: List[int] = []

    for p in percentiles:

        # Number of retained models
        k = max(1, int(np.ceil(p * num_experiments)))
        idxs = sorted_idxs[:k]
        selection_counts.append(k)

        means = []
        for i in idxs:
            # Predictive mean for each model summarizes its typical output
            u_test = torch.randn(per_model_test_size, 2)
            with torch.no_grad():
                out = experiment_models[i](u_test)
                means.append(out.mean(0))
        # Stack model-wise predictive means
        means = torch.stack(means, dim=0)

        # Deviation of each model's prediction from the ensemble average
        diffs = means - means.mean(0, keepdim=True)
        # Average squared norm (like the variance in scalar case) as one of the possible scalar uncertainty measures
        var = (diffs.norm(dim=1) ** 2).mean().item()
        uncertainties.append(var)

    return uncertainties, selection_counts
