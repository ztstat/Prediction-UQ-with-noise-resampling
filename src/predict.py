# src/predict.py
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
    sorted_idxs = np.argsort(experiment_losses)
    pts_data: Dict[float, np.ndarray] = {}

    for p in percentiles:
        k = max(1, int(np.ceil(p * num_experiments)))
        idxs = sorted_idxs[:k]

        if logger is not None:
            logger.log(
                f"[Truncation Top {int(p*100)}%] Retained models: {k}, "
                f"Per-model test points: {per_model_test_size}, Total: {per_model_test_size * k}"
            )

        pts_list = []
        for i in idxs:
            u_test = torch.randn(per_model_test_size, 2)
            with torch.no_grad():
                out = experiment_models[i](u_test).cpu().numpy()
            pts_list.append(out)

        pts_data[p] = np.vstack(pts_list)

    return sorted_idxs, pts_data
