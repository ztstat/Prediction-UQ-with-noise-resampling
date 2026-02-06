"""
This script wires together:
1. data generation
2. ensemble training
3. truncation-based predictive sampling
4. prediction-stage uncertainty quantification
5. visualization
"""

import torch

import numpy as np
from src.data import generate_two_moons
from src.train import train_one_model
from src.utils import open_logger
from src.predict import build_predictive_points
from src.predict import compute_predictive_uncertainty
from src.viz import plot_heatmaps_and_kde, plot_uncertainty_curve
from src.config import default_config, large_scale_config


torch.set_num_threads(8)
torch.set_num_interop_threads(8)

def main():
    """
    Run the full two-moons experiment using the configured pipeline.
    """

    np.random.seed(0)
    torch.manual_seed(0)

    cfg = default_config()

    logger = open_logger(cfg.results_dir)

    logger.log("========== Parameter Settings ==========")
    for k, v in cfg.to_dict().items():
        logger.log(f"{k}: {v}")
    logger.log("=========================================")

    X_real = generate_two_moons(
        n_samples=cfg.sample_size_m,
        noise=cfg.noise_level
    )
    logger.log(f"Fixed real data shape: {X_real.shape}")

    experiment_models, experiment_losses = [], []
    for exp in range(1, cfg.num_experiments + 1):
        model, min_l = train_one_model(
            x_real=X_real,
            sample_size_n=cfg.sample_size_n,
            num_flows=cfg.num_flows,
            lr=cfg.lr,
            sigma_pre=cfg.sigma_pre,
            min_iters=cfg.min_iters,
            max_iters=cfg.max_iters,
            tolerance=cfg.tolerance,
            patience=cfg.patience,
            logging_interval=cfg.logging_interval,
            logger=logger,
            exp_id=exp,
        )
        experiment_models.append(model)
        experiment_losses.append(min_l)
        logger.log(f"[Two-sample exp {exp}] min MMD = {min_l:.6f}")

    sorted_idxs, pts_data = build_predictive_points(
        experiment_models=experiment_models,
        experiment_losses=experiment_losses,
        percentiles=cfg.percentiles,
        num_experiments=cfg.num_experiments,
        per_model_test_size=cfg.per_model_test_size,
        logger=logger,
    )

    plot_heatmaps_and_kde(
        pts_data=pts_data,
        percentiles=cfg.percentiles,
        resolutions=cfg.resolutions,
        heatmap_range=cfg.heatmap_range,
        results_dir=cfg.results_dir,
    )

    uncertainties, selection_counts = compute_predictive_uncertainty(
        experiment_models=experiment_models,
        sorted_idxs=sorted_idxs,
        percentiles=cfg.percentiles,
        num_experiments=cfg.num_experiments,
        per_model_test_size=cfg.per_model_test_size,
    )

    plot_uncertainty_curve(
        percentiles=cfg.percentiles,
        uncertainties=uncertainties,
        selection_counts=selection_counts,
        results_dir=cfg.results_dir,
        logger=logger,
    )

    logger.file.close()

if __name__ == "__main__":
    main()





