import warnings


import torch




from src.data import generate_two_moons
from src.train import train_one_model
from src.utils import open_logger
from src.predict import build_predictive_points
from src.predict import compute_predictive_uncertainty
from src.viz import plot_heatmaps_and_kde, plot_uncertainty_curve


torch.set_num_threads(8)
torch.set_num_interop_threads(8)

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os




# --- Parameters ---
tolerance = 0.01
patience = 5
min_iters = 100
max_iters = 5000
logging_interval = 200

sample_size_m     = 100
sample_size_n     = 100
noise_level       = 0.01

num_experiments   = 5

num_flows = 2
lr        = 0.001
sigma_pre = 0.2

# --- Test Parameters ---
per_model_test_size = 100
heatmap_range = [[-2,2],[-2,2]]

percentiles = [0.1]
resolutions = [100]

# create logger
results_dir = "results"
logger = open_logger(results_dir)

logger.log("========== Parameter Settings ==========")
for k,v in [
    ("min_iters"       , min_iters),
    ("max_iters"       , max_iters),
    ("logging_interval", logging_interval),
    ("sample_size_m"   , sample_size_m),
    ("sample_size_n"   , sample_size_n),
    ("noise_level"     , noise_level),
    ("num_experiments" , num_experiments),
    ("num_flows"       , num_flows),
    ("lr"              , lr),
    ("sigma_pre"       , sigma_pre),
    ("per_model_test_size", per_model_test_size),
    ("heatmap_range"   , heatmap_range),
    ("resolutions"     , resolutions),
]:
    logger.log(f"{k}: {v}")
logger.log("=========================================")





X_real = generate_two_moons(n_samples=sample_size_m, noise=noise_level)

logger.log(f"Fixed real data shape: {X_real.shape}")





experiment_models, experiment_losses = [], []
for exp in range(1, num_experiments + 1):
    model, min_l = train_one_model(
        x_real=X_real,
        sample_size_n=sample_size_n,
        num_flows=num_flows,
        lr=lr,
        sigma_pre=sigma_pre,
        min_iters=min_iters,
        max_iters=max_iters,
        tolerance=tolerance,
        patience=patience,
        logging_interval=logging_interval,
        logger=logger,
        exp_id=exp,
    )
    experiment_models.append(model)
    experiment_losses.append(min_l)
    logger.log(f"[Two-sample exp {exp}] min MMD = {min_l:.6f}")

# --- 4. Assignm noise ---
sorted_idxs, pts_data = build_predictive_points(
    experiment_models=experiment_models,
    experiment_losses=experiment_losses,
    percentiles=percentiles,
    num_experiments=num_experiments,
    per_model_test_size=per_model_test_size,
    logger=logger,
)

# --- 5. Draw Heatmaps ---
plot_heatmaps_and_kde(
    pts_data=pts_data,
    percentiles=percentiles,
    resolutions=resolutions,
    heatmap_range=heatmap_range,
    results_dir=results_dir,
)

uncertainties, selection_counts = compute_predictive_uncertainty(
    experiment_models=experiment_models,
    sorted_idxs=sorted_idxs,
    percentiles=percentiles,
    num_experiments=num_experiments,
    per_model_test_size=per_model_test_size,
)

plot_uncertainty_curve(
    percentiles=percentiles,
    uncertainties=uncertainties,
    selection_counts=selection_counts,
    results_dir=results_dir,
    logger=logger,
)




