# src/viz.py

"""
Visualization utilities for prediction-focused uncertainty analysis.

This module visualizes the predictive distribution induced by truncated
ensembles (via heatmaps / KDE), and summarizes prediction uncertainty as a
function of truncation level.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde


def plot_heatmaps_and_kde(
    pts_data: Dict[float, np.ndarray],
    percentiles: List[float],
    resolutions: List[int],
    heatmap_range: List[List[float]],
    results_dir: str,
) -> None:
    """
    For each percentile p (top-k% retained models), this function visualizes the
    aggregated predictive samples in multiple complementary ways:
    - 2D histogram density heatmap (linear scale)
    - 2D histogram density heatmap (log scale) to highlight tails
    - KDE heatmap (log scale) for smoother density visualization
    - Incremental "diff" heatmaps showing which regions are newly covered when
      expanding from smaller truncation to larger truncation
    """
    x0, x1 = heatmap_range[0]
    y0, y1 = heatmap_range[1]

    for bins in resolutions:
        subdir = os.path.join(results_dir, f"{bins}bins")
        os.makedirs(subdir, exist_ok=True)

        global_max = 0.0
        density_maps = {}
        raw_maps = {}

        for p in percentiles:
            pts = pts_data[p]
            H_den, xe, ye = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=bins,
                range=heatmap_range,
                density=True
            )
            C_cnt, _, _ = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=bins,
                range=heatmap_range,
                density=False
            )
            density_maps[p] = (H_den, xe, ye)
            raw_maps[p] = C_cnt
            global_max = max(global_max, float(H_den.max()))

        # keep original logic (note: if global_max==0, LogNorm would fail; leaving as-is per your request)
        vmin = global_max * 1e-3

        for p in percentiles:
            pct = int(p * 100)
            H, xe, ye = density_maps[p]

            plt.figure(figsize=(6, 6))
            plt.imshow(
                H.T, origin="lower",
                extent=[x0, x1, y0, y1],
                aspect="auto",
                vmin=0, vmax=global_max,
                cmap="inferno"
            )
            plt.title(f"Heatmap Top {pct}% — {bins}×{bins}")
            plt.colorbar(label="Density")
            plt.savefig(os.path.join(subdir, f"heatmap_top{pct}pct_{bins}bins.png"))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                H.T, origin="lower",
                extent=[x0, x1, y0, y1],
                aspect="auto",
                norm=LogNorm(vmin=vmin, vmax=global_max),
                cmap="inferno"
            )
            plt.title(f"Enhanced Heatmap Top {pct}% — {bins}×{bins}")
            plt.colorbar(label="Density (log scale)")
            plt.savefig(os.path.join(subdir, f"heatmap_top{pct}pct_{bins}bins_enhanced.png"))
            plt.close()

        xx, yy = np.meshgrid(
            np.linspace(x0, x1, bins),
            np.linspace(y0, y1, bins)
        )
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])

        for p in percentiles:
            pct = int(p * 100)
            pts = pts_data[p].T
            kde = gaussian_kde(pts, bw_method=0.2)
            zz = kde(grid_coords).reshape((bins, bins))

            plt.figure(figsize=(6, 6))
            plt.imshow(
                zz.T, origin="lower",
                extent=[x0, x1, y0, y1],
                aspect="auto",
                norm=LogNorm(vmin=1e-3, vmax=float(zz.max())),
                cmap="inferno"
            )
            plt.title(f"KDE LogNorm Heatmap Top {pct}% — {bins}×{bins}")
            plt.colorbar(label="Density (log KDE)")
            plt.savefig(os.path.join(subdir, f"heatmap_top{pct}pct_{bins}bins_kde.png"))
            plt.close()

        prev = np.zeros((bins, bins))
        for p in percentiles:
            pct = int(p * 100)
            curr = raw_maps[p]
            diff = curr - prev
            prev = curr.copy()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                diff.T, origin="lower",
                extent=[x0, x1, y0, y1],
                aspect="auto",
                cmap="inferno"
            )
            plt.title(f"Diff Heatmap Contribution Top {pct}% — {bins}×{bins}")
            plt.colorbar(label="New sample counts")
            plt.savefig(os.path.join(subdir, f"heatmap_diff_top{pct}pct_{bins}bins.png"))
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(
                diff.T, origin="lower",
                extent=[x0, x1, y0, y1],
                aspect="auto",
                norm=LogNorm(vmin=1, vmax=(diff.max() if diff.max() > 1 else 1)),
                cmap="inferno"
            )
            plt.title(f"Enhanced Diff Heatmap Contribution Top {pct}% — {bins}×{bins}")
            plt.colorbar(label="New sample counts (log scale)")
            plt.savefig(os.path.join(subdir, f"heatmap_diff_top{pct}pct_{bins}bins_enhanced.png"))
            plt.close()


def plot_uncertainty_curve(
    percentiles: List[float],
    uncertainties: List[float],
    selection_counts: List[int],
    results_dir: str,
    logger=None,
) -> None:
    """
    Plot percentile vs uncertainty and selection count.

    - x-axis: truncation level (top-k% retained models)
    - left y-axis: predictive uncertainty metric computed in src/predict.py
    - right y-axis: number of retained models

    """
    os.makedirs(results_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot([p * 100 for p in percentiles], uncertainties, marker="o", label="Avg Uncertainty")
    ax1.set_xlabel("Top k% of Models")
    ax1.set_ylabel("Average Uncertainty")

    ax2 = ax1.twinx()
    ax2.plot([p * 100 for p in percentiles], selection_counts, marker="s", color="orange", label="Selected Models")
    ax2.set_ylabel("Number of Selected Models")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.title("Model Percentile vs Uncertainty & Selection Count")
    out_path = os.path.join(results_dir, "percentile_vs_uncertainty_and_selection.png")
    plt.savefig(out_path)
    plt.close()

    if logger is not None:
        logger.log(f"Saved {out_path}")
