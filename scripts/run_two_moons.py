import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mmd import mmd_loss
from src.flows import PlanarFlowModel




torch.set_num_threads(8)
torch.set_num_interop_threads(8)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from sklearn.datasets import make_moons

# --- Parameters ---
tolerance = 0.01
patience = 5
min_iters = 100
max_iters = 5000
logging_interval = 200

sample_size_m     = 100
sample_size_n     = 100
noise_level       = 0.01

num_experiments   = 100

num_flows = 2
lr        = 0.001
sigma_pre = 0.2

# --- Test Parameters ---
per_model_test_size = 100
heatmap_range = [[-2,2],[-2,2]]

percentiles = [0.1]
resolutions = [100]

os.makedirs("results", exist_ok=True)
log_file = open("results/final_percentile_heatmaps_and_uncertainty.txt", "w", encoding="utf-8")
def log(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

log("========== Parameter Settings ==========")
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
    log(f"{k}: {v}")
log("=========================================")

# --- Double Moon ---
def generate_two_moons(n_samples=100, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    return X


X_real_np = generate_two_moons(n_samples=sample_size_m, noise=noise_level)
X_real    = torch.tensor(X_real_np, dtype=torch.float32)
log(f"Fixed real data shape: {X_real.shape}")





experiment_models, experiment_losses = [], []
for exp in range(1, num_experiments+1):
    model = PlanarFlowModel(2, num_flows)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    prev, pat, min_l = None, 0, float('inf')
    U_small = torch.randn(sample_size_n, 2)
    for it in range(max_iters+1):
        opt.zero_grad()
        out  = model(U_small)
        loss = mmd_loss(out, X_real, sigma_pre)
        lval = loss.item()
        min_l = min(min_l, lval)
        if it % logging_interval == 0:
            log(f"[Two-sample exp {exp}] Iter {it}, Loss {lval:.6f}")
        loss.backward()
        opt.step()
        if it >= min_iters and prev is not None and abs(lval - prev) < tolerance:
            pat += 1
            if pat >= patience:
                log(f"[Two-sample exp {exp}] early stop at iter {it}")
                break
        prev = lval
    experiment_models.append(model)
    experiment_losses.append(min_l)
    log(f"[Two-sample exp {exp}] min MMD = {min_l:.6f}")

# --- 4. Assignm noise ---
sorted_idxs  = np.argsort(experiment_losses)
pts_data = {}

for p in percentiles:
    k = max(1, int(np.ceil(p * num_experiments)))
    idxs = sorted_idxs[:k]
    log(f"[Truncation Top {int(p*100)}%] Retained models: {k}, Per-model test points: {per_model_test_size}, Total: {per_model_test_size * k}")

    pts_list = []
    for i in idxs:
        U_test_i = torch.randn(per_model_test_size, 2)
        with torch.no_grad():
            out_i = experiment_models[i](U_test_i).cpu().numpy()
        pts_list.append(out_i)
    pts_data[p] = np.vstack(pts_list)

# --- 5. Draw Heatmaps ---
x0, x1 = heatmap_range[0]
y0, y1 = heatmap_range[1]

for bins in resolutions:
    subdir = f"results/{bins}bins"
    os.makedirs(subdir, exist_ok=True)

    global_max = 0.0
    density_maps = {}
    raw_maps = {}
    for p in percentiles:
        pts = pts_data[p]
        H_den, xe, ye = np.histogram2d(
            pts[:,0], pts[:,1],
            bins=bins,
            range=heatmap_range,
            density=True
        )
        C_cnt, _, _ = np.histogram2d(
            pts[:,0], pts[:,1],
            bins=bins,
            range=heatmap_range,
            density=False
        )
        density_maps[p] = (H_den, xe, ye)
        raw_maps[p]     = C_cnt
        global_max      = max(global_max, H_den.max())

    vmin = global_max * 1e-3

    for p in percentiles:
        pct = int(p*100)
        H, xe, ye = density_maps[p]

        plt.figure(figsize=(6,6))
        plt.imshow(H.T, origin='lower',
                   extent=[x0,x1,y0,y1],
                   aspect='auto',
                   vmin=0, vmax=global_max,
                   cmap='inferno')
        plt.title(f"Heatmap Top {pct}% — {bins}×{bins}")
        plt.colorbar(label="Density")
        plt.savefig(f"{subdir}/heatmap_top{pct}pct_{bins}bins.png")
        plt.close()

        plt.figure(figsize=(6,6))
        plt.imshow(H.T, origin='lower',
                   extent=[x0,x1,y0,y1],
                   aspect='auto',
                   norm=LogNorm(vmin=vmin, vmax=global_max),
                   cmap='inferno')
        plt.title(f"Enhanced Heatmap Top {pct}% — {bins}×{bins}")
        plt.colorbar(label="Density (log scale)")
        plt.savefig(f"{subdir}/heatmap_top{pct}pct_{bins}bins_enhanced.png")
        plt.close()

    xx, yy = np.meshgrid(
        np.linspace(x0, x1, bins),
        np.linspace(y0, y1, bins)
    )
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])

    for p in percentiles:
        pct = int(p*100)
        pts = pts_data[p].T
        kde = gaussian_kde(pts, bw_method=0.2)
        zz  = kde(grid_coords).reshape((bins, bins))

        plt.figure(figsize=(6,6))
        plt.imshow(zz.T, origin='lower',
                   extent=[x0,x1,y0,y1],
                   aspect='auto',
                   norm=LogNorm(vmin=1e-3, vmax=zz.max()),
                   cmap='inferno')
        plt.title(f"KDE LogNorm Heatmap Top {pct}% — {bins}×{bins}")
        plt.colorbar(label="Density (log KDE)")
        plt.savefig(f"{subdir}/heatmap_top{pct}pct_{bins}bins_kde.png")
        plt.close()

    prev = np.zeros((bins, bins))
    for p in percentiles:
        pct = int(p*100)
        curr = raw_maps[p]
        diff = curr - prev
        prev = curr.copy()

        plt.figure(figsize=(6,6))
        plt.imshow(diff.T, origin='lower',
                   extent=[x0,x1,y0,y1],
                   aspect='auto',
                   cmap='inferno')
        plt.title(f"Diff Heatmap Contribution Top {pct}% — {bins}×{bins}")
        plt.colorbar(label="New sample counts")
        plt.savefig(f"{subdir}/heatmap_diff_top{pct}pct_{bins}bins.png")
        plt.close()

        plt.figure(figsize=(6,6))
        plt.imshow(diff.T, origin='lower',
                   extent=[x0,x1,y0,y1],
                   aspect='auto',
                   norm=LogNorm(vmin=1, vmax=diff.max() if diff.max()>1 else 1),
                   cmap='inferno')
        plt.title(f"Enhanced Diff Heatmap Contribution Top {pct}% — {bins}×{bins}")
        plt.colorbar(label="New sample counts (log scale)")
        plt.savefig(f"{subdir}/heatmap_diff_top{pct}pct_{bins}bins_enhanced.png")
        plt.close()

# --- 6. Uncertainty Quantification ---
uncertainties    = []
selection_counts = []

for p in percentiles:
    k    = max(1, int(np.ceil(p * num_experiments)))
    idxs = sorted_idxs[:k]
    selection_counts.append(k)

    means = []
    for i in idxs:
        U_test_i = torch.randn(per_model_test_size, 2)
        with torch.no_grad():
            out_i = experiment_models[i](U_test_i)
            means.append(out_i.mean(0))
    means = torch.stack(means, dim=0)
    diffs = means - means.mean(0, keepdim=True)
    var   = (diffs.norm(dim=1)**2).mean().item()
    uncertainties.append(var)

fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot([p*100 for p in percentiles], uncertainties, marker='o', label='Avg Uncertainty')
ax1.set_xlabel('Top k% of Models')
ax1.set_ylabel('Average Uncertainty')
ax2 = ax1.twinx()
ax2.plot([p*100 for p in percentiles], selection_counts, marker='s', color='orange', label='Selected Models')
ax2.set_ylabel('Number of Selected Models')
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='best')
plt.title('Model Percentile vs Uncertainty & Selection Count')
plt.savefig("results/percentile_vs_uncertainty_and_selection.png")
plt.close()
log("Saved percentile_vs_uncertainty_and_selection.png")

log_file.close()
