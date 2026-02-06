# Uncertainty Quantification of Prediction

This repository presents a **prediction-focused uncertainty quantification framework**
based on **truncated ensembles of generative models**.


---

## Key Idea

> **We construct prediction samples by resampling latent noise to analyze the
> plug-in predictive distribution induced by the flow model.**
>
> We study *prediction uncertainty* directly in the predictive (output) space,
> rather than parameter inference. After fitting a generative flow, fresh latent
> noise is resampled and propagated through the learned mapping to produce
> predictive outputs, forming a bootstrap-like pseudo predictive distribution.
>
> This truncation-based resampling scheme allows us to quantify uncertainty of
> future observations and supports downstream tasks such as predictive density
> visualization and prediction interval construction.

The pipeline:
1. Train multiple generative models on the same data
2. Rank models by Maximum Mean Discrepancy (MMD)
3. Truncate to the top-performing models
4. Generate predictive samples
5. Quantify uncertainty via variability in predictive space

This aligns with **modern statistics and machine learning practice** where
**prediction quality is the primary objective**.

---

## Pipeline Overview

1. Generate reference data (two-moons)
2. Train an ensemble of flow-based generators
3. Rank models by MMD loss
4. Truncate ensemble (top-k%)
5. Sample predictive outputs
6. Quantify predictive uncertainty
7. Visualize predictive distributions and uncertainty

---

## Example Results

The following figures are generated using the default configuration on a laptop-scale run.
Full experimental results are omitted for reproducibility and storage reasons.


## How to Run

```bash
python scripts/run_two_moons.py
