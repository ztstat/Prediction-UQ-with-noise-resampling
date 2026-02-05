# Uncertainty Quantification of Prediction

This repository presents a **prediction-focused uncertainty quantification framework**
based on **truncated ensembles of generative models**.


---

## Key Idea

> **Uncertainty is defined as instability of predictions under model selection,
> rather than uncertainty of model parameters.**

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

## How to Run

```bash
python scripts/run_two_moons.py
