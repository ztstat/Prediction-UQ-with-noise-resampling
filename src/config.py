# src/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List


@dataclass(frozen=True)
class ExperimentConfig:
    # training / fitting
    tolerance: float = 0.01
    patience: int = 5
    min_iters: int = 100
    max_iters: int = 5000
    logging_interval: int = 200

    # data
    sample_size_m: int = 100
    sample_size_n: int = 100
    noise_level: float = 0.01

    # repeated training (ensemble)
    num_experiments: int = 5

    # generator model
    num_flows: int = 2
    lr: float = 0.001
    sigma_pre: float = 0.2

    # predictive evaluation (prediction UQ)
    per_model_test_size: int = 100
    heatmap_range: List[List[float]] = None  # set in __post_init__
    percentiles: List[float] = None          # set in __post_init__
    resolutions: List[int] = None            # set in __post_init__

    # outputs
    results_dir: str = "results"

    def __post_init__(self):
        object.__setattr__(self, "heatmap_range", self.heatmap_range or [[-2, 2], [-2, 2]])
        object.__setattr__(self, "percentiles", self.percentiles or [0.1])
        object.__setattr__(self, "resolutions", self.resolutions or [100])

    def to_dict(self):
        return asdict(self)
