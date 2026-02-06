# src/config.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    tolerance: float = 1e-3
    patience: int = 5
    min_iters: int = 200
    max_iters: int = 3000
    logging_interval: int = 200

    sample_size_m: int = 100
    sample_size_n: int = 100
    noise_level: float = 0.01

    num_experiments: int = 50

    num_flows: int = 2
    lr: float = 1e-3
    sigma_pre: float = 0.2

    per_model_test_size: int = 200
    heatmap_range: Optional[List[List[float]]] = None
    percentiles: Optional[List[float]] = None
    resolutions: Optional[List[int]] = None

    results_dir: str = "results"

    def __post_init__(self):
        object.__setattr__(self, "heatmap_range", self.heatmap_range or [[-2, 2], [-2, 2]])
        object.__setattr__(self, "percentiles", self.percentiles or [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
        object.__setattr__(self, "resolutions", self.resolutions or [100])

    def to_dict(self):
        return asdict(self)


def default_config() -> ExperimentConfig:
    """
    Laptop-friendly default.
    """
    return ExperimentConfig()


def large_scale_config() -> ExperimentConfig:
    """
    Reference config used for large-scale server runs.
    """
    return ExperimentConfig(
        tolerance=1e-4,
        min_iters=600,
        max_iters=5000,
        num_experiments=100,
        per_model_test_size=100,
        percentiles=[0.1],
        resolutions=[100],
    )
