# src/train.py
from typing import Tuple
import torch
from torch.optim import Adam

from src.flows import PlanarFlowModel
from src.mmd import mmd_loss
from src.utils import Logger

def train_one_model(
    x_real: torch.Tensor,
    sample_size_n: int,
    num_flows: int,
    lr: float,
    sigma_pre: float,
    min_iters: int,
    max_iters: int,
    tolerance: float,
    patience: int,
    logging_interval: int,
    logger: Logger,
    exp_id: int,
) -> Tuple[PlanarFlowModel, float]:
    model = PlanarFlowModel(2, num_flows)
    opt = Adam(model.parameters(), lr=lr)

    prev, pat, min_l = None, 0, float("inf")
    u_small = torch.randn(sample_size_n, 2)

    for it in range(max_iters + 1):
        opt.zero_grad()
        out = model(u_small)
        loss = mmd_loss(out, x_real, sigma_pre)
        lval = float(loss.item())
        min_l = min(min_l, lval)

        if it % logging_interval == 0:
            logger.log(f"[Two-sample exp {exp_id}] Iter {it}, Loss {lval:.6f}")

        loss.backward()
        opt.step()

        if it >= min_iters and prev is not None and abs(lval - prev) < tolerance:
            pat += 1
            if pat >= patience:
                logger.log(f"[Two-sample exp {exp_id}] early stop at iter {it}")
                break
        prev = lval

    logger.log(f"[Two-sample exp {exp_id}] min MMD = {min_l:.6f}")
    return model, min_l
