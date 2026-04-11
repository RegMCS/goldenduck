
import torch
import numpy as np


def get_noise_schedule(T: int = 200, beta_start: float = 1e-4,
                       beta_end: float = 0.02, device: str = "cpu"):
    betas     = torch.linspace(beta_start, beta_end, T).to(device)
    alphas    = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def q_sample(x0: torch.Tensor, t: torch.Tensor,
             noise: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
    ab = alpha_bar[t][:, None, None]
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise


def tail_weighted_loss(pred: torch.Tensor, target: torch.Tensor,
                       tail_thresh: float = 0.5,
                       tail_weight: float = 3.0) -> torch.Tensor:
    weights = torch.where(
        target.abs() > tail_thresh,
        torch.full_like(target, tail_weight),
        torch.ones_like(target)
    )
    return (weights * (pred - target) ** 2).mean()