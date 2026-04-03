"""
DDPM Noise Schedule (Stage 1)
Implements cosine-based variance schedule with fixed diffusion steps
"""

import math
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class DDPMScheduler:
    """
    Diffusion process scheduler with cosine noise schedule.
    Implements q_sample (forward process) and p_sample_loop (reverse process).
    """

    def __init__(self, T=200, s=0.008, device="cpu"):
        """
        Args:
            T: Number of diffusion steps (default 200)
            s: Scheduling constant for cosine schedule (default 0.008)
            device: torch device ("cpu" or "cuda")
        """
        self.T = T
        self.device = device

        # Compute alpha_bar using cosine schedule
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        alpha_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        # Compute betas from alpha_bar
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.02)
        alphas = 1.0 - betas

        # Store on device
        self.alpha_bar = alpha_bar[1:].to(device)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        self.betas = betas.to(device)
        self.sqrt_ab = self.alpha_bar.sqrt()
        self.sqrt_one_m_ab = (1 - self.alpha_bar).sqrt()

        # Posterior variance for reverse process
        self.posterior_var = (
            betas.to(device)
            * (1 - self.alpha_bar_prev)
            / (1 - self.alpha_bar).clamp(min=1e-5)
        ).clamp(min=1e-8)

        logger.info(
            f"DDPMScheduler initialized: T={T}, beta_range=[{betas.min():.4f}, {betas.max():.4f}]"
        )

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: Add noise to x0 at timestep t
        x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x0: Clean data (batch_size, channels, length)
            t: Timestep indices (batch_size,)
            noise: Optional pre-generated noise (default: random)

        Returns:
            x_t: Noisy data at timestep t
            noise: The noise tensor used
        """
        if noise is None:
            noise = torch.randn_like(x0)

        a = self.sqrt_ab[t].view(-1, 1, 1)
        b = self.sqrt_one_m_ab[t].view(-1, 1, 1)

        return a * x0 + b * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device):
        """
        Reverse process: Iteratively denoise from pure noise to data
        Implements DDPM sampling without guidance

        Args:
            model: Denoiser network
            shape: Output shape (batch_size, channels, length)
            device: torch device

        Returns:
            x_0: Generated samples
        """
        x = torch.randn(shape, device=device)

        for t_idx in reversed(range(self.T)):
            t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch)

            alpha = 1.0 - self.betas[t_idx]
            sqrt_one_m = self.sqrt_one_m_ab[t_idx].clamp(min=1e-5)
            coef = self.betas[t_idx] / sqrt_one_m

            # Posterior mean
            mean = (1.0 / alpha.sqrt()) * (x - coef * pred_noise)
            mean = mean.clamp(-2, 2)

            if t_idx > 0:
                # Add noise for t > 0
                noise = torch.randn_like(x)
                std = self.posterior_var[t_idx].sqrt().clamp(min=1e-8)
                x = mean + std * noise
            else:
                # Final step (t=0): no noise added
                x = mean

            x = x.clamp(-2, 2)

        return x
