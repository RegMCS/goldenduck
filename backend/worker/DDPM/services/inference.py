"""
Inference service for DDPM Stage 1 & Stage 2.
Implements unconditional (Stage 1) and conditional with CFG (Stage 2) path generation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InferenceService:
    """
    High-level inference interface for both DDPM stages.
    Stage 1: Unconditional denoising
    Stage 2: Conditional denoising with Classifier-Free Guidance (CFG)
    """

    def __init__(self, device="cpu"):
        """
        Args:
            device: torch device ("cpu" or "cuda")
        """
        self.device = device

    def generate_stage1(
        self,
        model: nn.Module,
        scheduler,
        num_paths: int = 2000,
        shape: Tuple[int, int, int] = (1, 7, 1260),
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Stage 1 paths: unconditional DDPM sampling.

        Args:
            model: Conv1DUNet denoiser (trained on unconditional task)
            scheduler: DDPMScheduler with noise schedule
            num_paths: Number of paths to generate
            shape: (batch_size, num_assets, seq_len) — adjust batch_size from num_paths
            seed: Random seed for reproducibility

        Returns:
            Generated paths array of shape (num_paths, num_assets, seq_len)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        model.eval()
        batch_size = shape[0]
        generated = []

        num_batches = (num_paths + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                remaining = min(batch_size, num_paths - batch_idx * batch_size)
                batch_shape = (remaining, shape[1], shape[2])

                # Run p_sample_loop from scheduler
                x_0 = scheduler.p_sample_loop(model, batch_shape, self.device)
                generated.append(x_0.cpu().numpy())

        generated = np.concatenate(generated, axis=0)[:num_paths]
        logger.info(f"Stage 1: Generated {generated.shape[0]} paths")
        return generated

    def generate_stage2_cfg(
        self,
        model: nn.Module,
        scheduler,
        conditioning: np.ndarray,
        num_paths: int = 2000,
        guidance_scale: float = 3.0,
        shape: Tuple[int, int, int] = (1, 7, 1260),
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Stage 2 paths: conditional DDPM with Classifier-Free Guidance.

        CFG formula:
          ε̂(x_t, t, c) = ε_u(x_t, t) + guidance_scale * (ε_c(x_t, t, c) - ε_u(x_t, t))
        where ε_u = unconditional, ε_c = conditional, c = conditioning vector

        Args:
            model: ConditionalDenoiser (trained on conditional task with CFG dropout)
            scheduler: DDPMScheduler with noise schedule
            conditioning: Array of shape (4,) — [realised_vol, drift, tail_index, momentum]
            num_paths: Number of paths to generate
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
            shape: (batch_size, num_assets, seq_len)
            seed: Random seed for reproducibility

        Returns:
            Generated paths array of shape (num_paths, num_assets, seq_len)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        model.eval()
        batch_size = shape[0]
        generated = []

        num_batches = (num_paths + batch_size - 1) // batch_size

        # Expand conditioning to batch
        c_batch = torch.from_numpy(conditioning).float().to(self.device)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                remaining = min(batch_size, num_paths - batch_idx * batch_size)
                batch_shape = (remaining, shape[1], shape[2])

                # Generate using CFG
                x_0 = self._p_sample_loop_cfg(
                    model, scheduler, batch_shape, c_batch[:remaining], guidance_scale
                )
                generated.append(x_0.cpu().numpy())

        generated = np.concatenate(generated, axis=0)[:num_paths]
        logger.info(
            f"Stage 2 (CFG scale={guidance_scale}): Generated {generated.shape[0]} paths"
        )
        return generated

    @torch.no_grad()
    def _p_sample_loop_cfg(
        self,
        model: nn.Module,
        scheduler,
        shape: Tuple[int, int, int],
        conditioning: torch.Tensor,
        guidance_scale: float = 3.0,
    ) -> torch.Tensor:
        """
        Iterative denoising with Classifier-Free Guidance.

        Args:
            model: ConditionalDenoiser
            scheduler: DDPMScheduler
            shape: Batch shape (batch_size, channels, length)
            conditioning: Conditioning tensor (batch_size, 4)
            guidance_scale: CFG scale

        Returns:
            Denoised tensor of shape (batch_size, channels, length)
        """
        x = torch.randn(shape, device=self.device)
        batch_size = shape[0]

        for t_idx in reversed(range(scheduler.T)):
            t_batch = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)

            # Conditional prediction
            pred_noise_cond = model(x, t_batch, conditioning)

            # Unconditional prediction (zero conditioning)
            c_uncond = torch.zeros_like(conditioning)
            pred_noise_uncond = model(x, t_batch, c_uncond)

            # CFG: interpolate between unconditional and conditional
            pred_noise = pred_noise_uncond + guidance_scale * (
                pred_noise_cond - pred_noise_uncond
            )

            alpha = 1.0 - scheduler.betas[t_idx]
            sqrt_one_m = scheduler.sqrt_one_m_ab[t_idx].clamp(min=1e-5)
            coef = scheduler.betas[t_idx] / sqrt_one_m

            # Posterior mean
            mean = (1.0 / alpha.sqrt()) * (x - coef * pred_noise)
            mean = mean.clamp(-2, 2)

            if t_idx > 0:
                noise = torch.randn_like(x)
                std = scheduler.posterior_var[t_idx].sqrt().clamp(min=1e-8)
                x = mean + std * noise
            else:
                x = mean

            x = x.clamp(-2, 2)

        return x

    def generate_paths(
        self,
        model: nn.Module,
        scheduler,
        stage: int = 1,
        num_paths: int = 2000,
        conditioning: Optional[np.ndarray] = None,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Unified interface for both stages.

        Args:
            model: Denoiser network (Conv1DUNet or ConditionalDenoiser)
            scheduler: DDPMScheduler
            stage: 1 or 2
            num_paths: Number of paths
            conditioning: [realised_vol, drift, tail_index, momentum] (required for stage=2)
            guidance_scale: CFG scale (stage=2 only)
            seed: Random seed

        Returns:
            Generated paths array (num_paths, 7, 1260)
        """
        if stage == 1:
            return self.generate_stage1(
                model, scheduler, num_paths=num_paths, seed=seed
            )
        elif stage == 2:
            assert conditioning is not None, "Conditioning required for stage=2"
            return self.generate_stage2_cfg(
                model,
                scheduler,
                conditioning,
                num_paths=num_paths,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")
