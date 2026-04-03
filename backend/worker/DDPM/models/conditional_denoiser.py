"""
Stage 2: Conditional DDPM Denoiser
Conv1D U-Net with strong conditioning on market regimes.
Implements Classifier-Free Guidance (CFG) at inference.
"""

import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional time embedding"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Time indices (batch_size,)

        Returns:
            Time embeddings (batch_size, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConditioningBlock(nn.Module):
    """Projects time+cond embedding to per-layer conditioning (scale injection)"""

    def __init__(self, emb_dim, out_ch):
        super().__init__()
        self.proj = nn.Linear(emb_dim, out_ch)

    def forward(self, emb):
        """
        Args:
            emb: Embedding (batch_size, emb_dim)

        Returns:
            Channel-wise conditioning (batch_size, out_ch, 1)
        """
        return self.proj(emb).unsqueeze(-1)


class ConditionalDenoiser(nn.Module):
    """
    Stage 2 Conditional Denoiser for 7-asset windows (B, 7, 1260).
    Conditioning injected at EVERY encoder + bottleneck block (FIX 2 from notebook).
    Supports Classifier-Free Guidance (CFG) at inference.
    """

    def __init__(
        self,
        seq_len=1260,
        in_ch=7,
        cond_dim=4,
        time_emb_dim=64,
        base_ch=64,
    ):
        super().__init__()
        self.seq_len = seq_len
        emb_dim = time_emb_dim

        # Time + conditioning shared embedding
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, emb_dim)
        self.cond_proj = nn.Linear(cond_dim, emb_dim)

        # Per-block conditioning projections (strong conditioning at every layer)
        self.cond_e1 = ConditioningBlock(emb_dim, base_ch)
        self.cond_e2 = ConditioningBlock(emb_dim, base_ch * 2)
        self.cond_e3 = ConditioningBlock(emb_dim, base_ch * 4)
        self.cond_m = ConditioningBlock(emb_dim, base_ch * 4)

        # Encoder blocks with exponential dilation
        self.enc1 = self._block(in_ch, base_ch, dilation=1)
        self.enc2 = self._block(base_ch, base_ch * 2, dilation=2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4, dilation=4)

        # Bottleneck
        self.mid = self._block(base_ch * 4, base_ch * 4, dilation=8)

        # Decoder blocks (skip connections, no extra conditioning)
        self.dec3 = self._block(base_ch * 4 + base_ch * 4, base_ch * 2, dilation=4)
        self.dec2 = self._block(base_ch * 2 + base_ch * 2, base_ch, dilation=2)
        self.dec1 = self._block(base_ch + base_ch, base_ch, dilation=1)

        # Output projection
        self.out = nn.Conv1d(base_ch, in_ch, kernel_size=1)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"ConditionalDenoiser created with {total_params:,} parameters")

    def _block(self, in_ch, out_ch, dilation):
        """
        Convolutional block: Conv1d → GroupNorm → SiLU → Conv1d → GroupNorm → SiLU
        """
        pad = dilation
        return nn.Sequential(
            nn.Conv1d(
                in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation
            ),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(
                out_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation
            ),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x, t, c):
        """
        Args:
            x: Input data (B, 7, seq_len)
            t: Time indices (B,) — 0 to T-1
            c: Conditioning vector (B, 4) — [realised_vol, drift, tail_index, momentum]
               Use zeros for unconditional (CFG) paths

        Returns:
            Noise prediction (B, 7, seq_len)
        """
        # Shared embedding: time + conditioning
        emb = self.time_proj(self.time_emb(t)) + self.cond_proj(c)  # (B, emb_dim)

        # Encoder with conditioning injection at each block
        e1 = self.enc1(x) + self.cond_e1(emb)  # (B, base_ch, L)
        e2 = self.enc2(e1) + self.cond_e2(emb)  # (B, base_ch*2, L)
        e3 = self.enc3(e2) + self.cond_e3(emb)  # (B, base_ch*4, L)

        # Bottleneck
        m = self.mid(e3) + self.cond_m(emb)  # (B, base_ch*4, L)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([m, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))

        return self.out(d1)  # (B, 7, seq_len)
