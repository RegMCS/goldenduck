"""
Stage 1: Unconditional DDPM Denoiser
Conv1D U-Net architecture with time embedding
"""

import math
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
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
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """
    Residual Conv1D block with AdaGN (Adaptive Group Norm) time conditioning.
    Time embeddings are injected as scale and shift.
    """

    def __init__(self, in_ch, out_ch, time_dim, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch * 2)  # scale + shift
        self.skip = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        """
        Args:
            x: Input (batch_size, in_ch, length)
            t_emb: Time embedding (batch_size, time_dim)

        Returns:
            Output (batch_size, out_ch, length)
        """
        ts = self.time_proj(self.act(t_emb))
        scale, shift = ts.chunk(2, dim=1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)

        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h * (1 + scale) + shift  # AdaGN injection
        h = self.act(self.norm2(h))
        h = self.conv2(h)

        return self.skip(x) + h


class Conv1DUNet(nn.Module):
    """
    Unconditional U-Net denoiser for Stage 1 DDPM.
    Architecture:
      - Input projection
      - Encoder (depth layers with increasing dilation)
      - Bottleneck
      - Decoder (mirrored encoder with skip connections)
      - Output projection
    """

    def __init__(
        self, num_assets, base_channels=64, time_emb_dim=128, depth=4
    ):
        super().__init__()
        C = base_channels

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        # Input projection
        self.input_proj = nn.Conv1d(num_assets, C, 1)

        # Encoder blocks with exponential dilation
        self.encoder = nn.ModuleList(
            [ResBlock(C, C, time_emb_dim, dilation=2**i) for i in range(depth)]
        )

        # Bottleneck
        self.bottleneck = ResBlock(C, C, time_emb_dim, dilation=2**depth)

        # Decoder blocks (mirrored with skip connections)
        self.decoder = nn.ModuleList(
            [
                ResBlock(C * 2, C, time_emb_dim, dilation=2**i)
                for i in reversed(range(depth))
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, C), nn.SiLU(), nn.Conv1d(C, num_assets, 1)
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Conv1DUNet created with {total_params:,} parameters")

    def forward(self, x, t):
        """
        Args:
            x: Noisy input (batch_size, num_assets, window_len)
            t: Time indices (batch_size,)

        Returns:
            Noise prediction (batch_size, num_assets, window_len)
        """
        t_emb = self.time_mlp(t)
        h = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for block in self.encoder:
            h = block(h, t_emb)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder with skip concatenation
        for block, skip in zip(self.decoder, reversed(skips)):
            h = block(torch.cat([h, skip], dim=1), t_emb)

        # Output
        return self.output_proj(h)
