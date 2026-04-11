import numpy as np
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half   = self.dim // 2
        freqs  = torch.exp(
            -np.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConditioningBlock(nn.Module):
    """Projects shared embedding to a per-block channel size for injection."""
    def __init__(self, emb_dim: int, out_ch: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, out_ch)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.proj(emb).unsqueeze(-1)


class ConditionalDenoiser(nn.Module):
    """
    Conv1D U-Net denoiser for multi-asset windows (B, n_assets, seq_len).
    Conditioning injected at enc1, enc2, enc3 and bottleneck (Fix 2).
    """
    def __init__(self, seq_len: int = 1260, in_ch: int = 7, cond_dim: int = 4,
                 time_emb_dim: int = 64, base_ch: int = 64):
        super().__init__()
        emb_dim = time_emb_dim

        self.time_emb  = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, emb_dim)
        self.cond_proj = nn.Linear(cond_dim, emb_dim)

        self.cond_e1 = ConditioningBlock(emb_dim, base_ch)
        self.cond_e2 = ConditioningBlock(emb_dim, base_ch * 2)
        self.cond_e3 = ConditioningBlock(emb_dim, base_ch * 4)
        self.cond_m  = ConditioningBlock(emb_dim, base_ch * 4)

        self.enc1 = self._block(in_ch,       base_ch,     dilation=1)
        self.enc2 = self._block(base_ch,     base_ch * 2, dilation=2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4, dilation=4)
        self.mid  = self._block(base_ch * 4, base_ch * 4, dilation=8)
        self.dec3 = self._block(base_ch * 8, base_ch * 2, dilation=4)
        self.dec2 = self._block(base_ch * 4, base_ch,     dilation=2)
        self.dec1 = self._block(base_ch * 2, base_ch,     dilation=1)
        self.out  = nn.Conv1d(base_ch, in_ch, kernel_size=1)

    def _block(self, in_ch: int, out_ch: int, dilation: int) -> nn.Sequential:
        pad = dilation
        return nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation),
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        emb = self.time_proj(self.time_emb(t)) + self.cond_proj(c)
        e1  = self.enc1(x)  + self.cond_e1(emb)
        e2  = self.enc2(e1) + self.cond_e2(emb)
        e3  = self.enc3(e2) + self.cond_e3(emb)
        m   = self.mid(e3)  + self.cond_m(emb)
        d3  = self.dec3(torch.cat([m,  e3], dim=1))
        d2  = self.dec2(torch.cat([d3, e2], dim=1))
        d1  = self.dec1(torch.cat([d2, e1], dim=1))
        return self.out(d1)