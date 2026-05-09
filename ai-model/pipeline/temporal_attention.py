"""
NovaCine — AI Model Pipeline
Temporal Attention Module + 3D U-Net utilities
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TemporalAttention(nn.Module):
    """
    Temporal self-attention across video frames.

    For input (B, C, F, H, W):
      1. Reshape to (B·H·W, F, C)
      2. Apply multi-head attention across frame axis
      3. Add learned relative temporal bias B_temp ∈ ℝ^(F×F)
      4. Reshape back

    Mathematical:
      Attn(Q,K,V) = softmax(QKᵀ/√d_k + B_temp) · V
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_frames: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable relative temporal bias
        self.rel_bias = nn.Parameter(torch.zeros(max_frames, max_frames))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, H, W) or (B·H·W, F, C)
        returns same shape
        """
        reshape_back = False
        if x.dim() == 5:
            B, C, F, H, W = x.shape
            x = rearrange(x, 'b c f h w -> (b h w) f c')
            reshape_back = True

        BHW, F, C = x.shape
        residual = x
        x = self.norm(x)

        # QKV projection
        qkv = self.to_qkv(x)  # (BHW, F, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Multi-head reshape
        q = rearrange(q, 'b f (h d) -> b h f d', h=self.num_heads)
        k = rearrange(k, 'b f (h d) -> b h f d', h=self.num_heads)
        v = rearrange(v, 'b f (h d) -> b h f d', h=self.num_heads)

        # Attention with temporal bias
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        bias = self.rel_bias[:F, :F]
        attn = attn + bias.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h f d -> b f (h d)')
        out = self.proj(out) + residual

        if reshape_back:
            out = rearrange(out, '(b h w) f c -> b c f h w', b=B, h=H, w=W)

        return out


class TemporalConvBlock(nn.Module):
    """
    1D temporal convolution applied per (H·W) spatial token.
    Captures local temporal motion patterns.

    Applied as: (B, C, F, H, W) → rearrange → conv1d → rearrange back
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,  # depthwise for efficiency
        )
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, F, H, W = x.shape
        # Merge spatial dims, apply temporal conv
        x_flat = rearrange(x, 'b c f h w -> (b h w) c f')
        out = self.act(self.norm(self.conv(x_flat)))
        return rearrange(out, '(b h w) c f -> b c f h w', b=B, h=H, w=W) + x


class MotionAwareAttention(nn.Module):
    """
    Enhanced temporal attention that uses optical-flow-estimated motion
    to modulate attention weights (Research enhancement).

    Flow magnitude: Σ|F_t| used as soft gate on temporal attention strength.
    """

    def __init__(self, dim: int, num_heads: int = 8, max_frames: int = 64):
        super().__init__()
        self.base_attn = TemporalAttention(dim, num_heads, max_frames)
        self.motion_gate = nn.Sequential(
            nn.Linear(1, dim),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor,
        motion_magnitudes: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.base_attn(x)
        if motion_magnitudes is not None and x.dim() == 5:
            # motion_magnitudes: (B, F) → gate applied per-frame
            gate = self.motion_gate(motion_magnitudes.unsqueeze(-1))  # (B, F, C)
            gate = gate.unsqueeze(-1).unsqueeze(-1)  # (B, F, C, 1, 1)
            gate = gate.permute(0, 2, 1, 3, 4)       # (B, C, F, 1, 1)
            out = out * gate
        return out
