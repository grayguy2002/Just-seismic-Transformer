"""1D Rotary Position Embedding for seismic time-series."""

import torch
from torch import nn
from einops import rearrange


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack([-x2, x1], dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryEmbedding1D(nn.Module):
    """
    1D RoPE along the time (patch) axis.

    For a sequence of ``seq_len`` patches, each head dimension is split;
    the first half is rotated by cos/sin at frequencies that increase
    from 1/theta to 1/theta^(dim/2) along the sequence dimension.
    """

    def __init__(
        self,
        dim: int,               # head_dim (hidden // num_heads)
        seq_len: int = 50,      # number of waveform patches
        num_cond_tokens: int = 3,
        theta: float = 10000.0,
    ):
        super().__init__()
        half_dim = dim // 2
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim).float() / half_dim))
        # positions: [0, 1, ..., seq_len-1] for patch tokens only
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)                         # [seq_len, half_dim]
        emb = freqs.repeat_interleave(2, dim=-1)               # [seq_len, dim] — pairs share freq

        if num_cond_tokens > 0:
            # condition tokens get cos=1, sin=0 (no rotation)
            cond_cos = torch.ones(num_cond_tokens, dim)
            cond_sin = torch.zeros(num_cond_tokens, dim)
            cos = torch.cat([cond_cos, emb.cos()], dim=0)     # [cond+seq, dim]
            sin = torch.cat([cond_sin, emb.sin()], dim=0)
        else:
            cos = emb.cos()
            sin = emb.sin()

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., total_tokens, head_dim). cos/sin broadcast naturally."""
        return x * self.cos + rotate_half(x) * self.sin
