"""JsT — Just seismic Transformer.

Conditional generative model for 3-component P-wave seismograms.
Architecture follows JiT (Kaiming He et al., 2025) adapted to 1-D:

  - Pixel-space x-prediction diffusion (no VAE / tokenizer).
  - 1-D patch embedding over the time axis.
  - 1-D RoPE positional encoding for patch tokens.
  - In-context condition tokens (source / path / receiver) prepended
    from layer 0 with learnable position embeddings.
  - adaLN-Zero timestep modulation in every transformer block
    (timestep only — condition information flows via self-attention).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope_1d import RotaryEmbedding1D


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(dtype) * self.weight


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """adaLN modulation: x * (1 + scale) + shift"""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep encoding → MLP (same as JiT)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def _embed(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(0, half, dtype=torch.float32, device=t.device)
            * (torch.log(torch.tensor(max_period, device=t.device)) / half)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._embed(t, self.freq_dim))


# ---------------------------------------------------------------------------
# Patch embedding (1-D)
# ---------------------------------------------------------------------------

class PatchEmbed1D(nn.Module):
    """Split a 3×N waveform into non-overlapping 1-D patches."""

    def __init__(
        self,
        n_samples: int = 3200,
        patch_size: int = 64,
        in_channels: int = 3,
        bottleneck_dim: int = 128,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.num_patches = n_samples // patch_size

        self.proj1 = nn.Conv1d(
            in_channels, bottleneck_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.proj2 = nn.Conv1d(bottleneck_dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, num_patches, embed_dim)"""
        x = self.proj2(self.proj1(x))          # (B, embed_dim, num_patches)
        return x.transpose(1, 2)               # (B, num_patches, embed_dim)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qk_norm: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: RotaryEmbedding1D) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE is applied per-head to the full token sequence.
        # reshape to [B*heads, N, head_dim] for rope, then back.
        B2, H, N2, D2 = q.shape
        q = rope(q.reshape(B2 * H, N2, D2)).reshape(B2, H, N2, D2)
        k = rope(k.reshape(B2 * H, N2, D2)).reshape(B2, H, N2, D2)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Feed-forward (SwiGLU)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.w3 = nn.Linear(hidden_dim, dim, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


# ---------------------------------------------------------------------------
# Transformer block (adaLN-Zero + FiLM condition modulation)
# ---------------------------------------------------------------------------

class ConditionFiLM(nn.Module):
    """Map a condition token to per-channel scale/shift for FiLM modulation."""

    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, cond_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cond_tok: (B, hidden) or (B, n_group_tokens, hidden)."""
        if cond_tok.ndim == 3:
            cond_tok = cond_tok.mean(dim=1)
        out = self.mlp(cond_tok)                        # (B, 2*hidden)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma.unsqueeze(1), beta.unsqueeze(1)    # (B, 1, hidden)


class JsTBlock(nn.Module):
    """
    Transformer block with:
      1. FiLM (entry): condition-driven scale/shift BEFORE attention.
         Force-injects source/path/receiver information.
      2. adaLN-Zero (internal): timestep-driven modulation of Norm+Attn+MLP.
         Same as DiT/JiT — controls denoising progression.
      3. In-context tokens still flow through self-attention for soft routing.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(
            hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        rope: RotaryEmbedding1D,
        film_gamma: torch.Tensor | None = None,
        film_beta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # FiLM entry modulation — condition-injected BEFORE attention.
        # If gamma/beta are None, this is a no-op (no FiLM for this block).
        if film_gamma is not None:
            x = x * (1.0 + film_gamma) + film_beta

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope=rope,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp),
        )
        return x


# ---------------------------------------------------------------------------
# Final unpatchify layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Project tokens back to waveform patches, then reassemble."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """(B, N, hidden) → (B, C, N*patch_size)"""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        x = self.linear(x)                                # (B, N, patch_size * C)
        B, N, _ = x.shape
        x = x.reshape(B, N, self.out_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3)                        # (B, C, N, patch_size)
        return x.reshape(B, self.out_channels, N * self.patch_size)


# ---------------------------------------------------------------------------
# JsT — top-level model
# ---------------------------------------------------------------------------

class JsT(nn.Module):
    """
    Just seismic Transformer.

    Parameters
    ----------
    n_samples: waveform length in samples (default 3200 → 80 s at 40 Hz).
    patch_size: samples per 1-D patch.
    in_channels: 3 (Z, N, E or E, N, Z — model is channel-agnostic).
    hidden_size: transformer width.
    depth: number of transformer blocks.
    num_heads: attention heads.
    n_cond_tokens: number of condition tokens prepended (default 11).
    """

    def __init__(
        self,
        n_samples: int = 3200,
        patch_size: int = 64,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        bottleneck_dim: int = 128,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        n_cond_tokens: int = 11,
        cond_token_groups: dict[str, list[int]] | None = None,
        film_groups: dict[str, list[int]] | None = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.hidden_size = hidden_size
        self.n_cond_tokens = n_cond_tokens
        if cond_token_groups is None:
            cond_token_groups = {
                "source": [0, 1, 2],
                "path": [3, 4, 5, 6],
                "receiver": [7, 8, 9, 10],
            }
        self.cond_token_groups = {k: list(v) for k, v in cond_token_groups.items()}
        if film_groups is None:
            film_groups = {
                "source":   [0, 1, 2],
                "path":     [3, 4, 5],
                "receiver": [6, 7],
            }
        self.film_groups = film_groups
        for group_name in self.film_groups:
            if group_name not in self.cond_token_groups:
                raise ValueError(f"FiLM group {group_name!r} has no condition-token group")
            if not self.cond_token_groups[group_name]:
                raise ValueError(f"Condition-token group {group_name!r} is empty")
            bad = [i for i in self.cond_token_groups[group_name] if i < 0 or i >= n_cond_tokens]
            if bad:
                raise ValueError(f"Condition-token group {group_name!r} has invalid indices {bad}")
        # Timestep embedder (adaLN source)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Patch embedder
        self.x_embedder = PatchEmbed1D(
            n_samples=n_samples,
            patch_size=patch_size,
            in_channels=in_channels,
            bottleneck_dim=bottleneck_dim,
            embed_dim=hidden_size,
        )
        num_patches = self.x_embedder.num_patches
        self.num_patches = num_patches

        # Fixed sin-cos position embedding for patch tokens
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False,
        )

        # Learnable position embeddings for condition tokens
        self.cond_pos_embed = nn.Parameter(
            torch.zeros(1, n_cond_tokens, hidden_size),
        )

        # Learnable null tokens (CFG unconditional forward)
        self.null_tokens = nn.Parameter(
            torch.zeros(1, n_cond_tokens, hidden_size),
        )

        # 1-D RoPE (covers both condition + patch tokens)
        self.rope = RotaryEmbedding1D(
            dim=hidden_size // num_heads,
            seq_len=num_patches,
            num_cond_tokens=n_cond_tokens,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            JsTBlock(
                hidden_size, num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
            )
            for i in range(depth)
        ])

        # --- FiLM modules per condition group ---
        self.film_modules = nn.ModuleDict()
        self._block_film: dict[int, tuple[str, ConditionFiLM]] = {}
        for group_name, block_indices in self.film_groups.items():
            film_mod = ConditionFiLM(hidden_size, hidden_size)
            self.film_modules[group_name] = film_mod
            for bi in block_indices:
                self._block_film[bi] = (group_name, film_mod)

        # Output projection
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation (following JiT conventions)
    # ------------------------------------------------------------------

    def _init_weights(self):
        # General: xavier uniform for Linear
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.apply(_basic_init)

        # 1-D sincos pos-embed (fixed)
        pe = _get_1d_sincos_pos_embed(self.hidden_size, self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # Condition position embeddings: small random init
        nn.init.normal_(self.cond_pos_embed, std=0.02)
        nn.init.normal_(self.null_tokens, std=0.02)

        # Patch embed Conv1d treated like Linear
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view(w1.shape[0], -1))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view(w2.shape[0], -1))
        nn.init.constant_(self.x_embedder.proj2.bias, 0.0)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers (so blocks start as identity)
        for blk in self.blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0.0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0.0)

        # Zero-out FiLM output layers (so FiLM starts as identity)
        for film_mod in self.film_modules.values():
            nn.init.constant_(film_mod.mlp[-1].weight, 0.0)
            nn.init.constant_(film_mod.mlp[-1].bias, 0.0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0.0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0.0)
        nn.init.constant_(self.final_layer.linear.weight, 0.0)
        nn.init.constant_(self.final_layer.linear.bias, 0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: (B, C, T) noisy waveform.
        t: (B,) diffusion timestep in [0, 1].
        cond_tokens: (B, n_cond_tokens, hidden_size) from SeismicConditionEncoder,
                     or null tokens for unconditional forward.

        Returns
        -------
        x_pred: (B, C, T) predicted clean waveform.
        """
        B = x.shape[0]

        # Timestep embedding → adaLN source
        t_emb = self.t_embedder(t)                    # (B, hidden)

        # Precompute FiLM (γ, β) per group from condition-token groups.
        film_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for group_name, film_mod in self.film_modules.items():
            idx = self.cond_token_groups[group_name]
            cond_group = cond_tokens[:, idx, :]
            film_cache[group_name] = film_mod(cond_group)

        # Patchify
        x = self.x_embedder(x)                        # (B, N_p, hidden)
        x = x + self.pos_embed

        # Prepend condition tokens at layer 0
        ct = cond_tokens + self.cond_pos_embed        # (B, N_c, hidden)
        x = torch.cat([ct, x], dim=1)                 # (B, N_c+N_p, hidden)

        # Determine FiLM group for each block
        block_to_group = {bi: gn for gn, bis in self.film_groups.items() for bi in bis}

        for i, blk in enumerate(self.blocks):
            group_name = block_to_group.get(i)
            if group_name is not None:
                gamma, beta = film_cache[group_name]
            else:
                gamma = beta = None
            x = blk(x, t_emb, rope=self.rope, film_gamma=gamma, film_beta=beta)

        # Discard condition tokens, keep patches only
        x = x[:, self.n_cond_tokens:]                 # (B, N_p, hidden)

        # Unpatchify
        return self.final_layer(x, t_emb)


# ---------------------------------------------------------------------------
# 1-D sinusoidal position embedding (fixed, not learnable)
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim: int, seq_len: int) -> "np.ndarray":
    pos = np.arange(seq_len, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000.0 ** (omega / (embed_dim / 2)))
    out = np.outer(pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    if embed_dim % 2:
        emb = np.concatenate([emb, np.zeros((seq_len, 1))], axis=1)
    return emb.astype(np.float32)
