"""Diffusion denoiser wrapping JsT with x-prediction + v-loss.

Follows the JiT flow-model formulation:
    z = t·x + (1-t)·ε          (interpolant between noise and data)
    v = (x - z) / (1 - t)      (velocity field)
    x̂ = net(z, t, cond)        (network predicts clean x)
    v̂ = (x̂ - z) / (1 - t)      (estimated velocity)
    loss = ||v - v̂||²

Training uses logit-normal timestep sampling (P_mean / P_std).
Inference uses ODE integration (Euler or Heun).
"""

from __future__ import annotations

import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import JsT


class Denoiser(nn.Module):
    def __init__(
        self,
        model: JsT,
        *,
        P_mean: float = -0.8,
        P_std: float = 0.8,
        noise_scale: float = 1.0,
        t_eps: float = 5e-2,
        cond_drop_prob: float = 0.15,
        # EMA
        ema_decay1: float = 0.9999,
        ema_decay2: float = 0.9996,
        # Sampling defaults
        sampling_method: Literal["euler", "heun"] = "heun",
        num_sampling_steps: int = 50,
        cfg_scale: float = 1.0,
        # z-amplitude jitter (for decoder info deprivation)
        z_amp_jitter: float = 0.0,
    ):
        super().__init__()
        self.net = model
        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.cond_drop_prob = cond_drop_prob

        # EMA
        self.ema_decay1 = ema_decay1
        self.ema_decay2 = ema_decay2
        self.ema_params1: list[torch.Tensor] | None = None
        self.ema_params2: list[torch.Tensor] | None = None

        # Sampling
        self.method = sampling_method
        self.steps = num_sampling_steps
        self.cfg_scale = cfg_scale
        self.z_amp_jitter = z_amp_jitter
        self.cond_token_groups = getattr(
            model,
            "cond_token_groups",
            {"source": [0], "path": [1], "receiver": [2]},
        )
        self._group_drop: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def sample_t(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)                    # (n,)

    # ------------------------------------------------------------------
    # Condition dropout (CFG) — per-token hard with warmup support
    # ------------------------------------------------------------------

    def set_cond_drop_prob(self, p: float) -> None:
        """Runtime override — call from training loop for phased CFG."""
        self.cond_drop_prob = p

    def set_group_drop_probs(
        self,
        source: float | None = None,
        path: float | None = None,
        receiver: float | None = None,
        **extra: float,
    ) -> None:
        """Per-semantic-group dropout probabilities."""
        probs = {}
        if source is not None:
            probs["source"] = source
        if path is not None:
            probs["path"] = path
        if receiver is not None:
            probs["receiver"] = receiver
        probs.update(extra)
        self._group_drop = probs

    def _sample_cond_drop(self, batch_size: int, n_tokens: int, device: torch.device) -> torch.Tensor | None:
        if not self.training or self.cond_drop_prob <= 0.0:
            return None
        if self._group_drop:
            drop = torch.zeros(batch_size, n_tokens, device=device, dtype=torch.bool)
            for group_name, prob in self._group_drop.items():
                idx = self.cond_token_groups.get(group_name)
                if not idx or prob <= 0.0:
                    continue
                group_drop = torch.rand(batch_size, device=device) < prob
                drop[:, idx] |= group_drop[:, None]
            return drop
        return torch.rand(batch_size, n_tokens, device=device) < self.cond_drop_prob

    def _maybe_drop_cond(self, cond_tokens: torch.Tensor, drop: torch.Tensor | None) -> torch.Tensor:
        if drop is None or not drop.any():
            return cond_tokens
        null = self.net.null_tokens.expand(cond_tokens.shape[0], -1, -1).to(cond_tokens.dtype)
        return torch.where(drop[:, :, None], null, cond_tokens)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        *,
        encoder_weight: float = 0.0,
        return_context: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        x: (B, C, T) clean waveform.
        cond_tokens: (B, n_cond, hidden) from SeismicConditionEncoder.

        Returns scalar v-loss.  If ``encoder_weight > 0``, returns
        ``(v_loss, encoder_loss)``.
        """
        drop = self._sample_cond_drop(cond_tokens.shape[0], cond_tokens.shape[1], cond_tokens.device)
        cond_tokens = self._maybe_drop_cond(cond_tokens, drop)

        # Sample t per batch element
        t = self.sample_t(x.size(0), device=x.device)       # (B,)
        t_3d = t[:, None, None]                              # (B, 1, 1)
        eps = torch.randn_like(x) * self.noise_scale

        z = t_3d * x + (1.0 - t_3d) * eps                     # interpolant
        v_target = (x - z) / (1.0 - t_3d).clamp_min(self.t_eps)

        # z-amplitude jitter: randomly rescale z so amplitude info is unreliable.
        # Decoder must consult source token for true amplitude.
        z_net = z
        if self.z_amp_jitter > 0.0:
            jitter = 1.0 + self.z_amp_jitter * torch.randn(x.size(0), 1, 1, device=x.device)
            z_net = z * jitter

        x_pred = self.net(z_net, t, cond_tokens)
        v_pred = (x_pred - z) / (1.0 - t_3d).clamp_min(self.t_eps)

        loss = (v_target - v_pred) ** 2

        v_loss = loss.mean()

        # Encoder separation loss — penalize when tokens are too close to null
        if encoder_weight > 0.0:
            src = cond_tokens  # (B, Nc, hidden)
            null = self.net.null_tokens.detach()  # (1, Nc, hidden)
            dist = (src - null).norm(dim=-1).mean()  # scalar
            target = 5.0  # minimum desired Euclidean distance from null
            enc_loss = F.relu(target - dist)
            if return_context:
                return v_loss, {
                    "encoder_loss": enc_loss,
                    "cond_tokens": cond_tokens,
                    "z": z,
                    "z_net": z_net,
                    "t": t,
                    "v_target": v_target,
                }
            return v_loss, enc_loss

        if return_context:
            return v_loss, {
                "cond_tokens": cond_tokens,
                "z": z,
                "z_net": z_net,
                "t": t,
                "v_target": v_target,
            }
        return v_loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        cond_tokens: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """ODE integration from noise to clean waveform."""
        device = cond_tokens.device
        bsz = cond_tokens.shape[0]
        C = self.net.in_channels
        T = self.net.n_samples

        n_steps = steps if steps is not None else self.steps
        noise = self.noise_scale * torch.randn(bsz, C, T, device=device)

        # Timesteps: (n_steps+1,) — scalar per step, same for whole batch
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        stepper = self._heun_step if self.method == "heun" else self._euler_step

        z = noise
        for i in range(n_steps - 1):
            z = stepper(z, ts[i], ts[i + 1], cond_tokens)
        z = self._euler_step(z, ts[-2], ts[-1], cond_tokens)
        return z

    # ------------------------------------------------------------------
    # ODE steps (with CFG)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_cfg(
        self,
        z: torch.Tensor,
        t_1d: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        t_3d = t_1d[:, None, None]
        x_cond = self.net(z, t_1d, cond_tokens)
        v_cond = (x_cond - z) / (1.0 - t_3d).clamp_min(self.t_eps)

        if self.cfg_scale == 1.0:
            return v_cond

        null = self.net.null_tokens.expand(z.shape[0], -1, -1).to(cond_tokens.dtype)
        x_uncond = self.net(z, t_1d, null)
        v_uncond = (x_uncond - z) / (1.0 - t_3d).clamp_min(self.t_eps)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        t_batch = t.expand(z.shape[0])                         # (B,)
        v = self._forward_cfg(z, t_batch, cond_tokens)
        return z + (t_next - t) * v

    @torch.no_grad()
    def _heun_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        t_batch = t.expand(z.shape[0])
        t_next_batch = t_next.expand(z.shape[0])
        v_t = self._forward_cfg(z, t_batch, cond_tokens)
        z_euler = z + (t_next - t) * v_t
        v_tn = self._forward_cfg(z_euler, t_next_batch, cond_tokens)
        return z + (t_next - t) * 0.5 * (v_t + v_tn)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_ema(self) -> None:
        self.ema_params1 = copy.deepcopy(list(self.parameters()))
        self.ema_params2 = copy.deepcopy(list(self.parameters()))

    @torch.no_grad()
    def update_ema(self) -> None:
        if self.ema_params1 is None:
            return
        for i, (targ, src) in enumerate(zip(self.ema_params1, self.parameters())):
            if targ.device != src.device:
                self.ema_params1[i] = targ.to(src.device)
                targ = self.ema_params1[i]
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1.0 - self.ema_decay1)
        for i, (targ, src) in enumerate(zip(self.ema_params2, self.parameters())):
            if targ.device != src.device:
                self.ema_params2[i] = targ.to(src.device)
                targ = self.ema_params2[i]
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1.0 - self.ema_decay2)

    @torch.no_grad()
    def apply_ema(self, which: int = 1) -> dict[str, torch.Tensor]:
        """Return a state dict with EMA weights (for evaluation)."""
        params = self.ema_params1 if which == 1 else self.ema_params2
        if params is None:
            raise RuntimeError("EMA not initialized. Call init_ema() first.")
        sd = self.state_dict()
        ema = {}
        for i, (name, _) in enumerate(self.named_parameters()):
            ema[name] = params[i].detach()
        return ema
