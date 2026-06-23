"""ChordEdit for JsT — training-free, inversion-free conditional waveform editing.

ChordEdit (CVPR 2026) adapted from image diffusion to seismic flow-matching.

Core algorithm:
  1. Start with a generated waveform x (from source condition c_src)
  2. At edit time t, add noise: z_t = t·x + (1-t)·ε
  3. Predict x̂ under BOTH c_src and c_tar at times t and t-δ
  4. Compute Chord control field: û = weighted average of condition-velocity differences
  5. Transport: x_edit = x + λ·û
  6. Optional clean-up: one denoising step at t_end

JST adaptation:
  - JsT is v-prediction (flow-matching): uses interpolant z = t·x + (1-t)·ε
  - No VAE — works in pixel/waveform space
  - Condition is dict[str, Tensor], not text embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


@torch.no_grad()
def chord_edit(
    denoiser,
    cond_encoder,
    x: torch.Tensor,          # (B, C, T) source waveform
    cond_src: dict,           # source condition dict
    cond_tar: dict,           # target condition dict
    *,
    t_start: float = 0.6,     # editing time (in [0,1])
    t_delta: float = 0.1,     # time window for Chord averaging
    step_scale: float = 1.0,  # step multiplier
    noise_samples: int = 1,   # number of noise samples to average over
    cleanup: bool = True,     # proximal refinement step
    t_cleanup: float = 0.3,   # cleanup denoising time
    seed: int = 42,
) -> torch.Tensor:
    """
    Edit waveform condition from cond_src → cond_tar using ChordEdit.

    Parameters
    ----------
    denoiser: JsT Denoiser module
    cond_encoder: SeismicConditionEncoder
    x: source waveform (B, 3, T), e.g. from denoiser.generate(cond_src)
    cond_src, cond_tar: condition dicts
    t_start: time to start editing (higher = more noise / more flexibility)
    t_delta: Chord averaging window width
    step_scale: step multiplier (1.0 = default Chord transport)
    noise_samples: number of independent noises to average over (1-8)
    cleanup: whether to apply proximal refinement (NFE +1)
    t_cleanup: cleanup time
    seed: random seed

    Returns
    -------
    x_edit: edited waveform (B, 3, T)
    """
    device = x.device
    B, C, T = x.shape
    dtype = x.dtype
    net = denoiser.net
    t_eps = denoiser.t_eps

    # Encode conditions
    ct_src = cond_encoder(cond_src)
    ct_tar = cond_encoder(cond_tar)

    # Prepare noises
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    noises = [torch.randn(B, C, T, device=device, dtype=dtype) for _ in range(noise_samples)]

    # Ensure t_start > t_delta
    if t_start <= t_delta:
        t_delta = t_start / 2.0

    t_start_t = torch.full((B,), t_start, device=device, dtype=dtype)
    t_prev_t = torch.full((B,), max(t_start - t_delta, t_eps), device=device, dtype=dtype)

    # Compute Chord control field
    # For each noise, compute x_pred under both conditions at both times
    dv_t_sum = torch.zeros_like(x)
    dv_t0_sum = torch.zeros_like(x)

    for eps in noises:
        # Interpolants
        z_t = t_start_t[:, None, None] * x + (1.0 - t_start_t[:, None, None]) * eps
        z_t0 = t_prev_t[:, None, None] * x + (1.0 - t_prev_t[:, None, None]) * eps

        # Predictions under source and target at time t
        x_pred_src_t = net(z_t, t_start_t, ct_src)
        x_pred_tar_t = net(z_t, t_start_t, ct_tar)

        # Predictions at time t-δ
        x_pred_src_t0 = net(z_t0, t_prev_t, ct_src)
        x_pred_tar_t0 = net(z_t0, t_prev_t, ct_tar)

        dv_t_sum += (x_pred_tar_t - x_pred_src_t)
        dv_t0_sum += (x_pred_tar_t0 - x_pred_src_t0)

    dv_t = dv_t_sum / noise_samples  # velocity difference at time t
    dv_t0 = dv_t0_sum / noise_samples  # velocity difference at time t-δ

    # Chord control field: weighted average
    denom = t_start + t_delta
    if denom > 0:
        u_hat = (t_delta * dv_t + t_start * dv_t0) / denom
    else:
        u_hat = dv_t

    # Transport
    x_edit = x + step_scale * u_hat

    # Optional proximal refinement
    if cleanup:
        t_end_t = torch.full((B,), t_cleanup, device=device, dtype=dtype)
        # Add noise at cleanup time
        torch.manual_seed(seed + 9999)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + 9999)
        eps_clean = torch.randn(B, C, T, device=device, dtype=dtype)
        z_clean = t_end_t[:, None, None] * x_edit + (1.0 - t_end_t[:, None, None]) * eps_clean
        x_edit = net(z_clean, t_end_t, ct_tar)

    return x_edit


@torch.no_grad()
def chord_edit_sweep(
    denoiser,
    cond_encoder,
    x: torch.Tensor,
    cond_src: dict,
    cond_tar: dict,
    *,
    param_grid: Optional[dict] = None,
    seed: int = 42,
) -> dict:
    """
    Sweep ChordEdit parameters and return results.

    Parameters
    ----------
    param_grid: dict of param_name -> list of values to test
        Default: {'t_start': [0.3, 0.5, 0.7], 't_delta': [0.05, 0.1, 0.2],
                  'step_scale': [0.5, 1.0, 2.0], 'noise_samples': [1, 4]}
    """
    if param_grid is None:
        param_grid = {
            "t_start": [0.4, 0.5, 0.6, 0.7],
            "t_delta": [0.05, 0.1, 0.2],
            "step_scale": [0.5, 0.75, 1.0, 1.5],
            "noise_samples": [1, 4],
        }

    results = {}
    total = 1
    for v in param_grid.values():
        total *= len(v)

    i = 0
    for t_start in param_grid.get("t_start", [0.6]):
        for t_delta in param_grid.get("t_delta", [0.1]):
            for step_scale in param_grid.get("step_scale", [1.0]):
                for noise_samples in param_grid.get("noise_samples", [1]):
                    for cleanup in param_grid.get("cleanup", [True, False]):
                        key = f"ts={t_start:.2f}_td={t_delta:.2f}_ss={step_scale:.2f}_ns={noise_samples}_cl={cleanup}"
                        try:
                            x_edit = chord_edit(
                                denoiser, cond_encoder, x,
                                cond_src, cond_tar,
                                t_start=t_start, t_delta=t_delta,
                                step_scale=step_scale,
                                noise_samples=noise_samples,
                                cleanup=cleanup,
                                seed=seed,
                            )
                            # Compute quality metrics
                            gp = x_edit.abs().max(dim=-1).values.max(dim=-1).values.mean()
                            gp_std = x_edit.abs().max(dim=-1).values.max(dim=-1).values.std()
                            energy = (x_edit ** 2).mean().sqrt()
                            results[key] = {
                                "x": x_edit.cpu(), "gen_peak": float(gp),
                                "gp_std": float(gp_std), "energy": float(energy),
                            }
                        except Exception as e:
                            results[key] = {"error": str(e)}
                        i += 1
    return results


def compute_edit_metrics(
    x_src: torch.Tensor,
    x_edit: torch.Tensor,
    x_tar_full: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute edit quality metrics.

    Parameters
    ----------
    x_src: source waveform (generated from cond_src)
    x_edit: edited waveform
    x_tar_full: optional, target waveform (generated from cond_tar by full regeneration)

    Returns dict with:
        - gen_peak_ratio: max amplitude of edit vs source
        - edit_magnitude: L2 norm of change
        - time_rel_l2: time-domain L2 vs source
        - preservation: 1 - edit_magnitude normalized
    """
    metrics = {}

    # Peak change
    src_peak = x_src.abs().max(dim=-1).values.max(dim=-1).values.mean()
    edit_peak = x_edit.abs().max(dim=-1).values.max(dim=-1).values.mean()
    metrics["gen_peak_src"] = float(src_peak)
    metrics["gen_peak_edit"] = float(edit_peak)
    metrics["peak_ratio"] = float(edit_peak / src_peak) if src_peak > 0 else 1.0

    # Edit magnitude
    diff = x_edit - x_src
    metrics["edit_l2"] = float((diff ** 2).mean().sqrt())
    src_l2 = float((x_src ** 2).mean().sqrt())
    metrics["relative_l2_change"] = metrics["edit_l2"] / src_l2 if src_l2 > 0 else 0

    # If full regeneration target available, compare
    if x_tar_full is not None:
        tar_l2 = float((x_tar_full ** 2).mean().sqrt())
        edit_tar_diff = x_edit - x_tar_full
        metrics["edit_vs_tar_l2"] = float((edit_tar_diff ** 2).mean().sqrt()) / tar_l2 if tar_l2 > 0 else 0

    return metrics
