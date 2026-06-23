"""JsT unconditional waveform inpainting via diffusion completion.

Given the first K patches before the P-wave arrival, JsT completes the
remaining waveform without any condition tokens (null-token CFG forward).

Algorithm (flow-matching inpainting):
  1. Start with initial noise z_0
  2. For known region only: z_known(t) = t * x_known + (1-t) * z_0_known
     (deterministic formula — no ODE needed for known region)
  3. For unknown region: standard ODE integration with null tokens
  4. At each step: z = mask * z_known + (1-mask) * z_ode
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    JsT,
    SeismicConditionEncoder,
    ConditionSpec,
    Denoiser,
    SeismicWaveformDataset,
    collate_conditions,
    load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


@torch.no_grad()
def inpaint_waveform(
    denoiser: Denoiser,
    x_known: torch.Tensor,              # (B, C, T_known) known pre-P waveform segment
    *,
    total_samples: int = 3200,
    ode_steps: int = 50,
    method: str = "heun",
    noise_scale: float = 1.0,
    t_eps: float = 5e-2,
    blend_patches: int = 0,             # 0 = hard mask (best prefix fidelity)
    resample_noise: bool = False,       # off by default; hurts flow-matching inpainting
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Complete a waveform from a known prefix (unconditional).

    Parameters
    ----------
    denoiser: trained Denoiser with JsT net
    x_known: (B, C, T_known) known waveform prefix
    total_samples: total waveform length
    ode_steps: ODE integration steps
    method: "heun" or "euler"
    noise_scale: noise std (must match denoiser.noise_scale)
    t_eps: minimum time
    blend_patches: number of patches for soft boundary transition
    resample_noise: add timestep-appropriate noise to known region (RePaint)
    seed: random seed
    device: target device
    verbose: print progress

    Returns
    -------
    x_completed: (B, C, total_samples) completed waveform
    """
    if device is None:
        device = x_known.device

    B, C, T_known = x_known.shape
    T_unknown = total_samples - T_known
    net = denoiser.net
    null_tokens = net.null_tokens.expand(B, -1, -1).to(device=device, dtype=x_known.dtype)

    # ---- 1. Build SOFT mask with transition zone ----
    # Hard known region: [0, T_known - blend_samples)
    # Transition zone:  [T_known - blend_samples, T_known + blend_samples)
    # Hard unknown:      [T_known + blend_samples, total_samples)
    blend_samples = blend_patches * 64  # 3 patches = 192 samples = 4.8s at 40Hz
    blend_start = max(0, T_known - blend_samples)
    blend_end = min(total_samples, T_known + blend_samples)

    mask = torch.zeros(B, C, total_samples, device=device, dtype=x_known.dtype)
    # Hard known: mask = 1.0
    mask[:, :, :blend_start] = 1.0
    # Transition: linear ramp from 1 → 0
    for pos in range(blend_start, blend_end):
        alpha = 1.0 - (pos - blend_start) / (blend_end - blend_start)
        mask[:, :, pos] = alpha
    # Hard unknown: mask = 0.0 (already zero)

    # ---- 2. Initial noise ----
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z_0 = noise_scale * torch.randn(
        B, C, total_samples, device=device, dtype=x_known.dtype, generator=generator
    )

    x_full = torch.zeros(B, C, total_samples, device=device, dtype=x_known.dtype)
    x_full[:, :, :T_known] = x_known

    # ---- 3. ODE integration ----
    ts = torch.linspace(0.0, 1.0, ode_steps + 1, device=device)
    z = z_0.clone()

    stepper = _heun_inpaint_step if method == "heun" else _euler_inpaint_step

    for i in range(ode_steps - 1):
        z = stepper(z, ts[i], ts[i + 1], x_full, mask, z_0, null_tokens,
                     net, t_eps, noise_scale, resample_noise, generator)
        if verbose and (i + 1) % 10 == 0:
            print(f"  ODE step {i+1}/{ode_steps}")

    z = _euler_inpaint_step(z, ts[-2], ts[-1], x_full, mask, z_0, null_tokens,
                             net, t_eps, noise_scale, resample_noise, generator)
    return z


@torch.no_grad()
def _euler_inpaint_step(
    z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor,
    x_full: torch.Tensor, mask: torch.Tensor, z_0: torch.Tensor,
    null_tokens: torch.Tensor, net: nn.Module, t_eps: float,
    noise_scale: float, resample_noise: bool, generator: torch.Generator,
) -> torch.Tensor:
    """Euler step with soft mask and RePaint-style known-region resampling."""
    B, C, T = z.shape
    t_batch = t.expand(B)
    t_next_3d = torch.full((B, 1, 1), t_next, device=z.device, dtype=z.dtype)
    t_3d = t_batch[:, None, None]

    # Step 1: ODE forward for the whole waveform
    x_pred = net(z, t_batch, null_tokens)
    v_pred = (x_pred - z) / (1.0 - t_3d).clamp_min(t_eps)
    dt = t_next - t
    z_ode = z + dt * v_pred

    # Step 2: Exact flow for known region
    z_known_exact = t_next_3d * x_full + (1.0 - t_next_3d) * z_0

    # Step 3: RePaint resampling — add timestep-appropriate noise to known region
    # so the model sees consistent noise level across the boundary
    if resample_noise and t_next > t_eps:
        sigma_t = (1.0 - t_next) * noise_scale  # noise level at this timestep
        noise = torch.randn(B, C, T, device=z.device, dtype=z.dtype, generator=generator)
        z_known_resampled = z_known_exact + sigma_t * noise
    else:
        z_known_resampled = z_known_exact

    # Step 4: Soft-mask blend between known and ODE
    z_next = mask * z_known_resampled + (1.0 - mask) * z_ode

    return z_next


@torch.no_grad()
def _heun_inpaint_step(
    z: torch.Tensor, t: torch.Tensor, t_next: torch.Tensor,
    x_full: torch.Tensor, mask: torch.Tensor, z_0: torch.Tensor,
    null_tokens: torch.Tensor, net: nn.Module, t_eps: float,
    noise_scale: float, resample_noise: bool, generator: torch.Generator,
) -> torch.Tensor:
    """Heun step with soft mask and RePaint resampling."""
    B, C, T = z.shape
    t_batch = t.expand(B)
    t_next_batch = t_next.expand(B)
    t_3d = t_batch[:, None, None]
    t_next_3d = torch.full((B, 1, 1), t_next, device=z.device, dtype=z.dtype)
    dt = t_next - t

    # Predictor
    x_pred = net(z, t_batch, null_tokens)
    v_t = (x_pred - z) / (1.0 - t_3d).clamp_min(t_eps)
    z_euler_ode = z + dt * v_t

    # Blend predictor with known region for corrector input
    z_known_exact_predictor = t_next_3d * x_full + (1.0 - t_next_3d) * z_0
    if resample_noise and t_next > t_eps:
        sigma_t = (1.0 - t_next) * noise_scale
        noise = torch.randn(B, C, T, device=z.device, dtype=z.dtype, generator=generator)
        z_known_resampled = z_known_exact_predictor + sigma_t * noise
    else:
        z_known_resampled = z_known_exact_predictor
    z_euler = mask * z_known_resampled + (1.0 - mask) * z_euler_ode

    # Corrector
    x_pred_tn = net(z_euler, t_next_batch, null_tokens)
    v_tn = (x_pred_tn - z_euler) / (1.0 - t_next_3d).clamp_min(t_eps)
    z_heun_ode = z + dt * 0.5 * (v_t + v_tn)

    # Final blend
    z_known_exact_final = t_next_3d * x_full + (1.0 - t_next_3d) * z_0
    if resample_noise and t_next > t_eps:
        sigma_t = (1.0 - t_next) * noise_scale
        noise = torch.randn(B, C, T, device=z.device, dtype=z.dtype, generator=generator)
        z_known_resampled_final = z_known_exact_final + sigma_t * noise
    else:
        z_known_resampled_final = z_known_exact_final
    z_next = mask * z_known_resampled_final + (1.0 - mask) * z_heun_ode

    return z_next


def compute_inpaint_metrics(
    x_completed: np.ndarray,
    x_ground_truth: np.ndarray,
    prefix_samples: int,
    sample_rate_hz: float = 40.0,
) -> dict:
    """Evaluate inpainting quality.

    Parameters
    ----------
    x_completed: (N, 3, T) completed waveforms
    x_ground_truth: (N, 3, T) ground truth waveforms
    prefix_samples: number of known samples (kept from truth)

    Returns dict with time-domain and spectral quality metrics, separately
    for the completed region and the full waveform.
    """
    T = x_completed.shape[2]
    # Prefix region: match is exact (forced by inpainting)
    # Completed region: match is measured
    unknown_mask = np.zeros(T)
    unknown_mask[prefix_samples:] = 1.0

    metrics = {"prefix_samples": prefix_samples, "total_samples": T}

    for region, mask_arr in [("completed", unknown_mask), ("full", np.ones(T))]:
        label = region

        # Time-domain L2
        diff = (x_completed - x_ground_truth) * mask_arr[None, None, :]
        ref = x_ground_truth * mask_arr[None, None, :]

        l2_norm = np.sqrt(np.sum(diff ** 2)) / max(np.sqrt(np.sum(ref ** 2)), 1e-12)
        metrics[f"{label}_l2"] = float(l2_norm)

        # Per-channel analysis
        for ch, ch_name in enumerate(["Z", "N", "E"]):
            diff_ch = (x_completed[:, ch, :] - x_ground_truth[:, ch, :]) * mask_arr[None, :]
            ref_ch = x_ground_truth[:, ch, :] * mask_arr[None, :]
            ch_l2 = np.sqrt(np.sum(diff_ch ** 2)) / max(np.sqrt(np.sum(ref_ch ** 2)), 1e-12)
            metrics[f"{label}_l2_{ch_name}"] = float(ch_l2)

        # Peak amplitude ratio
        for ch, ch_name in enumerate(["Z", "N", "E"]):
            completed_peak = np.max(np.abs(x_completed[:, ch, prefix_samples:]))
            truth_peak = np.max(np.abs(x_ground_truth[:, ch, prefix_samples:]))
            ratio = completed_peak / truth_peak if truth_peak > 1e-8 else 1.0
            metrics[f"{label}_peak_ratio_{ch_name}"] = float(ratio)

        # Spectral similarity (correlation of amplitude spectra)
        for ch, ch_name in enumerate(["Z", "N", "E"]):
            spec_completed = np.abs(np.fft.rfft(x_completed[:, ch, :]))
            spec_truth = np.abs(np.fft.rfft(x_ground_truth[:, ch, :]))
            corr = np.corrcoef(spec_completed.flatten(), spec_truth.flatten())[0, 1]
            metrics[f"{label}_spectral_corr_{ch_name}"] = float(corr)

    # Continuity at boundary: should be smooth (no jump at prefix_samples)
    for ch, ch_name in enumerate(["Z", "N", "E"]):
        # Difference across boundary
        left = x_completed[:, ch, max(0, prefix_samples - 5):prefix_samples]
        right = x_completed[:, ch, prefix_samples:prefix_samples + 5]
        if left.shape[1] > 0 and right.shape[1] > 0:
            boundary_jump = np.abs(np.mean(left[:, -1]) - np.mean(right[:, 0]))
            true_jump = np.abs(
                np.mean(x_ground_truth[:, ch, max(0, prefix_samples - 5):prefix_samples][:, -1])
                - np.mean(x_ground_truth[:, ch, prefix_samples:prefix_samples + 5][:, 0])
            )
            metrics[f"boundary_jump_{ch_name}"] = float(boundary_jump)
            metrics[f"boundary_jump_truth_{ch_name}"] = float(true_jump)

    return metrics


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    use_ema: bool = True,
    drop_tokens: list[int] | None = None,
) -> tuple[SeismicConditionEncoder, Denoiser, dict]:
    """Load trained JsT model with optional token ablation."""
    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=use_ema,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    return ce, dn, ckpt


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="JsT unconditional waveform inpainting")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--cache-prefix", default="pwave_v21")
    parser.add_argument("--output-dir", default="outputs/inpaint")
    parser.add_argument("--split", default="testing")
    parser.add_argument("--n-samples", type=int, default=16,
                        help="Number of test waveforms to inpaint")
    parser.add_argument("--prefix-patches", type=int, default=12,
                        help="Number of known patches before P-wave (12 = ~19s)")
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--blend-patches", type=int, default=0,
                        help="Soft transition patches (0=hard mask, best for prefix fidelity)")
    parser.add_argument("--resample", dest="resample_noise", action="store_true", default=False,
                        help="Enable RePaint-style resampling noise (usually hurts flow-matching)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", default="inpaint")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ce, dn, ckpt = load_model(args.checkpoint, device, use_ema=args.use_ema, drop_tokens=dropped)
    dn.eval()
    ce.eval()

    # Load data
    ds_train = SeismicWaveformDataset(
        args.data_dir, split="training", augment=False,
        cache_prefix=args.cache_prefix, condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        args.data_dir, split=args.split, augment=False, vocab_from=ds_train,
        cache_prefix=args.cache_prefix, condition_version="v2.1", field_policy="default",
    )

    prefix_samples = args.prefix_patches * 64  # each patch is 64 samples
    total_samples = 3200
    n_test = min(args.n_samples, len(ds_test))
    print(f"Prefix: {args.prefix_patches} patches = {prefix_samples} samples = {prefix_samples / 40:.1f}s")
    print(f"Total: {total_samples} samples = 80s")
    print(f"Unknown: {total_samples - prefix_samples} samples = {(total_samples - prefix_samples) / 40:.1f}s")
    print(f"Test samples: {n_test}")

    torch.manual_seed(args.seed)
    indices = torch.randperm(len(ds_test))[:n_test].tolist()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    all_waveforms_completed = []
    all_waveforms_truth = []

    print(f"\n=== Inpainting {n_test} waveforms ===")
    t0 = time.time()

    for idx, cache_idx in enumerate(indices):
        print(f"  [{idx+1}/{n_test}] cache_idx={cache_idx}")
        waveform, cond = ds_test[cache_idx]
        waveform = waveform.unsqueeze(0).to(device)  # (1, 3, 3200)

        # Extract known prefix
        x_known = waveform[:, :, :prefix_samples]

        # Inpaint
        completed = inpaint_waveform(
            dn, x_known,
            total_samples=total_samples,
            ode_steps=args.ode_steps,
            method=args.method,
            noise_scale=dn.noise_scale,
            blend_patches=args.blend_patches,
            resample_noise=args.resample_noise,
            seed=args.seed + idx,
            device=device,
            verbose=False,
        )

        all_waveforms_completed.append(completed.cpu().float().numpy())
        all_waveforms_truth.append(waveform.cpu().float().numpy())

    elapsed = time.time() - t0
    print(f"\nInpainted {n_test} waveforms in {elapsed:.1f}s ({elapsed / n_test:.1f}s/sample)")

    # Concatenate
    completed_arr = np.concatenate(all_waveforms_completed, axis=0)
    truth_arr = np.concatenate(all_waveforms_truth, axis=0)

    # Evaluate
    print("\n=== Evaluation ===")
    metrics = compute_inpaint_metrics(completed_arr, truth_arr, prefix_samples)
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"  {k:35s} = {v:.4f}")
        else:
            print(f"  {k:35s} = {v}")

    # Save
    np.savez_compressed(output_dir / f"{args.prefix}_waveforms.npz",
                        completed=completed_arr, truth=truth_arr)
    with open(output_dir / f"{args.prefix}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved to {output_dir}/")
    print("Done.")
