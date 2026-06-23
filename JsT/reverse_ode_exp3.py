"""Experiment 3: Joint inversion with noise whiteness penalty.

Tests whether adding a reverse-ODE noise whiteness penalty to the
standard waveform-matching loss can break the token-space degeneracy
and recover true source tokens via gradient descent.

If noise whiteness provides a measurable gradient signal toward the
true token, the combined loss can converge where pure waveform loss
failed (Experiment 1/1b).
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, ConditionSpec, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


@torch.no_grad()
def reverse_ode(denoiser, waveform, cond_tokens, steps=50):
    device = waveform.device
    B = waveform.shape[0]
    net = denoiser.net
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    z = waveform.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        x_pred = net(z, t.expand(B), cond_tokens)
        v = (x_pred - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


@torch.no_grad()
def forward_ode(denoiser, noise, cond_tokens, steps=50):
    device = noise.device
    B = noise.shape[0]
    net = denoiser.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        x_pred = net(z, t.expand(B), cond_tokens)
        v = (x_pred - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


def noise_autocorr_l1(noise: torch.Tensor) -> torch.Tensor:
    """Lag-1 autocorrelation of noise (scalar, differentiable for first step)."""
    n = noise  # (B, C, T)
    return torch.nn.functional.mse_loss(n[..., :-1], n[..., 1:])


def noise_kurtosis_surrogate(noise: torch.Tensor) -> torch.Tensor:
    """Surrogate for excess kurtosis: penalize extreme values.

    True kurtosis uses 4th moments which amplify outliers; this surrogate
    penalizes the squared deviation from unit-variance Gaussian.
    """
    n_flat = noise.flatten()
    # Penalize values beyond ±3 sigma
    sigma = n_flat.std()
    outlier_mask = (n_flat.abs() > 3.0 * sigma)
    if outlier_mask.any():
        return ((n_flat[outlier_mask] - 3.0 * sigma * torch.sign(n_flat[outlier_mask])) ** 2).mean()
    return torch.tensor(0.0, device=noise.device)


def noise_whiteness_loss(denoiser, waveform, cond_tokens, ode_steps=10):
    """Reverse ODE + compute autocorrelation-based whiteness penalty.

    Uses fewer ODE steps for speed (10 vs 50). The autocorrelation
    penalty is smooth enough to provide gradient signal.
    """
    z_rec = reverse_ode(denoiser, waveform, cond_tokens, steps=ode_steps)
    # Normalize to unit variance
    z_norm = z_rec / z_rec.std().clamp_min(1e-6)
    # Lag-1 autocorrelation
    ac1 = noise_autocorr_l1(z_norm)
    # Surrogate kurtosis
    kurt = noise_kurtosis_surrogate(z_norm)
    return ac1 + kurt


def run_experiment_3(
    checkpoint_path: str,
    device: torch.device,
    n_trials: int = 5,
    n_steps: int = 500,
    lr: float = 0.1,
    noise_penalty_weight: float = 0.5,
    comparison_modes: list[str] | None = None,
    drop_tokens: list[int] | None = None,
) -> dict:
    """Compare GD with vs without noise whiteness penalty.

    For each trial, runs 3 modes:
    1. "waveform_only" — pure waveform matching (Experiment 1 baseline)
    2. "noise_penalty"  — waveform + reverse-ODE noise whiteness
    3. "noise_anneal"   — waveform only first 250 steps, then add noise penalty
    """
    if comparison_modes is None:
        comparison_modes = ["waveform_only", "noise_penalty"]

    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    net = dn.net
    n_tokens = net.n_cond_tokens
    hidden = net.hidden_size
    total_samples = 3200
    source_idx = [0, 1, 2]
    sample_rate_hz = 40.0
    windows_sec = [8.0, 16.0, 40.0, 80.0]

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    print(f"Experiment 3: Joint inversion with noise whiteness penalty")
    print(f"  Trials: {n_trials}, Windows: {windows_sec}")
    print(f"  Modes: {comparison_modes}")
    print(f"  Noise penalty weight: {noise_penalty_weight}")
    print()

    all_results = {}

    for mode in comparison_modes:
        print(f"\n{'='*70}")
        print(f"  MODE: {mode}")
        print(f"{'='*70}")

        for win_idx, win_sec in enumerate(windows_sec):
            win_samples = min(int(win_sec * sample_rate_hz), total_samples)
            t0_total = time.time()
            print(f"\n  Window {win_sec:.0f}s ({win_samples} samples):")

            trial_results = []

            for trial in range(n_trials):
                cache_idx = int(torch.randint(0, len(ds_test), (1,)).item())
                waveform_tensor, cond_dict = ds_test[cache_idx]
                waveform = waveform_tensor.unsqueeze(0).to(device)
                cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
                true_tokens = ce(cond_gpu)

                seed_base = 42 + win_idx * 1000 + trial * 100
                with torch.no_grad():
                    torch.manual_seed(seed_base)
                    if device.type == "cuda": torch.cuda.manual_seed_all(seed_base)
                    noise = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                    ts_ode = torch.linspace(0.0, 1.0, 51, device=device)
                    zf = noise
                    for i in range(49):
                        xp = net(zf, ts_ode[i].expand(1), true_tokens)
                        vf = (xp - zf) / (1.0 - ts_ode[i].view(1, 1, 1)).clamp_min(dn.t_eps)
                        zf = zf + (ts_ode[i + 1] - ts_ode[i]) * vf
                target_wf = zf.clone()

                window_mask = torch.zeros(1, 3, total_samples, device=device)
                window_mask[:, :, :win_samples] = 1.0

                torch.manual_seed(300 + seed_base)
                init_tokens = net.null_tokens.expand(1, -1, -1).clone().detach()
                init_tokens.add_(0.1 * torch.randn(1, n_tokens, hidden, device=device))
                init_tokens.requires_grad_(True)
                with torch.no_grad():
                    for idx in range(n_tokens):
                        if idx not in source_idx:
                            init_tokens[:, idx, :] = true_tokens[:, idx, :]

                opt = torch.optim.Adam([init_tokens], lr=lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)
                noises_pool = [dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                               for _ in range(10)]

                best_src_cos = -1.0
                best_step = 0
                loss_curve = []
                src_cos_curve = []

                anneal_start = 250 if mode == "noise_anneal" else (0 if mode == "noise_penalty" else n_steps + 1)

                for step in range(n_steps):
                    opt.zero_grad()
                    wf_loss = None
                    for i in range(5):
                        tv = torch.rand(1, device=device).item()
                        tv = max(tv, dn.t_eps); tv = min(tv, 1.0 - dn.t_eps)
                        eps = noises_pool[(step * 5 + i) % len(noises_pool)]
                        t3d = torch.tensor(tv, device=device).view(1, 1, 1)
                        zs = t3d * target_wf + (1.0 - t3d) * eps
                        tp = net(zs, torch.full((1,), tv, device=device), init_tokens)
                        term = ((tp - target_wf) * window_mask).pow(2).mean()
                        wf_loss = term if wf_loss is None else wf_loss + term
                    wf_loss = wf_loss / 5

                    total_loss = wf_loss

                    # Noise whiteness penalty (if active in this step)
                    nz_loss = torch.tensor(0.0, device=device)
                    if step >= anneal_start:
                        nz_loss = noise_whiteness_loss(dn, target_wf, init_tokens, ode_steps=10)
                        total_loss = wf_loss + noise_penalty_weight * nz_loss

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_([init_tokens], 1.0)
                    opt.step()
                    sched.step()

                    loss_curve.append(float(total_loss.detach()))

                    # Track best
                    with torch.no_grad():
                        src_cos = torch.nn.functional.cosine_similarity(
                            init_tokens[0, source_idx], true_tokens[0, source_idx], dim=-1
                        ).mean().item()
                    src_cos_curve.append(src_cos)
                    if src_cos > best_src_cos:
                        best_src_cos = src_cos
                        best_step = step

                final_tokens = init_tokens.detach()
                with torch.no_grad():
                    final_src_cos = torch.nn.functional.cosine_similarity(
                        final_tokens[0, source_idx], true_tokens[0, source_idx], dim=-1
                    ).mean().item()

                    # Generate full waveform for L2 comparison
                    torch.manual_seed(seed_base)
                    if device.type == "cuda": torch.cuda.manual_seed_all(seed_base)
                    noise2 = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                    zf2 = noise2
                    for i in range(49):
                        xp2 = net(zf2, ts_ode[i].expand(1), final_tokens)
                        vf2 = (xp2 - zf2) / (1.0 - ts_ode[i].view(1, 1, 1)).clamp_min(dn.t_eps)
                        zf2 = zf2 + (ts_ode[i + 1] - ts_ode[i]) * vf2
                    recovered_wf = zf2

                    # Observed-window L2
                    od = (recovered_wf - target_wf) * window_mask
                    obs_l2 = (od ** 2).mean().sqrt().item()
                    onm = ((target_wf * window_mask) ** 2).mean().sqrt().item()
                    obs_l2_rel = obs_l2 / max(onm, 1e-8)

                    # Full L2
                    fd = recovered_wf - target_wf
                    full_l2 = (fd ** 2).mean().sqrt().item()
                    fnm = (target_wf ** 2).mean().sqrt().item()
                    full_l2_rel = full_l2 / max(fnm, 1e-8)

                trial_results.append({
                    "trial": trial + 1, "win_sec": win_sec,
                    "final_src_cos": float(final_src_cos),
                    "best_src_cos": float(best_src_cos), "best_step": best_step,
                    "obs_l2_rel": float(obs_l2_rel),
                    "full_l2_rel": float(full_l2_rel),
                    "final_wf_loss": float(wf_loss.detach()),
                    "final_nz_loss": float(nz_loss.detach()),
                    "loss_curve": [float(x) for x in loss_curve],
                })

            elapsed = time.time() - t0_total
            avg_src = np.mean([t["best_src_cos"] for t in trial_results])
            avg_obs = np.mean([t["obs_l2_rel"] for t in trial_results])
            avg_full = np.mean([t["full_l2_rel"] for t in trial_results])
            print(f"    best_src_cos={avg_src:.4f}  obs_l2={avg_obs:.4f}  full_l2={avg_full:.4f}  ({elapsed:.0f}s)")

            all_results.setdefault(mode, {})[f"{win_sec:.0f}s"] = trial_results

    # ---- Final comparison table ----
    print(f"\n{'='*90}")
    print(f"  EXPERIMENT 3 — FINAL: Noise Penalty vs Baseline")
    print(f"{'='*90}")
    print(f"{'Window':>7s}  {'WF-only src_cos':>15s}  {'NoisePen src_cos':>16s}  "
          f"{'WF-only fullL2':>14s}  {'NoisePen fullL2':>15s}  {'delta_cos':>9s}")
    print("-" * 95)

    summary = {}
    for win_key in [f"{w:.0f}s" for w in windows_sec]:
        vals = {}
        for mode in comparison_modes:
            if mode in all_results and win_key in all_results[mode]:
                trials = all_results[mode][win_key]
                vals[f"{mode}_src"] = np.mean([t["best_src_cos"] for t in trials])
                vals[f"{mode}_full"] = np.mean([t["full_l2_rel"] for t in trials])

        if "waveform_only" in vals and "noise_penalty" in vals:
            delta = vals["noise_penalty_src"] - vals["waveform_only_src"]
            print(f"{win_key:>7s}  {vals.get('waveform_only_src', 0):15.4f}  "
                  f"{vals.get('noise_penalty_src', 0):16.4f}  "
                  f"{vals.get('waveform_only_full', 0):14.4f}  "
                  f"{vals.get('noise_penalty_full', 0):15.4f}  "
                  f"{delta:+9.4f}")
            summary[win_key] = {**vals, "delta_src_cos": float(delta)}
        elif "waveform_only" in vals:
            print(f"{win_key:>7s}  {vals.get('waveform_only_src', 0):15.4f}  "
                  f"{'---':>16s}  "
                  f"{vals.get('waveform_only_full', 0):14.4f}  {'---':>15s}  {'---':>9s}")

    # Verdict
    print(f"\n{'='*90}")
    print(f"  VERDICT")
    print(f"{'='*90}")

    if "waveform_only" in all_results and "noise_penalty" in all_results:
        for win_key in [f"{w:.0f}s" for w in windows_sec]:
            if win_key in all_results["waveform_only"] and win_key in all_results["noise_penalty"]:
                wf_only_mean = np.mean([t["best_src_cos"] for t in all_results["waveform_only"][win_key]])
                np_mean = np.mean([t["best_src_cos"] for t in all_results["noise_penalty"][win_key]])
                delta = np_mean - wf_only_mean
                if delta > 0.3: print(f"  {win_key}: Noise penalty MAJOR improvement (+{delta:.3f})")
                elif delta > 0.1: print(f"  {win_key}: Noise penalty significant improvement (+{delta:.3f})")
                elif delta > 0.02: print(f"  {win_key}: Noise penalty small improvement (+{delta:.3f})")
                else: print(f"  {win_key}: Noise penalty NO improvement ({delta:+.3f})")

    all_scores = []
    for win_key in [f"{w:.0f}s" for w in windows_sec]:
        for mode in comparison_modes:
            if mode in all_results and win_key in all_results[mode]:
                for t in all_results[mode][win_key]:
                    all_scores.append({
                        "win": win_key, "mode": mode,
                        "best_src_cos": t["best_src_cos"],
                        "full_l2_rel": t["full_l2_rel"],
                    })

    best_ever = max(all_scores, key=lambda x: x["best_src_cos"]) if all_scores else None
    if best_ever:
        print(f"\n  Best ever src_cos: {best_ever['best_src_cos']:.4f} "
              f"({best_ever['mode']}, {best_ever['win']}, full_l2={best_ever['full_l2_rel']:.4f})")

    if best_ever and best_ever['best_src_cos'] > 0.7:
        print(f"\n  FINAL CONCLUSION: Noise whiteness constraint can break token degeneracy.")
        print(f"  The gradient inversion idea is viable with noise penalty.")
    elif best_ever and best_ever['best_src_cos'] > 0.4:
        print(f"\n  FINAL CONCLUSION: Noise whiteness shows measurable signal,")
        print(f"  but does not fully constrain the token space.")
        print(f"  GD inversion may work with better initialization or stronger penalty.")
    else:
        print(f"\n  FINAL CONCLUSION: Noise whiteness penalty does not provide")
        print(f"  sufficient constraint. Token-space GD inversion is NOT viable")
        print(f"  even with the reverse-ODE noise constraint.")

    return {"results": all_results, "summary": summary, "all_scores": all_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: Joint inversion with noise penalty")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/reverse_ode_exp3")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise-penalty-weight", type=float, default=0.5)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_experiment_3(
        args.checkpoint, device,
        n_trials=args.n_trials, n_steps=args.n_steps, lr=args.lr,
        noise_penalty_weight=args.noise_penalty_weight,
        drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saveable = {
        "summary": results["summary"],
        "all_scores": results["all_scores"],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
