"""Experiment 2: Reverse-ODE noise whiteness as a constraint for token inversion.

Hypothesis: When the ODE is run backwards with the CORRECT token, the
recovered initial noise z_0 should be ~N(0, I). With wrong tokens, the
noise carries compensating structure — testable via distribution tests.

If noise whiteness provides a measurable constraint, it can break the
multi-solution degeneracy observed in Experiment 1/1b.
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
    """Run ODE backward: clean waveform → initial noise.

    Forward ODE:  dz/dt = v = (x̂ - z) / (1-t)
    Reverse ODE:  dz/dt = -v = -(x̂ - z) / (1-t)
    """
    device = waveform.device
    B = waveform.shape[0]
    net = denoiser.net
    t_eps = denoiser.t_eps
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)  # backward
    z = waveform.clone()

    for i in range(steps):
        t = ts[i]
        t_next = ts[i + 1]
        t_batch = t.expand(B)

        x_pred = net(z, t_batch, cond_tokens)
        t_3d = t_batch[:, None, None]
        v = (x_pred - z) / (1.0 - t_3d).clamp_min(t_eps)

        # Reverse step: z(t + dt) = z(t) - dt * v
        dt = t_next - t  # negative (going backward)
        z = z + dt * v

    return z  # should be ~N(0, noise_scale)


@torch.no_grad()
def forward_ode(denoiser, noise, cond_tokens, steps=50):
    """Forward ODE: noise → clean waveform."""
    device = noise.device
    B = noise.shape[0]
    net = denoiser.net
    t_eps = denoiser.t_eps
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()

    for i in range(steps):
        t = ts[i]
        t_next = ts[i + 1]
        t_batch = t.expand(B)
        x_pred = net(z, t_batch, cond_tokens)
        t_3d = t_batch[:, None, None]
        v = (x_pred - z) / (1.0 - t_3d).clamp_min(t_eps)
        dt = t_next - t
        z = z + dt * v

    return z


def noise_whiteness_metrics(noise: torch.Tensor) -> dict:
    """Test whether a tensor looks like white noise N(0, noise_scale²).

    Parameters
    ----------
    noise: (B, C, T) recovered noise tensor

    Returns dict with:
        - mean, std: distribution moments
        - skewness, kurtosis: higher moments
        - autocorr_lag1: lag-1 autocorrelation (should be ~0 for white noise)
        - autocorr_lag5: lag-5 autocorrelation
        - shapiro_pvalue: Shapiro-Wilk normality p-value (per-channel mean)
        - energy_concentration: fraction of energy in first 10% of frequencies
        - channel_corr: mean cross-channel correlation (should be ~0)
    """
    from scipy import stats as sp_stats

    n = noise.cpu().float().numpy()
    B, C, T = n.shape
    n_flat = n.flatten()

    metrics = {
        "mean": float(np.mean(n_flat)),
        "std": float(np.std(n_flat)),
        "skewness": float(sp_stats.skew(n_flat)),
        "kurtosis": float(sp_stats.kurtosis(n_flat, fisher=True)),  # excess kurtosis
    }

    # Per-channel Shapiro-Wilk
    pvals = []
    for ch in range(C):
        ch_data = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
        if len(ch_data) > 5000:
            ch_data = np.random.choice(ch_data, 5000, replace=False)
        stat, pval = sp_stats.shapiro(ch_data)
        pvals.append(pval)
    metrics["shapiro_pval_mean"] = float(np.mean(pvals))
    metrics["shapiro_pval_min"] = float(np.min(pvals))

    # Autocorrelation
    for lag in [1, 5, 10, 20]:
        ac_values = []
        for ch in range(C):
            ch_data = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
            ac = np.corrcoef(ch_data[:-lag], ch_data[lag:])[0, 1] if len(ch_data) > lag + 1 else 0
            ac_values.append(ac)
        metrics[f"autocorr_lag{lag}"] = float(np.mean(ac_values))

    # Spectral flatness (energy concentration)
    for ch, ch_name in enumerate(["Z", "N", "E"]):
        ch_data = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
        spec = np.abs(np.fft.rfft(ch_data))
        first_10pct = int(len(spec) * 0.1)
        metrics[f"spec_conc_{ch_name}"] = float(np.sum(spec[:first_10pct]) / max(np.sum(spec), 1e-12))

    # Cross-channel correlation
    if B == 1:
        cc01 = np.corrcoef(n[0, 0], n[0, 1])[0, 1]
        cc02 = np.corrcoef(n[0, 0], n[0, 2])[0, 1]
        cc12 = np.corrcoef(n[0, 1], n[0, 2])[0, 1]
        metrics["channel_corr_mean"] = float(np.mean([cc01, cc02, cc12]))

    return metrics


def run_experiment_2(
    checkpoint_path: str,
    device: torch.device,
    n_trials: int = 5,
    ode_steps: int = 50,
    drop_tokens: list[int] | None = None,
) -> dict:
    """Test noise whiteness as token correctness detector.

    For each trial:
    1. Pick real waveform + ground truth tokens from test set
    2. Forward ODE with true tokens → W (synthetic target)
    3. Save ground-truth initial noise z_0_true
    4. Reverse ODE with TRUE tokens → ẑ_true (should be white noise)
    5. Reverse ODE with RANDOM tokens → ẑ_random (should NOT be white noise)
    6. Reverse ODE with NULL tokens → ẑ_null
    7. Reverse ODE with PERTURBED tokens → ẑ_perturbed
    8. Compare noise metrics across candidates

    KEY QUESTION: Can noise whiteness distinguish the true token from wrong ones?
    """
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

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    print(f"Experiment 2: Reverse-ODE noise whiteness")
    print(f"  Trials: {n_trials}")
    print(f"  ODE steps: {ode_steps}")
    print()

    all_results = []

    for trial in range(n_trials):
        print(f"--- Trial {trial+1}/{n_trials} ---")
        trial_results = {}

        # Pick random test sample
        cache_idx = int(torch.randint(0, len(ds_test), (1,)).item())
        waveform_tensor, cond_dict = ds_test[cache_idx]
        waveform = waveform_tensor.unsqueeze(0).to(device)
        cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
        true_tokens = ce(cond_gpu)

        # Forward: known noise → synthetic waveform (ground truth for reverse test)
        torch.manual_seed(42 + trial * 100)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42 + trial * 100)
        z0_true = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
        with torch.no_grad():
            W_syn = forward_ode(dn, z0_true, true_tokens, steps=ode_steps)

        # ---- Test 1: Reverse with TRUE tokens ----
        z_rec_true = reverse_ode(dn, W_syn, true_tokens, steps=ode_steps)
        trial_results["true_tokens"] = noise_whiteness_metrics(z_rec_true)
        print(f"  TRUE tokens:      autocorr_l1={trial_results['true_tokens']['autocorr_lag1']:.4f}  "
              f"shapiro_p={trial_results['true_tokens']['shapiro_pval_mean']:.4f}  "
              f"kurtosis={trial_results['true_tokens']['kurtosis']:.3f}")

        # ---- Test 2: Reverse with random tokens ----
        torch.manual_seed(99 + trial)
        random_tokens = torch.randn(1, n_tokens, hidden, device=device) * 2.0
        z_rec_random = reverse_ode(dn, W_syn, random_tokens, steps=ode_steps)
        trial_results["random_tokens"] = noise_whiteness_metrics(z_rec_random)
        print(f"  RANDOM tokens:    autocorr_l1={trial_results['random_tokens']['autocorr_lag1']:.4f}  "
              f"shapiro_p={trial_results['random_tokens']['shapiro_pval_mean']:.4f}  "
              f"kurtosis={trial_results['random_tokens']['kurtosis']:.3f}")

        # ---- Test 3: Reverse with null tokens ----
        null_tokens = net.null_tokens.expand(1, -1, -1).to(device)
        z_rec_null = reverse_ode(dn, W_syn, null_tokens, steps=ode_steps)
        trial_results["null_tokens"] = noise_whiteness_metrics(z_rec_null)
        print(f"  NULL tokens:      autocorr_l1={trial_results['null_tokens']['autocorr_lag1']:.4f}  "
              f"shapiro_p={trial_results['null_tokens']['shapiro_pval_mean']:.4f}  "
              f"kurtosis={trial_results['null_tokens']['kurtosis']:.3f}")

        # ---- Test 4: Reverse with perturbed tokens ----
        perturb_scale = 0.5
        torch.manual_seed(55 + trial)
        perturbed = true_tokens.clone() + perturb_scale * torch.randn(1, n_tokens, hidden, device=device)
        z_rec_pert = reverse_ode(dn, W_syn, perturbed, steps=ode_steps)
        trial_results["perturbed_tokens"] = noise_whiteness_metrics(z_rec_pert)
        print(f"  PERTURBED tokens: autocorr_l1={trial_results['perturbed_tokens']['autocorr_lag1']:.4f}  "
              f"shapiro_p={trial_results['perturbed_tokens']['shapiro_pval_mean']:.4f}  "
              f"kurtosis={trial_results['perturbed_tokens']['kurtosis']:.3f}")

        # ---- Test 5: Reverse with GD-optimized tokens (from Experiment 1 style) ----
        # Quick 200-step GD to get a "wrong but waveform-matching" token
        torch.manual_seed(77 + trial)
        gd_tokens = net.null_tokens.expand(1, -1, -1).clone().detach()
        gd_tokens.add_(0.1 * torch.randn(1, n_tokens, hidden, device=device))
        gd_tokens.requires_grad_(True)
        opt = torch.optim.Adam([gd_tokens], lr=0.1)

        noises_pool = [dn.noise_scale * torch.randn(1, 3, total_samples, device=device) for _ in range(20)]
        for step in range(200):
            opt.zero_grad()
            loss = None
            for i in range(5):
                tv = torch.rand(1, device=device).item()
                tv = max(tv, dn.t_eps); tv = min(tv, 1.0 - dn.t_eps)
                eps = noises_pool[(step * 5 + i) % len(noises_pool)]
                t3d = torch.tensor(tv, device=device).view(1, 1, 1)
                zs = t3d * W_syn + (1.0 - t3d) * eps
                tp = net(zs, torch.full((1,), tv, device=device), gd_tokens)
                term = ((tp - W_syn) ** 2).mean()
                loss = term if loss is None else loss + term
            loss = loss / 5
            loss.backward()
            opt.step()

        gd_tokens = gd_tokens.detach()
        z_rec_gd = reverse_ode(dn, W_syn, gd_tokens, steps=ode_steps)
        trial_results["gd_optimized"] = noise_whiteness_metrics(z_rec_gd)

        # Check waveform match of GD tokens
        with torch.no_grad():
            W_gd = forward_ode(dn, z0_true, gd_tokens, steps=ode_steps)
            gd_wf_l2 = ((W_gd - W_syn) ** 2).mean().sqrt().item()
            gd_wf_norm = (W_syn ** 2).mean().sqrt().item()
            gd_wf_rel = gd_wf_l2 / max(gd_wf_norm, 1e-8)
        print(f"  GD-OPT tokens:    autocorr_l1={trial_results['gd_optimized']['autocorr_lag1']:.4f}  "
              f"shapiro_p={trial_results['gd_optimized']['shapiro_pval_mean']:.4f}  "
              f"kurtosis={trial_results['gd_optimized']['kurtosis']:.3f}  "
              f"wf_l2_rel={gd_wf_rel:.4f}")

        all_results.append(trial_results)
        print()

    # ---- Summary ----
    print("=" * 80)
    print("  REVERSE-ODE NOISE WHITENESS — SUMMARY")
    print("=" * 80)
    print()

    # For white noise N(0, noise_scale²):
    #   mean ≈ 0, std ≈ noise_scale, skewness ≈ 0, excess kurtosis ≈ 0
    #   autocorr_lag1 ≈ 0, shapiro_p > 0.05, spec_conc ≈ 0.1, channel_corr ≈ 0

    metrics_to_compare = [
        ("autocorr_lag1", 0.0, "±0.02"),
        ("autocorr_lag5", 0.0, "±0.02"),
        ("skewness", 0.0, "±0.1"),
        ("kurtosis", 0.0, "±0.2"),
        ("shapiro_pval_mean", 1.0, ">0.05"),
        ("spec_conc_Z", 0.1, "±0.03"),
        ("channel_corr_mean", 0.0, "±0.1"),
    ]

    for metric, ideal, tolerance in metrics_to_compare:
        print(f"\n  {metric} (ideal: {ideal}, {tolerance}):")
        for label in ["true_tokens", "random_tokens", "null_tokens", "perturbed_tokens", "gd_optimized"]:
            values = [t[label][metric] for t in all_results if label in t and metric in t[label]]
            if values:
                mean_val = np.mean(values)
                marker = " ✓" if abs(mean_val - ideal) < 0.05 else " ✗"
                if ideal == 1.0:  # Shapiro
                    marker = " ✓" if mean_val > 0.05 else " ✗"
                print(f"    {label:20s}: {mean_val:+.4f}{marker}")

    # KEY QUESTION: Does the true token produce measurably whiter noise?
    print()
    print("=" * 80)
    print("  KEY QUESTION: Can reverse-ODE noise whiteness identify the true token?")
    print("=" * 80)

    # Score: how many of the 6 metrics are "white" for each candidate
    for label in ["true_tokens", "random_tokens", "null_tokens", "perturbed_tokens", "gd_optimized"]:
        white_count = 0
        total_count = 0
        for metric, ideal, _ in metrics_to_compare:
            values = [t[label][metric] for t in all_results if label in t and metric in t[label]]
            if values:
                mean_val = np.mean(values)
                if ideal == 1.0:
                    white = mean_val > 0.05
                else:
                    white = abs(mean_val - ideal) < 0.05
                if white: white_count += 1
                total_count += 1
        print(f"  {label:20s}: {white_count}/{total_count} white-noise metrics passed")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Reverse-ODE noise whiteness")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/reverse_ode_exp2")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_experiment_2(
        args.checkpoint, device,
        n_trials=args.n_trials,
        ode_steps=args.ode_steps,
        drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics (skip full noise tensors)
    saveable = []
    for t_result in results:
        trial_save = {}
        for label, metrics in t_result.items():
            if isinstance(metrics, dict):
                trial_save[label] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                                     for k, v in metrics.items()}
        saveable.append(trial_save)

    with open(output_dir / "results.json", "w") as f:
        json.dump(saveable, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
