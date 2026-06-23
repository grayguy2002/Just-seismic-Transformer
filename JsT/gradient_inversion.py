"""Experiment 1b: Window sweep — how much waveform uniquely constrains tokens?

Variable observation window [4, 8, 12, 16, 24, 40, 60, 80] seconds.
At each window, GD optimizes source tokens to match only the first W
seconds. Measures whether token recovery improves with more data.

Core EEW hypothesis: incoming waveform progressively collapses the
multi-solution ambiguity in JsT's token space.
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


def run_window_sweep(
    checkpoint_path: str,
    device: torch.device,
    n_trials: int = 3,
    n_steps: int = 500,
    lr: float = 0.1,
    t_samples: int = 5,
    windows_sec: list[float] | None = None,
    drop_tokens: list[int] | None = None,
) -> dict:
    if windows_sec is None:
        windows_sec = [4.0, 8.0, 12.0, 16.0, 24.0, 40.0, 60.0, 80.0]

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

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    print(f"Window sweep: {len(windows_sec)} windows x {n_trials} trials = {len(windows_sec)*n_trials} runs")
    print(f"Steps: {n_steps}, LR: {lr}, t_samples: {t_samples}")
    print(f"Source tokens only: {source_idx}")
    print()

    all_results = {}

    for win_idx, win_sec in enumerate(windows_sec):
        win_samples = min(int(win_sec * sample_rate_hz), total_samples)
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"[{win_idx+1}/{len(windows_sec)}] Window: {win_sec:.0f}s ({win_samples} samples)")
        print(f"{'='*60}")

        trial_results = []

        for trial in range(n_trials):
            # 1. Pick random test sample, encode true tokens
            cache_idx = int(torch.randint(0, len(ds_test), (1,)).item())
            waveform_tensor, cond_dict = ds_test[cache_idx]
            waveform = waveform_tensor.unsqueeze(0).to(device)
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            true_tokens = ce(cond_gpu)

            # 2. Generate synthetic target waveform
            seed_base = 42 + win_idx * 1000 + trial * 100
            with torch.no_grad():
                torch.manual_seed(seed_base)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed_base)
                noise = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                ts_ode = torch.linspace(0.0, 1.0, 51, device=device)
                z = noise
                for i in range(49):
                    t_b = ts_ode[i].expand(1)
                    xp = net(z, t_b, true_tokens)
                    t3 = ts_ode[i].view(1, 1, 1)
                    v = (xp - z) / (1.0 - t3).clamp_min(dn.t_eps)
                    z = z + (ts_ode[i + 1] - ts_ode[i]) * v
            target_wf = z.clone()

            # 3. Window mask
            window_mask = torch.zeros(1, 3, total_samples, device=device)
            window_mask[:, :, :win_samples] = 1.0

            # 4. Initialize tokens (null + noise), freeze non-source
            torch.manual_seed(300 + seed_base)
            init_tokens = net.null_tokens.expand(1, -1, -1).clone().detach()
            init_tokens.add_(0.1 * torch.randn(1, n_tokens, hidden, device=device))
            init_tokens.requires_grad_(True)
            with torch.no_grad():
                for idx in range(n_tokens):
                    if idx not in source_idx:
                        init_tokens[:, idx, :] = true_tokens[:, idx, :]

            # 5. Optimize
            opt = torch.optim.Adam([init_tokens], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.01)
            torch.manual_seed(400 + seed_base)
            noises_pool = [dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                           for _ in range(t_samples * 2)]

            for step in range(n_steps):
                opt.zero_grad()
                loss = None
                for i in range(t_samples):
                    tv = torch.rand(1, device=device).item()
                    tv = max(tv, dn.t_eps)
                    tv = min(tv, 1.0 - dn.t_eps)
                    eps = noises_pool[(step * t_samples + i) % len(noises_pool)]
                    t3d = torch.tensor(tv, device=device).view(1, 1, 1)
                    zs = t3d * target_wf + (1.0 - t3d) * eps
                    tp = net(zs, torch.full((1,), tv, device=device), init_tokens)
                    diff = (tp - target_wf) * window_mask
                    term = (diff ** 2).mean()
                    loss = term if loss is None else loss + term
                loss = loss / t_samples
                loss.backward()
                torch.nn.utils.clip_grad_norm_([init_tokens], 1.0)
                opt.step()
                sched.step()

            final_tokens = init_tokens.detach()

            # 6. Evaluate: generate from final tokens, compare
            with torch.no_grad():
                torch.manual_seed(seed_base)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed_base)
                noise2 = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                z2 = noise2
                for i in range(49):
                    t_b2 = ts_ode[i].expand(1)
                    xp2 = net(z2, t_b2, final_tokens)
                    t3d2 = ts_ode[i].view(1, 1, 1)
                    v2 = (xp2 - z2) / (1.0 - t3d2).clamp_min(dn.t_eps)
                    z2 = z2 + (ts_ode[i + 1] - ts_ode[i]) * v2
                recovered_wf = z2

                # Observed-window L2
                od = (recovered_wf - target_wf) * window_mask
                ol2 = (od ** 2).mean().sqrt().item()
                onm = ((target_wf * window_mask) ** 2).mean().sqrt().item()
                obs_l2 = ol2 / max(onm, 1e-8)

                # Full-waveform L2 (extrapolation)
                fd = recovered_wf - target_wf
                fl2 = (fd ** 2).mean().sqrt().item()
                fnm = (target_wf ** 2).mean().sqrt().item()
                full_l2 = fl2 / max(fnm, 1e-8)

                # Token cosine
                src_cos = torch.nn.functional.cosine_similarity(
                    final_tokens[0, source_idx], true_tokens[0, source_idx], dim=-1
                ).mean().item()

                all_cos = torch.nn.functional.cosine_similarity(
                    final_tokens[0], true_tokens[0], dim=-1
                ).mean().item()

                # Per-source-token cosine
                pcos = {}
                for si in source_idx:
                    pcos[f"t{si}_cos"] = torch.nn.functional.cosine_similarity(
                        final_tokens[0, si:si+1], true_tokens[0, si:si+1], dim=-1
                    ).item()

            trial_results.append({
                "win_sec": win_sec, "win_samples": win_samples, "trial": trial + 1,
                "obs_l2": float(obs_l2), "full_l2": float(full_l2),
                "src_cos": float(src_cos), "all_cos": float(all_cos), "pcos": pcos,
            })

        elapsed = time.time() - t0
        # Print per-window summary
        avg_obs = np.mean([t["obs_l2"] for t in trial_results])
        avg_full = np.mean([t["full_l2"] for t in trial_results])
        avg_src = np.mean([t["src_cos"] for t in trial_results])
        print(f"  → obs_l2={avg_obs:.4f}  full_l2={avg_full:.4f}  src_cos={avg_src:.4f}  ({elapsed:.0f}s)")

        all_results[f"{win_sec:.0f}s"] = trial_results

    # ---- Final summary table ----
    print(f"\n{'='*75}")
    print(f"  WINDOW SWEEP — FINAL SUMMARY")
    print(f"{'='*75}")
    print(f"{'Window':>8s}  {'obs L2':>7s}  {'full L2':>7s}  {'src cos':>7s}  {'all cos':>7s}  {'t0 cos':>7s}  {'t1 cos':>7s}  {'t2 cos':>7s}  {'VERDICT'}")
    print("-" * 90)

    summary = {}
    for win_key in [f"{w:.0f}s" for w in windows_sec]:
        if win_key not in all_results:
            continue
        trials = all_results[win_key]
        a_obs = np.mean([t["obs_l2"] for t in trials])
        a_full = np.mean([t["full_l2"] for t in trials])
        a_src = np.mean([t["src_cos"] for t in trials])
        a_all = np.mean([t["all_cos"] for t in trials])
        t0 = np.mean([t["pcos"].get("t0_cos", 0) for t in trials])
        t1 = np.mean([t["pcos"].get("t1_cos", 0) for t in trials])
        t2 = np.mean([t["pcos"].get("t2_cos", 0) for t in trials])

        if a_src > 0.9: v = "FULLY CONSTRAINED"
        elif a_src > 0.7: v = "CONVERGING"
        elif a_src > 0.4: v = "EMERGING"
        elif a_src > 0.25: v = "WEAK SIGNAL"
        else: v = "DEGENERATE"

        print(f"{win_key:>8s}  {a_obs:7.4f}  {a_full:7.4f}  {a_src:7.4f}  {a_all:7.4f}  {t0:7.4f}  {t1:7.4f}  {t2:7.4f}  {v}")
        summary[win_key] = {"avg_obs_l2": float(a_obs), "avg_full_l2": float(a_full),
                            "avg_src_cos": float(a_src), "avg_all_cos": float(a_all),
                            "t0_cos": float(t0), "t1_cos": float(t1), "t2_cos": float(t2),
                            "verdict": v}

    # Find critical window: where src_cos crosses meaningful thresholds
    windows_arr = np.array([float(k.replace("s", "")) for k in summary.keys()])
    src_cos_arr = np.array([summary[k]["avg_src_cos"] for k in summary.keys()])
    order = np.argsort(windows_arr)

    for threshold, label in [(0.5, "weak constraint"), (0.7, "meaningful constraint"), (0.9, "near-unique")]:
        above = np.where(src_cos_arr[order] >= threshold)[0]
        if len(above) > 0:
            crit_win = windows_arr[order][above[0]]
            print(f"\n  {label} (cos > {threshold:.1f}) at window ≥ {crit_win:.0f}s")
        else:
            print(f"\n  {label} (cos > {threshold:.1f}) — NOT REACHED in tested range")

    return {"windows": all_results, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JsT window sweep — Experiment 1b")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/grad_invert_exp1b")
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--t-samples", type=int, default=5)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_window_sweep(
        args.checkpoint, device,
        n_trials=args.n_trials, n_steps=args.n_steps,
        lr=args.lr, t_samples=args.t_samples,
        drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "window_sweep_results.json", "w") as f:
        json.dump({"summary": results["summary"]}, f, indent=2)
    print(f"\nSaved to {output_dir}/window_sweep_results.json")
