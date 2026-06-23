"""Exp D: Continuous latent space editing — magnitude sweep.

Core idea: fix noise seed, interpolate token[0] (source_size) from M3→M8,
generate waveforms. Same noise = "this earthquake, at different magnitudes".

Tests:
  1. Magnitude sweep: amplitude growth, duration change, frequency shift
  2. Depth sweep: token[1] interpolation
  3. Combined magnitude+depth grid
"""

from __future__ import annotations

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from JsT import (
    SeismicConditionEncoder, Denoiser,
    SeismicWaveformDataset, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


@torch.no_grad()
def generate_waveform(dn, tokens, noise, steps=50):
    device = tokens.device
    net = dn.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        xp = net(z, t.expand(z.shape[0]), tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
        z = z + (t_next - t) * v
    return z


def waveform_stats(wf, sr=100.0):
    """Compute summary stats for a 3-channel waveform."""
    # wf: (3, T)
    amps = {}
    for ch, name in enumerate(["Z", "N", "E"]):
        sig = wf[ch]
        amps[f"{name}_max_abs"] = float(np.max(np.abs(sig)))
        amps[f"{name}_rms"] = float(np.sqrt(np.mean(sig**2)))
        # P-wave dominant frequency
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
        amps[f"{name}_dom_freq"] = float(freqs[np.argmax(spec)])
        amps[f"{name}_centroid_freq"] = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-12))
    return amps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--output-dir", default="outputs/expD_magnitude_sweep")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-sweep", type=int, default=12)
    parser.add_argument("--n-examples", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    print("Loading checkpoint...")
    ce, dn, ckpt = load_checkpoint_models(args.checkpoint, dev, use_ema=True,
                                           sampling_method="heun", steps=50, cfg_scale=1.0)
    if dropped: ce = AblationConditionEncoder(ce, dropped)
    dn.eval(); ce.eval()
    total_samples = 3200

    ds_train = SeismicWaveformDataset(args.data_dir, split="training", augment=False,
                                       cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default")
    ds_test = SeismicWaveformDataset(args.data_dir, split="testing", augment=False, vocab_from=ds_train,
                                      cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default")

    # ── Select real earthquakes with known magnitudes ─────────────
    tc = ds_test.conditions.iloc[ds_test.indices].copy()
    # Filter to events with magnitude info
    tc = tc.dropna(subset=["source_magnitude"])
    mags = tc["source_magnitude"].values.astype(float)
    valid_mag = ~np.isnan(mags) & (mags >= 2.5) & (mags <= 8.0)
    tc = tc[valid_mag].copy()
    mags = mags[valid_mag]

    # Pick diverse examples spanning the magnitude range
    idx = np.arange(len(tc))
    rng = np.random.default_rng(42)
    selected = []
    for m_range in [(3.0, 4.5), (4.5, 6.0), (6.0, 8.0)]:
        in_range = idx[(mags >= m_range[0]) & (mags <= m_range[1])]
        if len(in_range) > 0:
            selected.extend(rng.choice(in_range, size=min(2, len(in_range)), replace=False))
    if len(selected) < args.n_examples:
        selected = rng.choice(idx, size=min(args.n_examples, len(idx)), replace=False)

    print(f"Selected {len(selected)} example earthquakes for magnitude sweep")

    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}
    n_sweep = args.n_sweep

    results = []
    sweep_waveforms = {}  # example_idx → list of (3, T) arrays

    for ex_i, ds_idx in enumerate(selected):
        row = tc.iloc[ds_idx]
        real_mag = float(row["source_magnitude"])
        didx = index_to_ds.get(int(row.name))  # row.name is the original index
        if didx is None:
            continue
        wf_tensor, cond_dict = ds_test[didx]

        # Get base tokens
        cond_gpu = {k: v.unsqueeze(0).to(dev) for k, v in cond_dict.items()}
        base_tokens = ce(cond_gpu).clone()  # (1, 8, 512)

        # Fixed noise seed (same earthquake, only magnitude changes)
        torch.manual_seed(42 + ex_i)
        if dev.type == "cuda": torch.cuda.manual_seed_all(42 + ex_i)
        noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=dev)

        # Magnitude sweep: interpolate token[0] linearly
        mags_sweep = np.linspace(3.0, 8.0, n_sweep)
        wf_sequence = []
        stats_sequence = []

        # Get token 0 reference from multiple events to establish scale
        # Simple approach: interpolate token[0] in embedding space
        # Map: magnitude → token[0] change
        # We compute the token[0] direction by comparing events at different magnitudes
        # For now, scale linearly: interpolate between 0.7× and 1.3× the original token

        tok0_orig = base_tokens[0, 0, :].clone()
        # Estimate token scaling from magnitude
        # Assumption: token[0] magnitude is ~linearly related to log moment magnitude
        mag_ratio = mags_sweep / max(real_mag, 1e-6)
        # Scale token in log-space: M ∝ log(M0), token ~ linear in log space
        token_scale = 0.5 + 0.5 * mag_ratio  # 0.5× at M=3 → 1.0× at real_mag → 1.5× at M=8

        for mi, mag_val in enumerate(mags_sweep):
            edited = base_tokens.clone()
            edited[0, 0, :] = tok0_orig * token_scale[mi]
            pred = generate_waveform(dn, edited, noise_fixed, steps=50).cpu().numpy()[0]
            wf_sequence.append(pred)
            stats_sequence.append(waveform_stats(pred))

        sweep_waveforms[str(ex_i)] = {
            "real_magnitude": float(real_mag),
            "sweep_magnitudes": mags_sweep.tolist(),
            "waveforms": [wf.tolist() for wf in wf_sequence],
            "stats": stats_sequence,
        }

        # Quick quality check
        z_max = [s["Z_max_abs"] for s in stats_sequence]
        print(f"  Ex {ex_i} (M={real_mag:.1f}): "
              f"Z_max: {z_max[0]:.3f} → {z_max[-1]:.3f} "
              f"ratio={z_max[-1]/max(z_max[0], 1e-6):.1f}×")

        results.append({
            "example_idx": int(ex_i),
            "real_magnitude": float(real_mag),
            "mag_range": [float(s) for s in [z_max[0], z_max[-1]]],
            "amplitude_ratio": float(z_max[-1] / max(z_max[0], 1e-6)),
            "amplitude_monotonic": all(z_max[i] <= z_max[i+1] for i in range(len(z_max)-1)),
        })

    # ── Save ──────────────────────────────────────────────────────

    # Save waveform data as compressed numpy (much smaller than JSON)
    np.savez_compressed(out / "sweep_waveforms.npz",
                         **{f"ex{i}": np.array(sweep_waveforms[str(i)]["waveforms"], dtype=np.float32)
                            for i, _ in enumerate(sweep_waveforms) if str(i) in sweep_waveforms})

    # Save stats + metadata as JSON
    meta = {
        "n_examples": len(results),
        "n_sweep_steps": n_sweep,
        "magnitude_range": [3.0, 8.0],
        "per_example": results,
        "sweep_magnitudes": [float(m) for m in np.linspace(3.0, 8.0, n_sweep)],
    }
    with open(out / "results.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Synthesis ─────────────────────────────────────────────────

    print(f"\n{'='*60}\n  Magnitude Sweep Results\n{'='*60}")
    n_mono = sum(1 for r in results if r["amplitude_monotonic"])
    amp_ratios = [r["amplitude_ratio"] for r in results]
    print(f"  Monotonic amplitude growth: {n_mono}/{len(results)} examples")
    print(f"  Amplitude ratio M8/M3: {np.mean(amp_ratios):.1f}× (range {min(amp_ratios):.1f}–{max(amp_ratios):.1f})")
    if n_mono >= 0.8 * len(results):
        print(f"  ✓ Token[0] interpolation produces physically meaningful magnitude scaling.")
    else:
        print(f"  ✗ Magnitude sweep not consistently monotonic — token direction may need refinement.")

    print(f"\nSaved: {out}/{{results.json, sweep_waveforms.npz}}")


if __name__ == "__main__":
    main()
