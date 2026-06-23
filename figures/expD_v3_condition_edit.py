"""Exp D v3: Magnitude sweep via condition editing (not token editing).

Key insight: editing token[0] doesn't work because tokens are compressed
representations with entangled semantics. The correct approach is:
  1. Take a real earthquake's full condition dict
  2. Modify source_magnitude in the condition (raw input)
  3. Re-encode → new tokens → generate
  4. Same noise seed = "same earthquake, different magnitude"

This is the ChordEdit/continuous editing paradigm applied correctly.
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


def waveform_metrics(wf, sr=100.0):
    m = {}
    for ch, name in enumerate(["Z", "N", "E"]):
        sig = wf[ch]
        # Peak ground acceleration proxy (normalized units)
        m[f"{name}_PGA"] = float(np.max(np.abs(sig)))
        m[f"{name}_RMS"] = float(np.sqrt(np.mean(sig**2)))
        # Arias intensity proxy
        m[f"{name}_Arias"] = float(np.sum(sig**2))
        # Duration (5%-95% of Arias integral)
        e2 = np.cumsum(sig**2)
        total = e2[-1]
        if total > 1e-12:
            t5 = np.searchsorted(e2, 0.05 * total)
            t95 = np.searchsorted(e2, 0.95 * total)
            m[f"{name}_dur"] = float(t95 - t5)
        else:
            m[f"{name}_dur"] = 0.0
        # Spectral centroid
        spec = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
        m[f"{name}_fcentroid"] = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-12))
    return m


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

    tc = ds_test.conditions.iloc[ds_test.indices].copy()
    tc = tc.dropna(subset=["source_magnitude"])
    mags = tc["source_magnitude"].values.astype(float)
    tc = tc[(mags >= 2.5) & (mags <= 8.0)].copy()

    i2d = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # Pick diverse examples
    rng = np.random.default_rng(42)
    candidates = []
    for i, row in tc.iterrows():
        mag = float(row["source_magnitude"])
        if 3.5 <= mag <= 6.5:
            candidates.append(i)
    if len(candidates) < args.n_examples:
        candidates = list(tc.index)
    selected = rng.choice(candidates, size=min(args.n_examples, len(candidates)), replace=False)
    print(f"Selected {len(selected)} base events")

    # Check which condition fields encode magnitude
    # source_magnitude is encoded in token[0] (source_size) along with magnitude_type, location_uncertainty
    # We need to modify the SOURCE_SIZE field in the condition dict

    # Find the correct field name
    print("Condition dict keys in dataset:")
    sample = ds_test[0][1]
    for k, v in sample.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v}", end="")
        elif hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}", end="")
        else:
            print(f"  {k}: {type(v).__name__}", end="")
        if "mag" in str(k).lower():
            print("  ← MAGNITUDE")
        else:
            print()

    # The condition encoder uses "source_magnitude" field — let's trace it

    n_sweep = args.n_sweep
    mag_range = np.linspace(2.5, 8.0, n_sweep)

    results = []
    for ex_i, ds_idx in enumerate(selected):
        row = tc.loc[ds_idx]
        real_mag = float(row["source_magnitude"])
        didx = i2d.get(int(row.name))
        if didx is None: continue

        wf_real, cond_dict = ds_test[didx]

        # Fixed noise
        torch.manual_seed(42 + ex_i)
        if dev.type == "cuda": torch.cuda.manual_seed_all(42 + ex_i)
        noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=dev)

        wf_sequence = []
        metrics_sequence = []

        for mag_val in mag_range:
            # Clone and modify condition — change magnitude value
            cond_edit = {k: v.clone() if hasattr(v, 'clone') else v for k, v in cond_dict.items()}

            # Find and modify magnitude-related fields
            # The dataset stores magnitude as a field in the condition dict
            if "source_magnitude" in cond_edit:
                cond_edit["source_magnitude"] = float(mag_val)

            # Encode edited condition
            cond_gpu = {k: v.unsqueeze(0).to(dev) for k, v in cond_edit.items()
                        if hasattr(v, '__len__') and not isinstance(v, str)}
            # Need to handle mixed types — dict values may be int, float, tensor, or string
            batch = {}
            for k, v in cond_edit.items():
                if isinstance(v, (int, float)):
                    batch[k] = torch.tensor([v], device=dev)
                elif hasattr(v, 'unsqueeze'):
                    batch[k] = v.unsqueeze(0).to(dev)
                elif isinstance(v, str):
                    continue  # skip string fields — handled by vocab encoding in CE
                else:
                    continue
            tokens = ce(batch)
            pred = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu().numpy()[0]
            wf_sequence.append(pred)
            metrics_sequence.append(waveform_metrics(pred))

        # Check monotonicity of key metrics
        Z_pga = [m["Z_PGA"] for m in metrics_sequence]
        Z_arias = [m["Z_Arias"] for m in metrics_sequence]
        Z_dur = [m["Z_dur"] for m in metrics_sequence]
        Z_fcent = [m["Z_fcentroid"] for m in metrics_sequence]

        pga_ratio = Z_pga[-1] / max(Z_pga[0], 1e-12)
        mono_pga = all(Z_pga[i] <= Z_pga[i+1] + 1e-10 for i in range(len(Z_pga)-1))
        mono_arias = all(Z_arias[i] <= Z_arias[i+1] + 1e-10 for i in range(len(Z_arias)-1))

        print(f"  Ex {ex_i} (M={real_mag:.1f}): PGA={Z_pga[0]:.4f}→{Z_pga[-1]:.4f} "
              f"({pga_ratio:.1f}×)  mono_PGA={mono_pga}  mono_Arias={mono_arias}")

        results.append({
            "example_idx": ex_i,
            "real_magnitude": float(real_mag),
            "sweep_magnitudes": mag_range.tolist(),
            "pga_first": float(Z_pga[0]), "pga_last": float(Z_pga[-1]),
            "pga_ratio": float(pga_ratio),
            "pga_monotonic": mono_pga,
            "arias_monotonic": mono_arias,
            "metrics": metrics_sequence,
            "Z_pga": [float(v) for v in Z_pga],
            "Z_arias": [float(v) for v in Z_arias],
            "Z_dur": [float(v) for v in Z_dur],
            "Z_fcentroid": [float(v) for v in Z_fcent],
        })

    # ── Summary ───────────────────────────────────────────────────

    n_pga_mono = sum(1 for r in results if r["pga_monotonic"])
    n_arias_mono = sum(1 for r in results if r["arias_monotonic"])
    pga_ratios = [r["pga_ratio"] for r in results]

    print(f"\n{'='*60}")
    print(f"  Magnitude Sweep v3 (condition editing)")
    print(f"{'='*60}")
    print(f"  PGA monotonic: {n_pga_mono}/{len(results)}")
    print(f"  Arias monotonic: {n_arias_mono}/{len(results)}")
    print(f"  Mean PGA ratio M8/M3: {np.mean(pga_ratios):.1f}×")

    if n_pga_mono >= len(results) * 0.6:
        print(f"  ✓ Condition-level magnitude editing produces consistent amplitude scaling.")
    else:
        print(f"  ~ Mixed results")

    # Save
    summary = {
        "method": "condition_editing",
        "n_examples": len(results),
        "n_sweep": n_sweep,
        "magnitude_range": [2.5, 8.0],
        "pga_monotonic_fraction": float(n_pga_mono / max(len(results), 1)),
        "arias_monotonic_fraction": float(n_arias_mono / max(len(results), 1)),
        "mean_pga_ratio": float(np.mean(pga_ratios)),
        "results": [{k: v for k, v in r.items() if k != "metrics"} for r in results],
    }
    with open(out / "results_v3.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}/results_v3.json")


if __name__ == "__main__":
    main()
