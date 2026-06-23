"""Exp D v2: Continuous magnitude sweep — learned direction approach.

Unlike v1 (simple token[0] linear scaling, which failed), this version:
  1. Finds event pairs at the SAME station with different magnitudes
  2. Learns Δtoken[0] / Δmagnitude direction from these pairs
  3. Interpolates along the learned direction for continuous magnitude editing

This is the ChordEdit paradigm: find the semantic direction in token space,
then traverse it continuously.

Also adds: depth sweep using token[1] direction.
"""

from __future__ import annotations

import sys, json, argparse
from pathlib import Path
from collections import defaultdict

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
    amps = {}
    for ch, name in enumerate(["Z", "N", "E"]):
        sig = wf[ch]
        amps[f"{name}_max_abs"] = float(np.max(np.abs(sig)))
        amps[f"{name}_rms"] = float(np.sqrt(np.mean(sig**2)))
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
    parser.add_argument("--n-sweep", type=int, default=15)
    parser.add_argument("--min-mag-gap", type=float, default=0.3,
                        help="Minimum magnitude difference for pair selection")
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
    token_dim = dn.net.hidden_size

    ds_train = SeismicWaveformDataset(args.data_dir, split="training", augment=False,
                                       cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default")
    ds_test = SeismicWaveformDataset(args.data_dir, split="testing", augment=False, vocab_from=ds_train,
                                      cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default")

    tc = ds_test.conditions.iloc[ds_test.indices].copy()
    tc["station_id"] = (tc["station_network_code"].fillna("UNKNOWN").astype(str)
                        + "." + tc["station_code"].fillna("UNKNOWN").astype(str))
    tc = tc.dropna(subset=["source_magnitude"])
    mags = tc["source_magnitude"].values.astype(float)
    tc = tc[(mags >= 2.5) & (mags <= 8.0)].copy()

    i2d = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # ── Step 1: Learn magnitude direction from pairs ──────────────

    print("Learning magnitude direction from event pairs...")
    mag_directions = []  # list of (Δmag, Δtoken[0])
    depth_directions = []  # list of (Δdepth, Δtoken[1])

    for sta_id, group in tc.groupby("station_id"):
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row_i, row_j = group.iloc[i], group.iloc[j]
                mag_i = float(row_i["source_magnitude"])
                mag_j = float(row_j["source_magnitude"])
                dmag = abs(mag_i - mag_j)
                if dmag < args.min_mag_gap:
                    continue
                # Get tokens for both events
                di_i = i2d.get(int(row_i.name))
                di_j = i2d.get(int(row_j.name))
                if di_i is None or di_j is None:
                    continue
                _, cd_i = ds_test[di_i]
                _, cd_j = ds_test[di_j]
                cg_i = {k: v.unsqueeze(0).to(dev) for k, v in cd_i.items()}
                cg_j = {k: v.unsqueeze(0).to(dev) for k, v in cd_j.items()}
                with torch.no_grad():
                    tok_i = ce(cg_i).cpu().numpy()[0]  # (8, D)
                    tok_j = ce(cg_j).cpu().numpy()[0]
                dtok0 = tok_j[0] - tok_i[0]  # Δtoken[0]
                dtok1 = tok_j[1] - tok_i[1]  # Δtoken[1]
                mag_directions.append({"dmag": dmag, "dtok0": dtok0,
                                        "dtok0_norm": float(np.linalg.norm(dtok0)),
                                        "sta": sta_id})
                # Also depth pairs
                depth_i = float(row_i.get("source_depth_km", 10))
                depth_j = float(row_j.get("source_depth_km", 10))
                ddepth = abs(depth_i - depth_j)
                if ddepth > 2.0:
                    depth_directions.append({"ddepth": ddepth, "dtok1": dtok1,
                                              "dtok1_norm": float(np.linalg.norm(dtok1)),
                                              "sta": sta_id})

    print(f"  Magnitude pairs: {len(mag_directions)}, Depth pairs: {len(depth_directions)}")

    if len(mag_directions) < 10:
        print("ERROR: too few magnitude pairs for direction learning. Lower --min-mag-gap?")
        return

    # Compute mean magnitude direction (weighted by 1/dmag for stability)
    dtoks = np.array([d["dtok0"] for d in mag_directions])
    dmags = np.array([d["dmag"] for d in mag_directions])
    # Normalise each dtok before averaging
    dtok_norms = np.array([max(n, 1e-8) for n in [d["dtok0_norm"] for d in mag_directions]])
    dtoks_normed = dtoks / dtok_norms[:, np.newaxis]
    # Weight by magnitude gap
    weights = dmags / dmags.sum()
    mag_dir = np.average(dtoks_normed, axis=0, weights=weights)
    mag_dir = mag_dir / (np.linalg.norm(mag_dir) + 1e-12)

    print(f"  Learned magnitude direction norm: {np.linalg.norm(mag_dir):.4f}")

    # ── Step 2: Select test earthquake and sweep ──────────────────

    # Pick a well-observed station+event as base
    rng = np.random.default_rng(42)
    # Prefer mid-magnitude events at high-event-count stations
    candidates = []
    for i, row in tc.iterrows():
        mag = float(row["source_magnitude"])
        if 3.5 <= mag <= 6.0:
            candidates.append(i)
    if len(candidates) < 3:
        candidates = list(tc.index)
    selected = rng.choice(candidates, size=min(3, len(candidates)), replace=False)

    print(f"\nSelected {len(selected)} base events for magnitude sweep")

    sweep_results = []
    for ex_idx, ds_idx in enumerate(selected):
        row = tc.loc[ds_idx]
        real_mag = float(row["source_magnitude"])
        real_depth = float(row.get("source_depth_km", 10)) if pd.notna(row.get("source_depth_km", 10)) else 10.0
        didx = i2d.get(int(row.name))
        if didx is None:
            continue

        wf_tensor, cond_dict = ds_test[didx]
        cond_gpu = {k: v.unsqueeze(0).to(dev) for k, v in cond_dict.items()}
        base_tokens = ce(cond_gpu).clone().detach()

        # Fixed noise
        torch.manual_seed(42 + ex_idx)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(42 + ex_idx)
        noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=dev)

        # Magnitude sweep along learned direction
        n_sweep = args.n_sweep
        mag_sweep_vals = np.linspace(2.5, 8.0, n_sweep)
        mag_waveforms = []
        mag_stats = []

        for mi, mag_val in enumerate(mag_sweep_vals):
            delta_mag = mag_val - real_mag
            # Scale token[0] along learned direction proportional to Δmag
            edited = base_tokens.clone()
            mag_step = torch.from_numpy(mag_dir * delta_mag * 0.5).float().to(dev)
            edited[0, 0, :] += mag_step
            pred = generate_waveform(dn, edited, noise_fixed, steps=50).cpu().numpy()[0]
            mag_waveforms.append(pred)
            mag_stats.append(waveform_stats(pred))

        z_max = [s["Z_max_abs"] for s in mag_stats]
        # Check monotonicity of absolute max amplitude
        z_abs = [s["Z_max_abs"] for s in mag_stats]
        ratio_amp = z_abs[-1] / max(z_abs[0], 1e-6)
        monotonic = all(z_abs[i] <= z_abs[i+1] + 1e-8 for i in range(len(z_abs)-1))

        print(f"  Ex {ex_idx} (M={real_mag:.1f}): "
              f"Z_max: {z_max[0]:.3f} → {z_max[-1]:.3f} "
              f"ratio={ratio_amp:.1f}×  monotonic={monotonic}")

        sweep_results.append({
            "example_idx": int(ex_idx),
            "real_magnitude": float(real_mag),
            "real_depth": float(real_depth),
            "sweep_magnitudes": mag_sweep_vals.tolist(),
            "z_max_first": float(z_max[0]),
            "z_max_last": float(z_max[-1]),
            "amplitude_ratio": float(ratio_amp),
            "monotonic": monotonic,
            "stats": mag_stats,
            "waveforms": [wf.tolist() for wf in mag_waveforms],
        })

    # ── Verdict ───────────────────────────────────────────────────

    n_mono = sum(1 for r in sweep_results if r["monotonic"])
    amp_ratios = [r["amplitude_ratio"] for r in sweep_results]

    print(f"\n{'='*60}\n  Magnitude Sweep Results (learned direction)\n{'='*60}")
    print(f"  Magnitude pairs used: {len(mag_directions)}")
    print(f"  Direction norm: {np.linalg.norm(mag_dir):.4f}")
    print(f"  Monotonic amplitude: {n_mono}/{len(sweep_results)} examples")
    print(f"  Amplitude ratio M8/M3: {np.mean(amp_ratios):.1f}×")

    if n_mono >= len(sweep_results) * 0.6:
        print(f"  ✓ Learned direction produces physically meaningful magnitude scaling.")
    else:
        print(f"  ~ Mixed — token[0] direction partially captures magnitude.")

    # ── Save ──────────────────────────────────────────────────────

    results = {
        "method": "learned_direction",
        "mag_pairs_n": len(mag_directions),
        "mag_direction": mag_dir.tolist(),
        "mag_direction_norm": float(np.linalg.norm(mag_dir)),
        "n_examples": len(sweep_results),
        "n_sweep": args.n_sweep,
        "monotonic_fraction": float(n_mono / max(len(sweep_results), 1)),
        "mean_amplitude_ratio": float(np.mean(amp_ratios)),
        "sweep_results": [{k: v for k, v in r.items() if k not in ("stats", "waveforms")}
                          for r in sweep_results],
        "stats_per_example": [r["stats"] for r in sweep_results],
    }

    with open(out / "results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save waveforms separately
    wf_dict = {}
    for ex_idx, r in enumerate(sweep_results):
        wf_dict[f"ex{ex_idx}"] = np.array(r["waveforms"], dtype=np.float32)
    np.savez_compressed(out / "sweep_waveforms_v2.npz", **wf_dict)

    print(f"\nSaved: {out}/{{results_v2.json, sweep_waveforms_v2.npz}}")


if __name__ == "__main__":
    import pandas as pd
    main()
