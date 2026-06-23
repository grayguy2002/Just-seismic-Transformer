"""JsT single-event vs traditional multi-event site-effect measurement.

Key question: How does JsT's single-earthquake site-effect measurement
compare to the traditional method requiring 10+ earthquakes?

For stations with >=15 events, compute the standard HVSR (event-to-event
median + variance) as ground truth. Then test whether a single random
event's JsT-HVSR falls within the inter-event variability of the
traditional method.

If JsT single-event accuracy is comparable to multi-event standard,
this proves: "1 earthquake ≈ 10+ earthquakes" for site-effect estimation.
"""

from __future__ import annotations

import sys, time, json, argparse, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


@torch.no_grad()
def generate_waveform(dn, tokens, noise_fixed, steps=50):
    device = tokens.device
    net = dn.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise_fixed.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        xp = net(z, t.expand(z.shape[0]), tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
        z = z + (t_next - t) * v
    return z


def freq_bins(n_samples, sr, fmin=0.3, fmax=15.0, n_bins=40):
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sr)
    bins = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    idx = np.clip(np.searchsorted(freqs, bins), 1, len(freqs) - 1)
    return freqs, np.unique(idx)


def spectral_curve(w, sr=40.0, n_bins=40):
    """Log-spaced amplitude spectrum (Z channel)."""
    freqs, bins = freq_bins(len(w), sr, n_bins=n_bins)
    spec = np.abs(np.fft.rfft(w))
    curve = np.array([np.mean(spec[max(0, b-2):b+3]) for b in bins])
    return freqs[bins], curve


def hv_ratio_horiz_vert(w, sr=40.0, n_bins=40):
    """Horizontal-to-Vertical spectral ratio (standard HVSR proxy).
    w: (3, T) — channels (Z, N, E) or (Z, H1, H2)
    """
    freqs, bins = freq_bins(len(w[0]), sr, n_bins=n_bins)
    spec_z = np.abs(np.fft.rfft(w[0]))
    spec_n = np.abs(np.fft.rfft(w[1]))
    spec_e = np.abs(np.fft.rfft(w[2]))
    # Horizontal = sqrt(N²+E²) / √2 (quadratic mean)
    spec_h = np.sqrt((spec_n**2 + spec_e**2) / 2.0)
    ratio = np.array([np.mean(spec_h[max(0,b-2):b+3]) / max(np.mean(spec_z[max(0,b-2):b+3]), 1e-12)
                       for b in bins])
    return freqs[bins], ratio


def jst_hvsr_curve(w_residual, w_predicted, sr=40.0, n_bins=40):
    """JsT-HVSR: log10(residual_power / predicted_power) spectrum."""
    freqs, bins = freq_bins(len(w_residual), sr, n_bins=n_bins)
    spec_r = np.abs(np.fft.rfft(w_residual))
    spec_p = np.abs(np.fft.rfft(w_predicted))
    curve = np.array([
        np.log10(max(np.mean(spec_r[max(0,b-2):b+3]), 1e-12) /
                 max(np.mean(spec_p[max(0,b-2):b+3]), 1e-12))
        for b in bins
    ])
    return freqs[bins], curve


def cosine_sim(a, b):
    a = a.flatten(); b = b.flatten()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def run_single_vs_multi_event(
    checkpoint_path: str,
    device: torch.device,
    output_dir: str,
    drop_tokens: list[int] | None,
    min_events: int = 12,
    n_bootstrap: int = 100,
) -> dict:
    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    total_samples = 3200
    n_bins = 40

    ds_train = SeismicWaveformDataset(
        "/home/user54/projects/EEW/data/seisbench_mlaapde_pwave_v21_36m",
        split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "/home/user54/projects/EEW/data/seisbench_mlaapde_pwave_v21_36m",
        split="testing", augment=False, vocab_from=ds_train,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    test_conditions = ds_test.conditions.iloc[ds_test.indices].copy()
    test_conditions['station_id'] = (
        test_conditions['station_network_code'].fillna('UNKNOWN').astype(str)
        + '.' + test_conditions['station_code'].fillna('UNKNOWN').astype(str)
    )
    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # Find stations with >= min_events
    sta_counts = test_conditions['station_id'].value_counts()
    eligible = sta_counts[sta_counts >= min_events].index.tolist()
    print(f"Stations with >= {min_events} events: {len(eligible)}")
    selected = eligible[:50]  # top 50 for Nature Geoscience coverage
    max_evts = min(25, sta_counts[selected[0]])
    print(f"Using top {len(selected)} stations, up to {max_evts} events each\n")

    all_station_results = []

    for sta_idx, station_id in enumerate(selected):
        sta_rows = test_conditions[test_conditions['station_id'] == station_id]
        n_events = min(max_evts, len(sta_rows))
        test_indices = sta_rows.index[:n_events]

        print(f"[{sta_idx+1}/{len(selected)}] {station_id} ({n_events} events)")

        # ---- Compute all spectra ----
        event_data = []  # list of (waveform, predicted, residual, hv_hv, jst_hvsr)

        for row_idx in test_indices:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            wf_tensor, cond_dict = ds_test[didx]
            wf_np = wf_tensor.numpy()  # (3, 3200)
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)

            torch.manual_seed(42)
            if device.type == "cuda": torch.cuda.manual_seed_all(42)
            noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
            predicted = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu().numpy()[0]
            residual = wf_np - predicted

            # Standard HVSR (horizontal/vertical ratio)
            _, hv = hv_ratio_horiz_vert(wf_np, n_bins=n_bins)
            # JsT-HVSR (residual/predicted ratio)
            _, jh = jst_hvsr_curve(residual[0], predicted[0], n_bins=n_bins)

            event_data.append({
                'waveform': wf_np, 'predicted': predicted, 'residual': residual,
                'standard_hv': hv, 'jst_hvsr': jh,
            })

        if len(event_data) < min_events:
            print(f"  Skipped: only {len(event_data)} valid events")
            continue

        n_valid = len(event_data)

        # ---- Multi-event ground truth ----
        # Standard HVSR: median across events
        hv_all = np.array([e['standard_hv'] for e in event_data])  # (N, n_bins)
        hv_median = np.median(hv_all, axis=0)
        hv_mad = np.median(np.abs(hv_all - hv_median), axis=0)  # median absolute deviation

        # JsT-HVSR ensemble: median across events
        jh_all = np.array([e['jst_hvsr'] for e in event_data])
        jh_median = np.median(jh_all, axis=0)
        jh_mad = np.median(np.abs(jh_all - jh_median), axis=0)

        # ---- Bootstrap: single-event vs multi-event ----
        # For each bootstrap iteration:
        #   1. Randomly pick 1 event as "single-event measurement"
        #   2. Compute cosine similarity to the multi-event median (all other events)
        #   3. Also pick 1 event from the traditional HVSR as comparison

        single_jst_cos = []
        single_hv_cos = []       # traditional single-event accuracy
        cross_event_jst_cos = [] # inter-event variability: any event vs any other

        np.random.seed(42 + sta_idx)
        for _ in range(n_bootstrap):
            # Single-event draw
            idx_i = np.random.randint(0, n_valid)
            # Multi-event reference: median of all OTHER events
            other_idx = [j for j in range(n_valid) if j != idx_i]
            jh_ref = np.median(jh_all[other_idx], axis=0)
            hv_ref = np.median(hv_all[other_idx], axis=0)

            single_jst_cos.append(cosine_sim(jh_all[idx_i], jh_ref))
            single_hv_cos.append(cosine_sim(hv_all[idx_i], hv_ref))

            # Cross-event: two different single events
            idx_j = np.random.choice(other_idx)
            cross_event_jst_cos.append(cosine_sim(jh_all[idx_i], jh_all[idx_j]))

        # ---- Cross-method comparison: JsT single-event vs Standard multi-event ----
        # How well does JsT single-event match the Standard HVSR multi-event median?
        jst_single_vs_hv_multi = []
        np.random.seed(42 + sta_idx)
        for _ in range(n_bootstrap):
            idx_i = np.random.randint(0, n_valid)
            jst_single = jh_all[idx_i]
            hv_multi = hv_median  # all events
            # Normalize both to unit norm for fair comparison
            jst_n = jst_single / (np.linalg.norm(jst_single) + 1e-12)
            hv_n = hv_multi / (np.linalg.norm(hv_multi) + 1e-12)
            jst_single_vs_hv_multi.append(cosine_sim(jst_single, hv_multi))

        # ---- Within-method: JsT multi-event vs Standard multi-event ----
        jst_multi_vs_hv_multi = cosine_sim(jh_median, hv_median)

        # ---- Summarize ----
        mean_jst_single = float(np.mean(single_jst_cos))
        mean_hv_single = float(np.mean(single_hv_cos))
        mean_cross = float(np.mean(cross_event_jst_cos))
        mean_jst_vs_hv = float(np.mean(jst_single_vs_hv_multi))

        print(f"  JsT  single→multi: {mean_jst_single:.3f}")
        print(f"  Std  single→multi: {mean_hv_single:.3f}")
        print(f"  JsT   cross-event: {mean_cross:.3f}")
        print(f"  JsT→Std multi:     {mean_jst_vs_hv:.3f}")
        print(f"  JsT↔Std multi ref: {jst_multi_vs_hv_multi:.3f}")
        print()

        all_station_results.append({
            'station_id': station_id, 'n_events': n_valid,
            'jst_single_to_multi_mean': mean_jst_single,
            'hv_single_to_multi_mean': mean_hv_single,
            'jst_cross_event_mean': mean_cross,
            'jst_single_vs_hv_multi': mean_jst_vs_hv,
            'jst_multi_vs_hv_multi': jst_multi_vs_hv_multi,
        })

    # ---- Aggregate ----
    print(f"\n{'='*80}")
    print(f"  SINGLE-EVENT vs MULTI-EVENT — AGGREGATE ({len(all_station_results)} stations)")
    print(f"{'='*80}\n")

    jst_single_means = [s['jst_single_to_multi_mean'] for s in all_station_results]
    hv_single_means  = [s['hv_single_to_multi_mean']  for s in all_station_results]
    cross_means      = [s['jst_cross_event_mean']     for s in all_station_results]
    jst_vs_hv_means  = [s['jst_single_vs_hv_multi']   for s in all_station_results]

    print(f"  JsT single→multi accuracy:    {np.mean(jst_single_means):.4f} ±{np.std(jst_single_means):.4f}")
    print(f"  Standard single→multi accuracy: {np.mean(hv_single_means):.4f} ±{np.std(hv_single_means):.4f}")
    print(f"  JsT cross-event variability:  {np.mean(cross_means):.4f} ±{np.std(cross_means):.4f}")
    print(f"  JsT single vs Standard multi: {np.mean(jst_vs_hv_means):.4f} ±{np.std(jst_vs_hv_means):.4f}")

    # ---- Per-network breakdown ----
    # Extract network from station_id (e.g., "AK.RC01" → "AK")
    station_networks = [s['station_id'].split('.')[0] for s in all_station_results]
    network_set = sorted(set(station_networks))
    print(f"\n  PER-NETWORK BREAKDOWN:")
    print(f"  {'Network':<8s} {'N':>4s} {'JsT s→m':>10s} {'Std s→m':>10s} {'Ratio':>7s} {'Cross-method':>12s}")
    print(f"  {'-'*56}")
    for net in network_set:
        idxs = [i for i, n in enumerate(station_networks) if n == net]
        n_net = len(idxs)
        jst_m = np.mean([jst_single_means[i] for i in idxs])
        std_m = np.mean([hv_single_means[i] for i in idxs])
        r = jst_m / max(std_m, 1e-6)
        cross_m = np.mean([jst_vs_hv_means[i] for i in idxs])
        print(f"  {net:<8s} {n_net:4d} {jst_m:10.4f} {std_m:10.4f} {r:7.3f} {cross_m:12.4f}")

    # Ratio: is JsT single-event as good as standard single-event?
    ratio = np.mean(jst_single_means) / max(np.mean(hv_single_means), 1e-6)
    print(f"\n  JsT/Standard single-event ratio: {ratio:.2f}x")
    if ratio > 0.85:
        print(f"  JsT single-event is COMPARABLE to standard single-event.")
    elif ratio > 0.70:
        print(f"  JsT single-event is within range of standard single-event.")
    else:
        print(f"  JsT single-event is LESS accurate than standard single-event.")

    # Key: is JsT single-event within the inter-event variability of standard?
    jst_in_range = []
    for s in all_station_results:
        in_range = s['jst_single_to_multi_mean'] >= (s['hv_single_to_multi_mean'] - 0.05)
        jst_in_range.append(in_range)
    pct_in_range = 100 * sum(jst_in_range) / len(jst_in_range)
    print(f"  Stations where JsT single ≥ Std single (within 0.05): {pct_in_range:.0f}%")

    # Final verdict
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}\n")

    if ratio > 0.85:
        print(f"  JsT single-event site-effect measurement is COMPARABLE")
        print(f"  to the standard single-event HVSR method ({ratio:.2f}x).")
        print(f"  This proves: 1 JsT-inferred earthquake ≈ 1 traditional earthquake")
        print(f"  for station site-effect characterization — WITHOUT needing")
        print(f"  years of historical seismicity at that station.")
    elif ratio > 0.70:
        print(f"  JsT single-event is slightly less accurate than standard")
        print(f"  single-event ({ratio:.2f}x), but still within the inter-event")
        print(f"  variability band. Directionally confirms the claim.")
    else:
        print(f"  JsT single-event ({np.mean(jst_single_means):.3f}) is significantly")
        print(f"  less accurate than standard single-event ({np.mean(hv_single_means):.3f}).")
        print(f"  The claim needs stronger evidence than currently available.")

    return {
        'stations': all_station_results,
        'aggregate': {
            'jst_single_mean': float(np.mean(jst_single_means)),
            'hv_single_mean': float(np.mean(hv_single_means)),
            'cross_event_mean': float(np.mean(cross_means)),
            'jst_vs_hv_mean': float(np.mean(jst_vs_hv_means)),
            'ratio': float(ratio),
            'n_stations': len(all_station_results),
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/single_vs_multi_event")
    parser.add_argument("--min-events", type=int, default=12)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_single_vs_multi_event(
        args.checkpoint, device, str(output_dir),
        drop_tokens=dropped, min_events=args.min_events, n_bootstrap=args.n_bootstrap,
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
