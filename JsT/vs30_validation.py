"""Comprehensive Vs30 validation for JsT-HVSR across all matched stations.

Uses the USGS Seismic Station Compilation Vs30 matches produced by Codex
to test whether JsT-HVSR amplification correlates with measured Vs30
across multiple networks and geological settings.

Key test: Spearman ρ(JsT-HVSR mean amplification, measured Vs30)
Expected: ρ < 0 (higher Vs30 = harder rock = LESS JsT amplification)
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
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


def jst_hvsr_amplification(residual, predicted, sr=40.0, fmin=0.3, fmax=15.0, n_bins=40):
    """Compute mean log10(residual_power/predicted_power) in log-spaced freq bins."""
    freqs_r = np.fft.rfftfreq(len(residual), d=1.0/sr)
    spec_r = np.abs(np.fft.rfft(residual))
    spec_p = np.abs(np.fft.rfft(predicted))
    bins = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    idx = np.clip(np.searchsorted(freqs_r, bins), 1, len(freqs_r)-1)
    idx = np.unique(idx)
    amps = []
    for b in idx:
        rp = np.mean(spec_r[max(0,b-2):b+3])
        pp = np.mean(spec_p[max(0,b-2):b+3])
        amps.append(np.log10(max(rp, 1e-12) / max(pp, 1e-12)))
    return float(np.mean(np.maximum(amps, 0)))  # positive amplification only


def run_vs30_validation(
    checkpoint_path: str,
    device: torch.device,
    vs30_csv: str,
    output_dir: str,
    drop_tokens: list[int] | None,
    max_events_per_station: int = 6,
    min_events_per_station: int = 3,
) -> dict:
    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    total_samples = 3200

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

    # Load Vs30 match table
    vs30_df = pd.read_csv(vs30_csv)
    print(f"Loaded {len(vs30_df)} Vs30 matches")

    # Build test set station index
    test_conditions = ds_test.conditions.iloc[ds_test.indices].copy()
    test_conditions['station_id'] = (
        test_conditions['station_network_code'].fillna('UNKNOWN').astype(str)
        + '.' + test_conditions['station_code'].fillna('UNKNOWN').astype(str)
    )
    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # Group by station and vs30_kind
    results = []
    matched_stations = set(vs30_df['station_id'].tolist())

    for station_id in sorted(matched_stations):
        # Get Vs30 info
        vs30_rows = vs30_df[vs30_df['station_id'] == station_id]
        vs30_val = float(vs30_rows['vs30_m_s'].iloc[0])
        vs30_kind = str(vs30_rows['vs30_kind'].iloc[0])
        vs30_src = str(vs30_rows['vs30_source'].iloc[0])
        event_count = int(vs30_rows['event_count'].iloc[0])

        if event_count < min_events_per_station:
            continue

        # Get test set rows for this station
        sta_rows = test_conditions[test_conditions['station_id'] == station_id]
        n_events = min(max_events_per_station, len(sta_rows))

        amplifications = []
        n_success = 0

        for row_idx in sta_rows.index[:n_events]:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            wf_tensor, cond_dict = ds_test[didx]
            wf_cpu = wf_tensor.unsqueeze(0)
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)

            torch.manual_seed(42)
            if device.type == "cuda": torch.cuda.manual_seed_all(42)
            noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
            predicted = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu()
            residual = wf_cpu - predicted

            # JsT-HVSR on Z channel
            amp = jst_hvsr_amplification(
                residual[0, 0].numpy(), predicted[0, 0].numpy()
            )
            amplifications.append(amp)
            n_success += 1

        if n_success >= min_events_per_station:
            mean_amp = float(np.mean(amplifications))
            std_amp = float(np.std(amplifications))
            results.append({
                'station_id': station_id, 'vs30': vs30_val,
                'vs30_kind': vs30_kind, 'vs30_source': vs30_src,
                'n_events': n_success, 'mean_amp': mean_amp,
                'std_amp': std_amp,
            })

    print(f"Stations with >= {min_events_per_station} events: {len(results)}")

    # ---- Rank correlation: measured/profile vs all ----
    df_all = pd.DataFrame(results)

    # Split by data quality
    measured = df_all[df_all['vs30_kind'] == 'measured_or_profile_database'].copy()
    proxy = df_all[df_all['vs30_kind'] == 'proxy'].copy()
    ngasub = df_all[df_all['vs30_kind'] == 'selected_for_gmm_may_be_inferred'].copy()

    print(f"\n  Data quality breakdown:")
    print(f"    measured/profile:  {len(measured)} stations")
    print(f"    proxy (mosaic/etc): {len(proxy)} stations")
    print(f"    NGA-sub selected:   {len(ngasub)} stations")

    correlations = {}
    for label, subset in [('ALL', df_all), ('measured_profile', measured),
                           ('proxy', proxy), ('nga_sub', ngasub)]:
        if len(subset) < 5:
            correlations[label] = {'rho': None, 'p': None, 'r': None, 'r_p': None, 'N': len(subset)}
            continue
        rho, p = spearmanr(subset['vs30'], subset['mean_amp'])
        r_pearson, p_pearson = pearsonr(subset['vs30'], subset['mean_amp'])
        correlations[label] = {'rho': float(rho), 'p': float(p),
                               'r': float(r_pearson), 'r_p': float(p_pearson),
                               'N': len(subset)}
        print(f"  {label:20s}: N={len(subset):3d}  Spearman rho={rho:+.4f} (p={p:.3f})  Pearson r={r_pearson:+.4f}")

    # ---- Save detailed per-station table ----
    df_all.to_csv(f"{output_dir}/jst_hvsr_vs_vs30_results.csv", index=False)

    # ---- VERDICT ----
    print(f"\n{'='*70}")
    print(f"  CROSS-NETWORK Vs30 VALIDATION — VERDICT")
    print(f"{'='*70}\n")

    if measured_rho := correlations.get('measured_profile', {}).get('rho'):
        meas_p = correlations['measured_profile']['p']
        if abs(measured_rho) > 0.3 and meas_p < 0.05:
            print(f"  STRONG: JsT-HVSR amplification correlats with measured Vs30")
            print(f"  (rho={measured_rho:+.3f}, p={meas_p:.3f}, N={correlations['measured_profile']['N']})")
        elif abs(measured_rho) > 0.2:
            print(f"  TREND: JsT-HVSR correlates with measured Vs30")
            print(f"  (rho={measured_rho:+.3f}, p={meas_p:.3f}, N={correlations['measured_profile']['N']})")
        else:
            print(f"  WEAK: No measurable correlation with existing measured stations")
    else:
        print(f"  INSUFFICIENT: Need >= 5 measured stations for correlation test")

    if proxy_rho := correlations.get('proxy', {}).get('rho'):
        print(f"  Proxy-based Vs30: rho={proxy_rho:+.3f} (N={correlations['proxy']['N']})")

    return {'correlations': correlations, 'n_stations': len(results)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vs30-csv", required=True, help="Codex-produced standardized match CSV")
    parser.add_argument("--output-dir", default="outputs/vs30_validation")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = run_vs30_validation(
        args.checkpoint, device, args.vs30_csv, output_dir, drop_tokens=dropped,
    )

    with open(f"{output_dir}/correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/")
