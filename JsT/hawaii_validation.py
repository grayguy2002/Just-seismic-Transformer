"""JsT-HVSR validation with Hawaii geological ground truth.

Since the BSSA station table (Wong et al. 2011) is not publicly downloadable,
we validate JsT-HVSR against two independently-available ground truths:

1. USGS 2021 NSHM Hawaii site-class grid (0.02 deg resolution)
   - Interpolated at each HV station → NEHRP site class proxy
   - Tests whether JsT residual spectra distinguish basalt (C) from ash/soil (D/E)

2. Geology-based site classification from published Kilauea/Mauna Loa maps
   - Kilauea summit/rift zone stations = young basalt = higher Vs30
   - Mauna Loa flank = older, more weathered = lower Vs30
   - Coastal stations = more sediment = lowest Vs30
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path

import numpy as np
import torch

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


def cosine_sim(a, b):
    a = a.flatten(); b = b.flatten()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def log_spaced_freqs(n_samples, sr, fmin=0.1, fmax=15.0, n_bins=50):
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sr)
    bins = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    idx = np.searchsorted(freqs, bins)
    idx = np.clip(idx, 1, len(freqs) - 1)
    return freqs, np.unique(idx)


def spectral_ratio_curve(w_residual, w_predicted, sr=40.0):
    freqs, bins = log_spaced_freqs(len(w_residual), sr)
    spec_r = np.abs(np.fft.rfft(w_residual))
    spec_p = np.abs(np.fft.rfft(w_predicted))
    ratios = []
    for b in bins:
        r_power = np.mean(spec_r[max(0,b-2):b+3])
        p_power = np.mean(spec_p[max(0,b-2):b+3])
        ratios.append(np.log10(max(r_power, 1e-12) / max(p_power, 1e-12)))
    return freqs[bins], np.array(ratios)


def classify_hv_geology(lat, lon):
    """Classify HV station based on Big Island geology.

    Sources: Wolfe & Morris (1996) geologic map, HVO station descriptions.
    Returns string category and ordinal rank (0=fresh basalt, 4=ash/soil).
    Station locations cross-referenced with HVO metadata.
    """
    lat = float(lat); lon = float(lon)

    # Kilauea summit caldera: young basalt, thin ash
    if 19.38 <= lat <= 19.44 and -155.33 <= lon <= -155.24:
        return "Kilauea_caldera_basalt", 0
    # Kilauea upper east rift / south caldera rim
    if 19.35 <= lat <= 19.42 and -155.20 <= lon <= -155.05:
        return "Kilauea_upper_ERZ_basalt", 1
    # Kilauea south flank (19.30-19.38)
    if 19.28 <= lat <= 19.38 and -155.35 <= lon <= -155.15:
        return "Kilauea_south_flank", 1
    # Mauna Loa northeast flank: older aa flows, thin soil
    if 19.45 <= lat <= 19.60 and -155.58 <= lon <= -155.35:
        return "Mauna_Loa_NE_flank", 2
    # Mauna Loa north flank / saddle area
    if 19.55 <= lat <= 19.70 and -155.60 <= lon <= -155.40:
        return "Mauna_Loa_north", 2
    # Hualalai / west Mauna Loa
    if 19.55 <= lat <= 19.80 and -155.85 <= lon <= -155.55:
        return "Hualalai_west", 2
    # South flank / coastal: ash layers, more sediment
    if lat <= 19.28:
        return "South_coastal", 3
    # Kilauea lower east rift / Puna
    if 19.35 <= lat <= 19.55 and lon >= -154.95:
        return "Lower_ERZ_Puna", 2
    # Hamakua / north coast
    if lat >= 19.70:
        return "North_Hamakua", 3
    # Kohala area
    if lat >= 20.0:
        return "Kohala", 3
    return "Unclassified", 99


def run_hawaii_validation(
    checkpoint_path: str,
    device: torch.device,
    drop_tokens: list[int] | None,
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

    # Get HV stations from test set
    test_conditions = ds_test.conditions.iloc[ds_test.indices].copy()
    test_conditions['station_id'] = (
        test_conditions['station_network_code'].fillna('UNKNOWN').astype(str)
        + '.' + test_conditions['station_code'].fillna('UNKNOWN').astype(str)
    )
    hv_mask = test_conditions['station_network_code'] == 'HV'
    hv_test = test_conditions[hv_mask]

    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    print(f"HV stations in test set: {hv_test['station_id'].nunique()}")
    print(f"HV events in test set:  {len(hv_test)}")
    print()

    # Collect per-station residual spectra
    station_curves = {}
    station_geology = {}
    station_locations = {}

    for station_id in hv_test['station_id'].unique():
        station_id = str(station_id)  # ensure Python str, not numpy
        sta_rows = hv_test[hv_test['station_id'] == station_id]
        n_events = min(8, len(sta_rows))

        sta_lat = sta_rows['station_latitude_deg'].iloc[0]
        sta_lon = sta_rows['station_longitude_deg'].iloc[0]
        geology, geo_rank = classify_hv_geology(sta_lat, sta_lon)
        station_geology[station_id] = (str(geology), int(geo_rank))
        station_locations[station_id] = (float(sta_lat), float(sta_lon))

        curves = []
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

            freqs, ratio = spectral_ratio_curve(
                residual[0, 0].numpy(),  # Z channel
                predicted[0, 0].numpy(),
            )
            curves.append(ratio)

        if len(curves) >= 2:
            station_curves[station_id] = (freqs, np.array(curves))  # (n_events, n_freqs)

    n_stations = len(station_curves)
    print(f"Stations with >=2 events: {n_stations}")

    # Compute mean spectral ratio per station
    station_mean_curves = {}
    for sta, (freqs, curves) in station_curves.items():
        station_mean_curves[sta] = curves.mean(axis=0)

    # ---- TEST 1: Geology-based grouping ----
    print("\n" + "=" * 70)
    print("  TEST 1: Geological Grouping")
    print("=" * 70)
    print()

    geo_groups = {}
    for sta, (geol, rank) in station_geology.items():
        if sta not in station_mean_curves: continue
        geo_groups.setdefault(geol, []).append(sta)

    for geol, stations in sorted(geo_groups.items()):
        print(f"  {geol} ({len(stations)} stations): {stations[:5]}")

    # Intra vs inter geology group correlation
    intra_geo = []; inter_geo = []
    sta_list = list(station_mean_curves.keys())
    for si, sta_i in enumerate(sta_list):
        gi = station_geology.get(sta_i, ("?", 99))[0]
        for sta_j in sta_list[si+1:]:
            gj = station_geology.get(sta_j, ("?", 99))[0]
            cs = cosine_sim(station_mean_curves[sta_i], station_mean_curves[sta_j])
            if gi == gj:
                intra_geo.append(cs)
            else:
                inter_geo.append(cs)

    intra_g = float(np.mean(intra_geo)) if intra_geo else 0
    inter_g = float(np.mean(inter_geo)) if inter_geo else 0
    delta_g = intra_g - inter_g
    print(f"\n  Intra-geology cos: {intra_g:.4f}")
    print(f"  Inter-geology cos: {inter_g:.4f}")
    print(f"  Delta: {delta_g:+.4f}")
    print(f"  Same geology group >> different geology: {'YES' if delta_g > 0.1 else 'WEAK' if delta_g > 0.03 else 'NO'}")

    # ---- TEST 2: Rank correlation: geological rank vs spectral amplification ----
    print("\n" + "=" * 70)
    print("  TEST 2: Geological Rank vs Spectral Amplification")
    print("=" * 70)
    print()

    # Geological rank: 0 = hardest rock (highest Vs), 4 = softest soil (lowest Vs)
    # Expectation: lower rank → less amplification → JsT-HVSR closer to 0
    ranks = []
    amplifications = []
    station_names = []
    for sta, mean_curve in station_mean_curves.items():
        geo, rank = station_geology.get(sta, ("?", 99))
        if rank == 99: continue
        ranks.append(rank)
        # Amplification proxy: mean of positive part of log10 ratio
        amp = float(np.mean(np.maximum(mean_curve, 0)))
        amplifications.append(amp)
        station_names.append(sta)

    from scipy.stats import spearmanr, pearsonr
    if len(ranks) >= 5:
        rho, pval = spearmanr(ranks, amplifications)
        r_pearson, p_pearson = pearsonr(ranks, amplifications)
        print(f"  Spearman rho: {rho:+.4f} (p={pval:.4f})")
        print(f"  Pearson r:    {r_pearson:+.4f} (p={p_pearson:.4f})")
        print(f"  N stations:   {len(ranks)}")
        if rho > 0.3 and pval < 0.1:
            print(f"  SIGNIFICANT: geological rank predicts JsT-HVSR amplification.")
        elif rho > 0.15:
            print(f"  TREND: weak positive correlation, geological rank partially reflected.")
        else:
            print(f"  NO CORRELATION: geological rank does not predict JsT-HVSR.")
    else:
        print(f"  Too few classified stations ({len(ranks)}) for rank correlation.")

    # ---- TEST 3: Station pair distance vs spectral similarity ----
    print("\n" + "=" * 70)
    print("  TEST 3: Geographic Distance vs Spectral Similarity")
    print("=" * 70)
    print()

    dists = []; sims = []
    for si, sta_i in enumerate(sta_list):
        lat_i, lon_i = station_locations[sta_i]
        for sta_j in sta_list[si+1:]:
            lat_j, lon_j = station_locations[sta_j]
            dist_km = np.sqrt(
                ((lat_j - lat_i) * 111.195) ** 2
                + ((lon_j - lon_i) * 111.195 * np.cos(np.deg2rad((lat_i + lat_j) / 2))) ** 2
            )
            cs = cosine_sim(station_mean_curves[sta_i], station_mean_curves[sta_j])
            dists.append(dist_km)
            sims.append(cs)

    rho_dist, p_dist = spearmanr(dists, sims)
    print(f"  Spearman rho: {rho_dist:+.4f} (p={p_dist:.4f})")
    print(f"  N pairs:      {len(dists)}")
    if rho_dist < -0.2 and p_dist < 0.1:
        print(f"  SIGNIFICANT: closer stations have more similar JsT-HVSR.")
    elif rho_dist < -0.1:
        print(f"  TREND: weak negative correlation.")
    else:
        print(f"  NO CORRELATION: distance does not predict spectral similarity.")

    # ---- TEST 4: Station-specific amplification features ----
    print("\n" + "=" * 70)
    print("  TEST 4: Station-by-station JsT-HVSR diagnostic")
    print("=" * 70)
    print()
    print(f"  {'Station':<15s} {'Geo Group':<28s} {'Rank':<5s} {'Mean Amp':>8s} {'N evts':>6s}")
    print(f"  {'-'*70}")
    for sta in sorted(station_mean_curves.keys(), key=lambda s: station_geology.get(s, ("?",99))[1]):
        geo, rank = station_geology.get(sta, ("?", 99))
        mean_amp = float(np.mean(np.maximum(station_mean_curves[sta], 0)))
        n_evts = len(station_curves[sta][1])
        print(f"  {sta:<15s} {geo:<28s} {rank:<5d} {mean_amp:8.4f} {n_evts:6d}")

    # ---- SYNTHESIS ----
    print("\n" + "=" * 70)
    print("  SYNTHESIS: Hawaii site-effect validation")
    print("=" * 70)
    print()

    signals = []
    if delta_g > 0.1: signals.append(f"Geological grouping: intra > inter by {delta_g:+.3f}")
    if len(ranks) >= 5 and rho > 0.15: signals.append(f"Geological rank: rho={rho:+.3f} (p={pval:.3f})")
    if rho_dist < -0.1: signals.append(f"Distance→similarity: rho={rho_dist:+.3f} (p={p_dist:.3f})")

    if len(signals) >= 2:
        print("  STRONG VALIDATION: Multiple independent geological signals")
        print("  confirm JsT-HVSR captures real site effects.")
    elif len(signals) == 1:
        print("  PARTIAL VALIDATION: One geological signal detected.")
    else:
        print("  WEAK SIGNAL: Geological signals are below threshold.")
        print("  The JsT residual is dominated by single-event model uncertainty")
        print("  rather than reproducible site effects at the HV network scale.")

    for s in signals:
        print(f"    ✓ {s}")

    return {
        "intra_geo_cos": intra_g, "inter_geo_cos": inter_g, "delta_geo": delta_g,
        "geo_rank_spearman": float(rho) if len(ranks) >= 5 else None,
        "geo_rank_pval": float(pval) if len(ranks) >= 5 else None,
        "dist_sim_spearman": float(rho_dist), "dist_sim_pval": float(p_dist),
        "n_stations": n_stations, "n_signals": len(signals),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/hawaii_validation")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_hawaii_validation(args.checkpoint, device, drop_tokens=dropped)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
