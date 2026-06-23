"""Fig 2 compute — GPU inference for Hawaii panel (c) only.

Panels (a) and (b) read pre-computed CSV/JSON (no GPU needed).
Cache file is used by fig2_render.py for local iteration.

Outputs:
  outputs/fig2_cache/hawaii_pairs.json  — 300 station pairs with distance + cos_sim
  outputs/fig2_cache/hawaii_stats.json  — rho, p_value, N_stations
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


# ── matched to hawaii_validation.py (Z-channel, sr=40, 50 bins) ─────


def log_spaced_freqs(n_samples, sr, fmin=0.1, fmax=15.0, n_bins=50):
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
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
        r_power = np.mean(spec_r[max(0, b - 2):b + 3])
        p_power = np.mean(spec_p[max(0, b - 2):b + 3])
        ratios.append(np.log10(max(r_power, 1e-12) / max(p_power, 1e-12)))
    return freqs[bins], np.array(ratios)


def cosine_sim(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--output-dir", default="outputs/fig2_cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Loading checkpoint...")
    ce, dn, ckpt = load_checkpoint_models(
        args.checkpoint, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    ce = AblationConditionEncoder(ce, [8, 9, 10])
    dn.eval(); ce.eval()
    total_samples = 3200

    ds_train = SeismicWaveformDataset(
        args.data_dir, split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        args.data_dir, split="testing", augment=False, vocab_from=ds_train,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    test_conditions = ds_test.conditions.iloc[ds_test.indices].copy()
    test_conditions["station_id"] = (
        test_conditions["station_network_code"].fillna("UNKNOWN").astype(str)
        + "." + test_conditions["station_code"].fillna("UNKNOWN").astype(str)
    )
    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    hv_mask = test_conditions["station_network_code"] == "HV"
    hawaii = test_conditions[hv_mask]
    print(f"Hawaii events: {len(hawaii)}")

    @torch.no_grad()
    def gen(tokens):
        net = dn.net
        ts = torch.linspace(0.0, 1.0, 50 + 1, device=device)
        z = noise_fixed.clone()
        for i in range(50):
            t = ts[i]; t_next = ts[i + 1]
            xp = net(z, t.expand(z.shape[0]), tokens)
            v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
            z = z + (t_next - t) * v
        return z

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)

    # Per-station mean JsT-HVSR (Z-channel)
    station_data = {}
    for sta_id, group in hawaii.groupby("station_id"):
        sta_id = str(sta_id)
        n_events = min(8, len(group))
        lat = float(group["station_latitude_deg"].iloc[0])
        lon = float(group["station_longitude_deg"].iloc[0])
        curves = []
        for row_idx in group.index[:n_events]:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            wf_tensor, cond_dict = ds_test[didx]
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)
            predicted = gen(tokens).cpu().numpy()[0]
            residual = wf_tensor.numpy() - predicted
            _, ratio = spectral_ratio_curve(residual[0], predicted[0])
            curves.append(ratio)
        if len(curves) >= 2:
            station_data[sta_id] = {
                "mean_curve": np.mean(curves, axis=0).tolist(),
                "lat": lat, "lon": lon, "n_events": len(curves),
            }

    print(f"Stations with >=2 events: {len(station_data)}")

    # All pairwise distances + similarities
    sta_ids = sorted(station_data.keys())
    pairs = []
    for i in range(len(sta_ids)):
        lat_i, lon_i = station_data[sta_ids[i]]["lat"], station_data[sta_ids[i]]["lon"]
        for j in range(i + 1, len(sta_ids)):
            lat_j, lon_j = station_data[sta_ids[j]]["lat"], station_data[sta_ids[j]]["lon"]
            d = np.sqrt(((lat_j - lat_i) * 111.195) ** 2
                        + ((lon_j - lon_i) * 111.195
                           * np.cos(np.deg2rad((lat_i + lat_j) / 2))) ** 2)
            cs = cosine_sim(station_data[sta_ids[i]]["mean_curve"],
                            station_data[sta_ids[j]]["mean_curve"])
            pairs.append({
                "sta1": sta_ids[i], "sta2": sta_ids[j],
                "distance_km": float(d), "cos_sim": float(cs),
            })

    from scipy.stats import spearmanr
    distances = np.array([p["distance_km"] for p in pairs])
    similarities = np.array([p["cos_sim"] for p in pairs])
    rho, pval = spearmanr(distances, similarities)

    print(f"Hawaii ρ={rho:+.4f}  p={pval:.2e}  N={len(pairs)}")

    with open(output_dir / "hawaii_pairs.json", "w") as f:
        json.dump({"pairs": pairs}, f, indent=1)
    with open(output_dir / "hawaii_stats.json", "w") as f:
        json.dump({
            "spearman_rho": float(rho), "p_value": float(pval),
            "n_pairs": len(pairs), "n_stations": len(station_data),
        }, f, indent=1)
    print(f"Saved: {output_dir}/hawaii_{{pairs,stats}}.json")


if __name__ == "__main__":
    main()
