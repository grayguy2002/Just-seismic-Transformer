"""Fig 1 compute — GPU inference only. Saves cache for local rendering.

Outputs:
  outputs/fig1_cache/station_hvsr.json     — per-station JsT-HVSR vectors + metadata
  outputs/fig1_cache/td_residuals.npz      — flattened time-domain residuals + labels
  outputs/fig1_cache/stats.json            — intra/inter Pearson statistics
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


N_FREQ_BINS = 40
F_MIN, F_MAX = 0.3, 15.0
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)


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


def compute_hvsr(residual: np.ndarray, predicted: np.ndarray, eps: float = 1e-12):
    per_comp = []
    for ch in range(3):
        spec_r = np.abs(np.fft.rfft(residual[ch]))
        spec_p = np.abs(np.fft.rfft(predicted[ch]))
        freqs = np.fft.rfftfreq(residual.shape[1], d=1 / 100.0)
        binned_r = np.zeros(N_FREQ_BINS)
        binned_p = np.zeros(N_FREQ_BINS)
        for b in range(N_FREQ_BINS):
            mask = (freqs >= FREQ_EDGES[b]) & (freqs < FREQ_EDGES[b + 1])
            if mask.any():
                binned_r[b] = np.mean(spec_r[mask])
                binned_p[b] = np.mean(spec_p[mask])
        ratio = np.log10(np.maximum(binned_r, eps) / np.maximum(binned_p, eps))
        per_comp.append(ratio)
    return np.mean(per_comp, axis=0)


def compute_pairwise_stats(curves_flat, labels, rng_seed=42):
    from scipy.stats import pearsonr
    n = len(curves_flat)
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r_val = pearsonr(curves_flat[i], curves_flat[j])[0]
            if labels[i] == labels[j]:
                intra.append(r_val)
            else:
                inter.append(r_val)

    mi = float(np.mean(intra)) if intra else 0.0
    mj = float(np.mean(inter)) if inter else 0.0

    rng = np.random.default_rng(rng_seed)
    shuffled = rng.permutation(labels)
    null_intra, null_inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            r_val = pearsonr(curves_flat[i], curves_flat[j])[0]
            if shuffled[i] == shuffled[j]:
                null_intra.append(r_val)
            else:
                null_inter.append(r_val)

    return {
        "intra": mi, "inter": mj, "delta": mi - mj,
        "ratio": mi / max(mj, 1e-12),
        "intra_vals": [float(v) for v in intra],
        "inter_vals": [float(v) for v in inter],
        "null_intra": float(np.mean(null_intra)) if null_intra else 0.0,
        "null_inter": float(np.mean(null_inter)) if null_inter else 0.0,
        "null_delta": float(np.mean(null_intra) - np.mean(null_inter)),
        "null_intra_vals": [float(v) for v in null_intra],
        "null_inter_vals": [float(v) for v in null_inter],
    }


def classify_geology(lat: float, lon: float, network: str) -> str:
    lat, lon = float(lat), float(lon)
    network = str(network)
    if network == "HV":
        if 19.38 <= lat <= 19.45 and -155.35 <= lon <= -155.22:
            return "Basalt_Kilauea"
        if lat <= 19.28:    return "Basalt_coastal"
        if lat >= 19.50:    return "Basalt_weathered"
        return "Basalt_flank"
    if network == "AK":
        if lat >= 64.0:     return "Metamorphic_interior"
        if lat <= 60.0:     return "Sedimentary_coastal"
        return "Metamorphic_range"
    if network == "AT":
        return "Volcanic_arc" if lon <= -160 else "Sedimentary_coastal"
    if network == "OK":        return "Sedimentary_basin"
    if network == "GS":        return "Sedimentary_centralUS"
    if network == "UU":        return "Basin_range"
    if network == "UW":        return "Volcanic_cascades"
    if network == "NN":        return "Basin_range"
    if network in ("CI","NC"): return "Active_margin"
    if network == "NM":        return "Sedimentary_embayment"
    if lat >= 48.0:            return "Craton_north"
    if lat <= 35.0 and lon <= -100: return "Active_margin"
    if lat <= 38.0 and lon <= -88:  return "Sedimentary_embayment"
    if lon >= -75:             return "Passive_margin_east"
    return "Interior_platform"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--output-dir", default="outputs/fig1_cache")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--max-stations", type=int, default=150)
    parser.add_argument("--max-events-per-station", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    print("Loading checkpoint...")
    ce, dn, ckpt = load_checkpoint_models(
        args.checkpoint, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if dropped:
        ce = AblationConditionEncoder(ce, dropped)
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

    sta_counts = test_conditions["station_id"].value_counts()
    qualified = sorted(
        [s for s in sta_counts[sta_counts >= args.min_events].index],
        key=lambda s: sta_counts[s], reverse=True,
    )
    n_stations = min(args.max_stations, len(qualified))
    selected = qualified[:n_stations]
    print(f"Stations: {n_stations} (range {sta_counts[selected[-1]]}–{sta_counts[selected[0]]} evts)")

    # ── GPU inference ──────────────────────────────────────────────

    station_hvsr = defaultdict(list)
    station_residuals_flat = defaultdict(list)
    station_locations = {}
    all_residuals_flat = []
    all_station_labels = []
    label_idx = 0

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)

    n_total = 0
    for sta_id in selected:
        sta_rows = test_conditions[test_conditions["station_id"] == sta_id]
        n_events = min(args.max_events_per_station, len(sta_rows))
        lat = float(sta_rows["station_latitude_deg"].iloc[0])
        lon = float(sta_rows["station_longitude_deg"].iloc[0])
        net = str(sta_rows["station_network_code"].iloc[0])
        station_locations[sta_id] = (lat, lon, net)

        curves, residuals = [], []
        for row_idx in sta_rows.index[:n_events]:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            wf_tensor, cond_dict = ds_test[didx]
            wf_np = wf_tensor.numpy()
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)
            predicted = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu().numpy()[0]
            residual = wf_np - predicted
            hvsr = compute_hvsr(residual, predicted)
            curves.append(hvsr.tolist())
            residuals.append(residual.ravel().tolist())
            all_residuals_flat.append(residual.ravel())
            all_station_labels.append(label_idx)
            n_total += 1

        if curves:
            geol = classify_geology(*station_locations[sta_id])
            station_hvsr[sta_id] = {
                "curves": curves,
                "lat": lat, "lon": lon, "net": net,
                "geology": geol,
                "n_events": len(curves),
            }
            station_residuals_flat[sta_id] = residuals
            label_idx += 1

    print(f"Total: {n_total} JsT-HVSR curves across {label_idx} stations")

    # ── Stats ──────────────────────────────────────────────────────

    print("Computing pairwise statistics...")
    td_stats = compute_pairwise_stats(all_residuals_flat, all_station_labels)
    print(f"  intra={td_stats['intra']:.4f}  inter={td_stats['inter']:.4f}  "
          f"delta={td_stats['delta']:+.4f}  ratio={td_stats['ratio']:.1f}×  "
          f"null={td_stats['null_delta']:+.4f}")

    # ── Save cache ─────────────────────────────────────────────────

    with open(output_dir / "station_hvsr.json", "w") as f:
        json.dump(station_hvsr, f, indent=1)
    print(f"Saved: {output_dir / 'station_hvsr.json'}")

    # Td residuals as compressed numpy archive (smaller than JSON for 150*8*9600 floats)
    np.savez_compressed(
        output_dir / "td_residuals.npz",
        residuals=np.array([r.tolist() for r in all_residuals_flat], dtype=np.float32),
        labels=np.array(all_station_labels, dtype=np.int16),
    )
    print(f"Saved: {output_dir / 'td_residuals.npz'}")

    with open(output_dir / "stats.json", "w") as f:
        json.dump(td_stats, f, indent=1)
    print(f"Saved: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
