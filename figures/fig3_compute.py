"""Fig 3 compute — Token 7 embedding extraction + PCA + cosine stats.

GPU: token extraction only (fast, no waveform generation).
CPU: PCA, intra/inter group cosine similarity.

Outputs:
  outputs/fig3_cache/pca_projection.npz    — X_pca_2d coords, station_ids, geo_labels
  outputs/fig3_cache/intra_summary.json    — per-group cosine stats
"""

from __future__ import annotations

import sys, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from JsT import SeismicWaveformDataset, load_checkpoint_models
from JsT.ablation import AblationConditionEncoder


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


def cosine_sim(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--output-dir", default="outputs/fig3_cache")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    print("Loading checkpoint (token extraction only)...")
    ce, dn, ckpt = load_checkpoint_models(
        args.checkpoint, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if dropped:
        ce = AblationConditionEncoder(ce, dropped)
    ce.eval()

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

    # ── Extract Token 7 ──────────────────────────────────────────

    print("Extracting Token 7 embeddings...")
    station_tokens = {}
    station_locations = {}

    for sta_id in test_conditions["station_id"].unique():
        sta_rows = test_conditions[test_conditions["station_id"] == sta_id]
        n_evts = min(6, len(sta_rows))
        lat = float(sta_rows["station_latitude_deg"].iloc[0])
        lon = float(sta_rows["station_longitude_deg"].iloc[0])
        net = str(sta_rows["station_network_code"].iloc[0])
        station_locations[sta_id] = (lat, lon, net)

        tokens_list = []
        for row_idx in sta_rows.index[:n_evts]:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            _, cond_dict = ds_test[didx]
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            with torch.no_grad():
                tok = ce(cond_gpu).cpu()
            tokens_list.append(tok[0, 7, :].numpy())
        if len(tokens_list) >= 2:
            station_tokens[sta_id] = np.array(tokens_list)

    n_stations = len(station_tokens)
    station_ids = sorted(station_tokens.keys())
    X_raw = np.array([station_tokens[s].mean(axis=0) for s in station_ids])
    geo_labels = [classify_geology(*station_locations[s]) for s in station_ids]
    geo_set = sorted(set(geo_labels))

    print(f"Stations: {n_stations}, Token 7 dim: {X_raw.shape[1]}, groups: {len(geo_set)}")
    for g in geo_set:
        print(f"  {g}: {sum(1 for l in geo_labels if l == g)}")

    # ── PCA ──────────────────────────────────────────────────────

    pca30 = PCA(n_components=min(30, X_raw.shape[0] - 1, X_raw.shape[1]))
    X_pca30 = pca30.fit_transform(X_raw)
    pca2 = PCA(n_components=2)
    X_pca_2d = pca2.fit_transform(X_pca30)

    print(f"PCA: 30d variance={pca30.explained_variance_ratio_.sum():.3f}  "
          f"2d variance={pca2.explained_variance_ratio_.sum():.3f}")

    np.savez_compressed(
        output_dir / "pca_projection.npz",
        X_pca_2d=X_pca_2d.astype(np.float32),
        station_ids=np.array(station_ids),
        geo_labels=np.array(geo_labels),
        pca_explained_30d=float(pca30.explained_variance_ratio_.sum()),
        pca_explained_2d=float(pca2.explained_variance_ratio_.sum()),
    )

    # ── Intra / inter group cosine ────────────────────────────────

    print("Computing intra/inter group cosine...")
    intra_group = defaultdict(list)
    inter_group = []
    for i in range(len(station_ids)):
        for j in range(i + 1, len(station_ids)):
            cs = cosine_sim(X_raw[i], X_raw[j])
            if geo_labels[i] == geo_labels[j]:
                intra_group[geo_labels[i]].append(cs)
            else:
                inter_group.append(cs)

    intra_summary = {}
    for g in geo_set:
        vals = intra_group.get(g, [])
        intra_summary[g] = {
            "mean_cos": float(np.mean(vals)) if vals else 0.0,
            "n_stations": sum(1 for l in geo_labels if l == g),
            "n_pairs": len(vals),
        }

    all_intra = [v for vals in intra_group.values() for v in vals]
    mean_intra = float(np.mean(all_intra)) if all_intra else 0.0
    mean_inter = float(np.mean(inter_group)) if inter_group else 0.0

    print(f"  ALL INTRA: {mean_intra:.4f}  ALL INTER: {mean_inter:.4f}  "
          f"DELTA: {mean_intra - mean_inter:+.4f}")

    with open(output_dir / "intra_summary.json", "w") as f:
        json.dump({
            "all_intra": mean_intra, "all_inter": mean_inter,
            "delta": mean_intra - mean_inter,
            "ratio": mean_intra / max(mean_inter, 1e-12),
            "n_stations": n_stations,
            "n_geo_groups": len(geo_set),
            "kilauea_intra_cos": intra_summary.get("Basalt_Kilauea", {}).get("mean_cos", 0),
            "groups": {g: intra_summary[g] for g in sorted(intra_summary)},
        }, f, indent=1)
    print(f"Saved: {output_dir}/intra_summary.json")


if __name__ == "__main__":
    main()
