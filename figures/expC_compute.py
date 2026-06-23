"""Exp C compute: Token 7 geology vs geography disentanglement.

GPU: token extraction (fast). CPU: coordinate regression + PCA + tests.

Tests:
  1. Coordinate-removed: token7 ⊥ ECEF → residual PCA → intra/inter re-compute
  2. Same-region different geology: Hawaii pairs (Kilauea basalt vs coastal sediment)
  3. Cross-region same geology: OK basin vs GS centralUS vs embayment
"""

from __future__ import annotations

import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from JsT import SeismicWaveformDataset, load_checkpoint_models
from JsT.ablation import AblationConditionEncoder


def classify_geology(lat, lon, network):
    lat, lon = float(lat), float(lon)
    network = str(network)
    if network == "HV":
        if 19.38 <= lat <= 19.45 and -155.35 <= lon <= -155.22: return "Basalt_Kilauea"
        if lat <= 19.28: return "Basalt_coastal"
        if lat >= 19.50: return "Basalt_weathered"
        return "Basalt_flank"
    if network == "AK":
        if lat >= 64.0: return "Metamorphic_interior"
        if lat <= 60.0: return "Sedimentary_coastal"
        return "Metamorphic_range"
    if network == "AT":
        return "Volcanic_arc" if lon <= -160 else "Sedimentary_coastal"
    if network == "OK": return "Sedimentary_basin"
    if network == "GS": return "Sedimentary_centralUS"
    if network == "UU": return "Basin_range"
    if network == "UW": return "Volcanic_cascades"
    if network == "NN": return "Basin_range"
    if network in ("CI","NC"): return "Active_margin"
    if network == "NM": return "Sedimentary_embayment"
    if lat >= 48.0: return "Craton_north"
    if lat <= 35.0 and lon <= -100: return "Active_margin"
    if lat <= 38.0 and lon <= -88: return "Sedimentary_embayment"
    if lon >= -75: return "Passive_margin_east"
    return "Interior_platform"


def ecef(lat, lon):
    a, f = 6378.137, 1.0/298.257223563
    e2 = 2*f - f**2
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    N = a / np.sqrt(1 - e2 * np.sin(lat_r)**2)
    return np.array([N * np.cos(lat_r) * np.cos(lon_r),
                     N * np.cos(lat_r) * np.sin(lon_r),
                     N * (1-e2) * np.sin(lat_r)])


def cosine_sim(a, b):
    a, b = np.ravel(a), np.ravel(b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def intra_inter(X, labels):
    intra, between = [], []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            cs = cosine_sim(X[i], X[j])
            if labels[i] == labels[j]: intra.append(cs)
            else: between.append(cs)
    mi = float(np.mean(intra)) if intra else 0.0
    mb = float(np.mean(between)) if between else 0.0
    return mi, mb, mi - mb


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    p.add_argument("--output-dir", default="outputs/expC_token7_disentangle")
    p.add_argument("--drop-tokens", type=str, default="8,9,10")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    print("Loading checkpoint...")
    ce, dn, ckpt = load_checkpoint_models(args.checkpoint, dev, use_ema=True,
                                           sampling_method="heun", steps=50, cfg_scale=1.0)
    if dropped: ce = AblationConditionEncoder(ce, dropped)
    ce.eval()

    ds_train = SeismicWaveformDataset(args.data_dir, split="training", augment=False,
                                       cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default")
    ds_test = SeismicWaveformDataset(args.data_dir, split="testing", augment=False, vocab_from=ds_train,
                                      cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default")
    tc = ds_test.conditions.iloc[ds_test.indices].copy()
    tc["station_id"] = tc["station_network_code"].fillna("UNKNOWN").astype(str) + "." + tc["station_code"].fillna("UNKNOWN").astype(str)
    i2d = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # Extract token 7
    print("Extracting token 7...")
    station_tokens = {}
    station_info = {}
    for sta_id in tc["station_id"].unique():
        rows = tc[tc["station_id"] == sta_id]
        n_evts = min(6, len(rows))
        lat, lon = float(rows["station_latitude_deg"].iloc[0]), float(rows["station_longitude_deg"].iloc[0])
        net = str(rows["station_network_code"].iloc[0])
        elev = float(rows.get("station_elevation_m", pd.Series([0.0])).iloc[0]) if "station_elevation_m" in rows.columns else 0.0
        tokens = []
        for ri in rows.index[:n_evts]:
            d = i2d.get(int(ri))
            if d is None: continue
            _, cd = ds_test[d]
            cg = {k: v.unsqueeze(0).to(dev) for k, v in cd.items()}
            with torch.no_grad(): tok = ce(cg).cpu()
            tokens.append(tok[0, 7, :].numpy())
        if len(tokens) >= 2:
            station_tokens[sta_id] = np.mean(tokens, axis=0)
            station_info[sta_id] = {"lat": lat, "lon": lon, "net": net, "elevation": elev,
                                     "ecef": ecef(lat, lon)}

    sids = sorted(station_tokens.keys())
    X_raw = np.array([station_tokens[s] for s in sids])
    geo = [classify_geology(station_info[s]["lat"], station_info[s]["lon"], station_info[s]["net"]) for s in sids]
    print(f"{len(sids)} stations, token dim={X_raw.shape[1]}")

    # ── Test 1: Coordinate-removed PCA ─────────────────────────────

    print("\n=== Test 1: Coordinate-removed token 7 ===")
    ECEF = np.array([station_info[s]["ecef"] for s in sids])
    # Standardise ECEF
    ECEF_s = (ECEF - ECEF.mean(0)) / ECEF.std(0)
    # Regress out ECEF from each token dimension
    reg = LinearRegression().fit(ECEF_s, X_raw)
    X_resid = X_raw - reg.predict(ECEF_s)
    # Re-PCA
    pca_orig = PCA(n_components=30).fit(X_raw)
    pca_resid = PCA(n_components=30).fit(X_resid)
    pca2_orig = PCA(n_components=2).fit_transform(pca_orig.transform(X_raw))
    pca2_resid = PCA(n_components=2).fit_transform(pca_resid.transform(X_resid))

    # Intra/inter before and after coordinate removal
    mi_orig, mb_orig, delta_orig = intra_inter(pca_orig.transform(X_raw), geo)
    mi_resid, mb_resid, delta_resid = intra_inter(pca_resid.transform(X_resid), geo)

    print(f"  Original: intra={mi_orig:.4f} inter={mb_orig:.4f} Δ={delta_orig:+.4f}")
    print(f"  ECEF-removed: intra={mi_resid:.4f} inter={mb_resid:.4f} Δ={delta_resid:+.4f}")
    delta_ratio = delta_resid / max(delta_orig, 1e-12)
    print(f"  Δ retained: {delta_ratio:.1%}")

    test1 = {
        "original_intra": mi_orig, "original_inter": mb_orig, "original_delta": delta_orig,
        "ecef_removed_intra": mi_resid, "ecef_removed_inter": mb_resid, "ecef_removed_delta": delta_resid,
        "delta_retained_ratio": float(delta_ratio),
        "ecef_variance_explained": float(np.trace(reg.coef_ @ ECEF_s.T @ ECEF_s @ reg.coef_.T) / np.trace(X_raw.T @ X_raw)),
    }
    if delta_resid > 0.04:
        print("  ✓ Geological clustering survives coordinate removal — not purely geographic.")
    else:
        print("  ✗ Geological clustering collapses under coordinate removal — may be geographic proxy.")

    # ── Test 2: Same-region different geology (Hawaii) ────────────

    print("\n=== Test 2: Same-region different geology ===")
    hi_sids = [s for s in sids if station_info[s]["net"] == "HV"]
    if len(hi_sids) >= 8:
        hi_geo = [classify_geology(station_info[s]["lat"], station_info[s]["lon"], "HV") for s in hi_sids]
        hi_X = np.array([station_tokens[s] for s in hi_sids])

        # Kilauea vs coastal
        ki_mask = np.array([g == "Basalt_Kilauea" for g in hi_geo])
        coastal_mask = np.array([g == "Basalt_coastal" for g in hi_geo])
        flank_mask = np.array([g == "Basalt_flank" for g in hi_geo])

        # Within Kilauea
        ki_within = []
        ki_idx = np.where(ki_mask)[0]
        for i in range(len(ki_idx)):
            for j in range(i+1, len(ki_idx)):
                ki_within.append(cosine_sim(hi_X[ki_idx[i]], hi_X[ki_idx[j]]))
        ki_intra = float(np.mean(ki_within)) if ki_within else 0.0

        # Kilauea vs coastal
        ki_vs_coastal = []
        for i in np.where(ki_mask)[0]:
            for j in np.where(coastal_mask)[0]:
                ki_vs_coastal.append(cosine_sim(hi_X[i], hi_X[j]))
        ki_coastal_cross = float(np.mean(ki_vs_coastal)) if ki_vs_coastal else 0.0

        # Kilauea vs flank
        ki_vs_flank = []
        for i in np.where(ki_mask)[0]:
            for j in np.where(flank_mask)[0]:
                ki_vs_flank.append(cosine_sim(hi_X[i], hi_X[j]))
        ki_flank_cross = float(np.mean(ki_vs_flank)) if ki_vs_flank else 0.0

        test2 = {
            "kilauea_intra": ki_intra,
            "kilauea_vs_coastal": ki_coastal_cross,
            "kilauea_vs_flank": ki_flank_cross,
            "n_kilauea": int(ki_mask.sum()),
            "n_coastal": int(coastal_mask.sum()),
            "n_flank": int(flank_mask.sum()),
        }
        print(f"  Kilauea intra: {ki_intra:.4f}")
        print(f"  Kilauea vs coastal: {ki_coastal_cross:.4f}  Δ={ki_intra-ki_coastal_cross:+.4f}")
        print(f"  Kilauea vs flank: {ki_flank_cross:.4f}  Δ={ki_intra-ki_flank_cross:+.4f}")
    else:
        test2 = {"error": "insufficient Hawaii stations"}
        print("  Insufficient Hawaii stations.")

    # ── Test 3: Cross-region same geology ─────────────────────────

    print("\n=== Test 3: Cross-region same geology ===")
    basin_groups = ["Sedimentary_basin", "Sedimentary_centralUS", "Sedimentary_embayment"]
    basin_data = {}
    for g in basin_groups:
        mask = np.array([gl == g for gl in geo])
        if mask.sum() >= 5:
            basin_data[g] = {
                "stations": [s for s, m in zip(sids, mask) if m],
                "X": X_raw[mask],
                "n": int(mask.sum()),
            }
    if len(basin_data) >= 2:
        # Within each basin group
        within = {}
        for g, d in basin_data.items():
            cs_vals = []
            for i in range(d["n"]):
                for j in range(i+1, d["n"]):
                    cs_vals.append(cosine_sim(d["X"][i], d["X"][j]))
            within[g] = float(np.mean(cs_vals)) if cs_vals else 0.0

        # Cross: between basin groups
        cross = {}
        gnames = sorted(basin_data.keys())
        for gi in range(len(gnames)):
            for gj in range(gi+1, len(gnames)):
                g1, g2 = gnames[gi], gnames[gj]
                cs_vals = []
                for i in range(basin_data[g1]["n"]):
                    for j in range(basin_data[g2]["n"]):
                        cs_vals.append(cosine_sim(basin_data[g1]["X"][i], basin_data[g2]["X"][j]))
                cross[f"{g1}_vs_{g2}"] = float(np.mean(cs_vals)) if cs_vals else 0.0

        within_mean = np.mean(list(within.values()))
        cross_mean = np.mean(list(cross.values()))
        test3 = {
            "within_basin": within,
            "cross_basin": cross,
            "mean_within": float(within_mean),
            "mean_cross": float(cross_mean),
            "delta": float(within_mean - cross_mean),
        }
        print(f"  Within basins: {within}")
        print(f"  Cross basins: {cross}")
        print(f"  Mean within={within_mean:.4f}  cross={cross_mean:.4f}  Δ={within_mean-cross_mean:+.4f}")
        if within_mean > cross_mean:
            print("  ✓ Same-geology basins cluster across regions — geological signal transcends geography.")
        else:
            print("  ✗ Basin groups do not cluster across regions — token 7 may be region-specific.")
    else:
        test3 = {"error": "insufficient basin groups"}
        print("  Insufficient basin groups.")

    # ── Save ──────────────────────────────────────────────────────

    results = {
        "n_stations": len(sids),
        "test1_coordinate_removed": test1,
        "test2_hawaii_same_region": test2,
        "test3_cross_region_basins": test3,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Synthesis ───────────────────────────────────────—

    print(f"\n{'='*60}\n  VERDICT\n{'='*60}")
    checks = []
    if test1.get("delta_retained_ratio", 0) > 0.5:
        checks.append("✓ Geological clusters survive ECEF removal")
    if test2.get("kilauea_intra", 0) > test2.get("kilauea_vs_coastal", 1):
        checks.append("✓ Kilauea token 7 separable from coastal within Hawaii")
    if test3.get("delta", -1) > 0:
        checks.append("✓ Same-geology basins converge across regions")

    for c in checks: print(f"  {c}")
    if len(checks) >= 2:
        print(f"\n  Token 7 encodes GEOLOGICAL knowledge, not just geographic proximity.")
    else:
        print(f"\n  Token 7 may be partially geographic — results mixed ({len(checks)}/3).")

    print(f"\nSaved: {out / 'results.json'}")


if __name__ == "__main__":
    import pandas as pd
    main()
