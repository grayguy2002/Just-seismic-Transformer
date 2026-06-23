"""Exp B follow-up: Does JsT-HVSR / standard-HVSR agreement depend on geological complexity?

Hypothesis:
  - Simple geology (sedimentary basin): near-surface dominates → both methods agree
  - Complex geology (subduction, volcanic): deep crustal → methods diverge

Test:
  1. Classify each station's geological complexity (0=simple basin, 4=complex subduction)
  2. Per-station cross-method cos vs complexity rank
  3. Per-network cross-method cos vs geological type
  4. Rank ρ within each geological class

Data: outputs/single_vs_multi_event/results.json + station metadata
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── geological complexity scoring ────────────────────────────────────────

def geo_complexity(network: str, lat: float, lon: float) -> tuple[int, str]:
    """Score 0 (simplest) to 4 (most complex)."""
    network = str(network)
    lat, lon = float(lat), float(lon)

    # Type 0: Craton interiors — simplest velocity structure
    if network in ("GS",):         return 0, "Craton"
    if lat >= 50 and lon <= -90:   return 0, "Craton"

    # Type 1: Sedimentary basins — layered, but relatively homogeneous
    if network in ("OK",):         return 1, "Sedimentary basin"
    if network in ("NM",):         return 1, "Sedimentary basin"
    if (35 <= lat <= 40) and (-100 <= lon <= -88): return 1, "Sedimentary basin"

    # Type 2: Active margins / basin-range — moderate heterogeneity
    if network in ("UU", "NN"):    return 2, "Basin-range"
    if network in ("CI", "NC"):    return 2, "Active margin"

    # Type 3: Volcanic arcs / Cascades — strong lateral heterogeneity
    if network in ("UW",):         return 3, "Volcanic arc"
    if network in ("AV", "AT"):    return 3, "Volcanic arc"

    # Type 4: Subduction zones / oceanic islands — most complex
    if network in ("AK",):         return 4, "Subduction zone"
    if network in ("HV",):         return 4, "Volcanic island"

    # Default by geography
    if lat >= 60 and lon <= -130:  return 4, "Subduction zone"  # Alaska
    if 19 <= lat <= 22 and -156 <= lon <= -154: return 4, "Volcanic island"  # Hawaii

    return 2, "Other"


def main():
    results_path = Path("outputs/single_vs_multi_event/results.json")
    with open(results_path) as f:
        data = json.load(f)

    stations = data["stations"]

    # Get station coordinates from test dataset
    from JsT import SeismicWaveformDataset
    ds = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    conds = ds.conditions.iloc[ds.indices].copy()
    conds["station_id"] = (conds["station_network_code"].fillna("UNKNOWN").astype(str)
                           + "." + conds["station_code"].fillna("UNKNOWN").astype(str))
    sta_meta = {}
    for _, row in conds.iterrows():
        sid = row["station_id"]
        if sid not in sta_meta:
            sta_meta[sid] = {
                "lat": float(row["station_latitude_deg"]),
                "lon": float(row["station_longitude_deg"]),
                "net": str(row["station_network_code"]),
            }

    # ── Per-station analysis ──────────────────────────────────────

    per_station = []
    for s in stations:
        sid = s["station_id"]
        net = sid.split(".")[0]
        meta = sta_meta.get(sid, {"lat": 40, "lon": -100, "net": net})
        complexity, geo_type = geo_complexity(net, meta["lat"], meta["lon"])

        per_station.append({
            "station_id": sid,
            "network": net,
            "complexity": complexity,
            "geo_type": geo_type,
            "jst_within": s["jst_single_to_multi_mean"],
            "hv_within": s["hv_single_to_multi_mean"],
            "cross_cos": s.get("jst_single_vs_hv_multi", None),
        })

    # ── Test 1: Cross-method cos vs complexity ────────────────────

    valid = [s for s in per_station if s["cross_cos"] is not None]
    complexities = np.array([s["complexity"] for s in valid])
    cross_cos = np.array([s["cross_cos"] for s in valid])

    rho_comp, p_comp = spearmanr(complexities, cross_cos)
    print(f"=== Cross-method cos vs geological complexity ===")
    print(f"  ρ = {rho_comp:+.4f}  (p = {p_comp:.4f})")
    if rho_comp < 0:
        print(f"  ✓ Cross-agreement DECREASES with complexity — supports hypothesis.")
    else:
        print(f"  ~ No systematic complexity trend.")

    # ── Test 2: Per-complexity-class breakdown ────────────────────

    print(f"\n=== Per complexity class ===")
    classes = defaultdict(list)
    for s in valid:
        classes[s["complexity"]].append(s)

    class_labels = {0: "Craton", 1: "Sedimentary basin", 2: "Basin-range/Active margin",
                    3: "Volcanic arc", 4: "Subduction/Volcanic island"}
    for c in sorted(classes):
        vals = [s["cross_cos"] for s in classes[c]]
        jst_w = [s["jst_within"] for s in classes[c]]
        hv_w = [s["hv_within"] for s in classes[c]]
        print(f"  {class_labels.get(c, f'Class {c}')}:")
        print(f"    N={len(vals)}  cross-cos={np.mean(vals):.4f}±{np.std(vals):.4f}")
        print(f"    JsT within={np.mean(jst_w):.3f}  HV within={np.mean(hv_w):.3f}")

    # ── Test 3: Within-class rank correlation ─────────────────────

    print(f"\n=== JsT-HVSR / HVSR rank correlation by complexity ===")
    for c in sorted(classes):
        if len(classes[c]) < 4:
            continue
        jst_vals = [s["jst_within"] for s in classes[c]]
        hv_vals = [s["hv_within"] for s in classes[c]]
        rho_c, p_c = spearmanr(jst_vals, hv_vals)
        print(f"  {class_labels.get(c, f'Class {c}')}: ρ={rho_c:+.3f} (p={p_c:.3f}) N={len(jst_vals)}")

    # ── Test 4: Per-geological-type ──────────────────────────────

    print(f"\n=== Per geological type ===")
    geo_groups = defaultdict(list)
    for s in valid:
        geo_groups[s["geo_type"]].append(s)

    for g in sorted(geo_groups):
        vals = [s["cross_cos"] for s in geo_groups[g]]
        print(f"  {g:25s}  N={len(vals):2d}  cross-cos={np.mean(vals):.4f}±{np.std(vals):.4f}")

    # ── Test 5: Basin vs Non-basin comparison ─────────────────────

    basin_mask = np.array([s["complexity"] <= 1 for s in valid])
    non_basin_mask = ~basin_mask

    basin_cross = cross_cos[basin_mask]
    non_basin_cross = cross_cos[non_basin_mask]

    print(f"\n=== Basin (simple) vs Non-basin (complex) ===")
    print(f"  Basin (complexity ≤1):    cross-cos={np.mean(basin_cross):.4f}±{np.std(basin_cross):.4f} N={len(basin_cross)}")
    print(f"  Non-basin (complexity ≥2): cross-cos={np.mean(non_basin_cross):.4f}±{np.std(non_basin_cross):.4f} N={len(non_basin_cross)}")
    delta = np.mean(basin_cross) - np.mean(non_basin_cross)
    print(f"  Δ = {delta:+.4f}")

    # ── Test 6: Sedimentary stations list ─────────────────────────

    print(f"\n=== Top sedimentary stations (should have highest cross-cos) ===")
    sed_valid = [s for s in valid if s["complexity"] <= 1]
    sed_sorted = sorted(sed_valid, key=lambda s: s["cross_cos"], reverse=True)
    for s in sed_sorted[:10]:
        print(f"  {s['station_id']:20s}  {s['geo_type']:20s}  cross={s['cross_cos']:.4f}")

    # ── Save ──────────────────────────────────────────────────────

    summary = {
        "complexity_cross_spearman": float(rho_comp),
        "complexity_cross_pvalue": float(p_comp),
        "basin_cross_mean": float(np.mean(basin_cross)),
        "non_basin_cross_mean": float(np.mean(non_basin_cross)),
        "basin_minus_non_basin": float(delta),
        "n_basin": int(basin_mask.sum()),
        "n_non_basin": int(non_basin_mask.sum()),
        "per_class": {str(c): {
            "n": len(classes[c]),
            "cross_cos_mean": float(np.mean([s["cross_cos"] for s in classes[c]])),
            "cross_cos_std": float(np.std([s["cross_cos"] for s in classes[c]])),
        } for c in sorted(classes)},
        "per_geo_type": {g: {
            "n": len(geo_groups[g]),
            "cross_cos_mean": float(np.mean([s["cross_cos"] for s in geo_groups[g]])),
        } for g in sorted(geo_groups)},
    }

    out = Path("outputs/expB_followup")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Verdict ───────────────────────────────────────────────────

    print(f"\n{'='*60}\n  VERDICT\n{'='*60}")
    if delta > 0.05:
        print(f"  ✓ Cross-method agreement {delta:+.3f} HIGHER in simple geology.")
        print(f"  JsT-HVSR and standard HVSR converge in basins, diverge in complex terrain.")
        print(f"  This supports: JsT-HVSR captures deeper crust — standard HVSR captures near-surface.")
        print(f"  They are COMPLEMENTARY measurements, not competitors.")
    else:
        print(f"  ~ No clear complexity dependence (Δ={delta:+.3f}).")
        print(f"  Cross-method cos similar across geological settings.")

    print(f"\nSaved: {out / 'results.json'}")


if __name__ == "__main__":
    main()
