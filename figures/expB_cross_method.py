"""Exp B: Cross-method agreement — JsT single-event vs standard multi-event HVSR.

Key questions from reviewer risk #3:
  1. Is 0.91× within-method reproducibility or true cross-method agreement?
  2. What is the direct cos similarity between JsT single-event and standard multi-event?
  3. Is the rank ordering of stations preserved across methods?

Data: outputs/single_vs_multi_event/results.json (50 stations, 16 networks)
"""

from __future__ import annotations

import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    results_path = Path("outputs/single_vs_multi_event/results.json")
    with open(results_path) as f:
        data = json.load(f)

    stations = data["stations"]
    aggregate = data["aggregate"]

    # Extract per-station metrics
    station_metrics = []
    for s in stations:
        station_metrics.append({
            "station_id": s["station_id"],
            "n_events": s.get("n_events", 0),
            "jst_single_to_multi": s["jst_single_to_multi_mean"],
            "hv_single_to_multi": s["hv_single_to_multi_mean"],
            "jst_cross_event": s.get("jst_cross_event_mean", None),
            "jst_single_vs_hv_multi": s.get("jst_single_vs_hv_multi", None),
            "jst_multi_vs_hv_multi": s.get("jst_multi_vs_hv_multi", None),
        })

    # ── 1. Cross-method agreement: JsT single vs standard multi ───

    cross_vals = [s["jst_single_vs_hv_multi"] for s in station_metrics
                  if s["jst_single_vs_hv_multi"] is not None]
    cross_mean = float(np.mean(cross_vals)) if cross_vals else 0.0
    cross_std = float(np.std(cross_vals)) if cross_vals else 0.0

    # Compare: JsT single → multi  vs  HV single → multi  (within-method)
    jst_within = [s["jst_single_to_multi"] for s in station_metrics]
    hv_within = [s["hv_single_to_multi"] for s in station_metrics]
    jst_within_mean = float(np.mean(jst_within))
    hv_within_mean = float(np.mean(hv_within))

    print(f"=== Cross-method agreement ===")
    print(f"  JsT single → standard multi: cos = {cross_mean:.4f} ± {cross_std:.4f}")
    print(f"  JsT single → JsT multi:     cos = {jst_within_mean:.4f}")
    print(f"  HV single → HV multi:       cos = {hv_within_mean:.4f}")
    print(f"  Ratio (JsT cross / HV within): {cross_mean / max(hv_within_mean, 1e-12):.3f}×")

    # ── 2. Per-network cross-method agreement ──────────────────────

    net_cross = defaultdict(list)
    for s in station_metrics:
        net = s["station_id"].split(".")[0]
        net_cross[net].append(s)

    per_net = []
    for net, stns in sorted(net_cross.items()):
        vals = [s["jst_single_vs_hv_multi"] for s in stns if s["jst_single_vs_hv_multi"] is not None]
        if len(vals) >= 1:
            per_net.append({
                "network": net,
                "n": len(vals),
                "cross_cos_mean": float(np.mean(vals)),
                "cross_cos_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                "jst_within_mean": float(np.mean([s["jst_single_to_multi"] for s in stns])),
            })

    print(f"\n=== Per-network cross-method ===")
    for d in per_net:
        ratio = d["cross_cos_mean"] / max(d["jst_within_mean"], 1e-12)
        print(f"  {d['network']:4s} N={d['n']:2d}  "
              f"cross={d['cross_cos_mean']:.3f}  "
              f"JST_within={d['jst_within_mean']:.3f}  "
              f"ratio={ratio:.2f}×")

    # ── 3. Rank correlation: JsT-HVSR vs standard HVSR ────────────

    # Stations ordered by JsT amplification vs ordered by standard HVSR
    jst_amps = [s["jst_single_to_multi"] for s in station_metrics]
    hv_amps = [s["hv_single_to_multi"] for s in station_metrics]
    rho_rank, p_rank = spearmanr(jst_amps, hv_amps)

    print(f"\n=== Rank agreement ===")
    print(f"  JsT rank vs HV rank: ρ={rho_rank:+.4f} (p={p_rank:.4f})")

    # ── 4. Key summary metric ─────────────────────────────────────

    # What fraction of stations have cross-method cos > 0.7?
    n_strong = sum(1 for v in cross_vals if v > 0.7)
    frac_strong = n_strong / max(len(cross_vals), 1)

    print(f"\n=== Summary ===")
    print(f"  Cross-method cos: {cross_mean:.3f} ± {cross_std:.3f}")
    print(f"  {n_strong}/{len(cross_vals)} stations have cross > 0.7 ({frac_strong:.0%})")
    print(f"  JsT within-method: {jst_within_mean:.3f}")
    print(f"  True ratio: {cross_mean / max(jst_within_mean, 1e-12):.2f}× "
          f"(JsT single vs standard multi / JsT single vs JsT multi)")

    # ── Save ──────────────────────────────────────────────────────

    results = {
        "cross_method_mean": cross_mean,
        "cross_method_std": cross_std,
        "jst_within_mean": jst_within_mean,
        "hv_within_mean": hv_within_mean,
        "cross_vs_jst_within_ratio": cross_mean / max(jst_within_mean, 1e-12),
        "cross_vs_hv_within_ratio": cross_mean / max(hv_within_mean, 1e-12),
        "fraction_cross_gt_07": frac_strong,
        "rank_spearman": float(rho_rank),
        "rank_p_value": float(p_rank),
        "per_network": per_net,
        "aggregate": aggregate,
    }

    out_dir = Path("outputs/expB_cross_method")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
