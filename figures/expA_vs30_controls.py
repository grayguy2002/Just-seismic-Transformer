"""Exp A: Proxy Vs30 confounding controls.

Tests whether JsT-HVSR vs proxy Vs30 correlation survives spatial/network controls:
  1. Partial correlation: controlling for lat, lon, elevation, network dummy
  2. Leave-one-network-out: ρ stability when dropping each major network
  3. Within-network: ρ within each network (directional consistency check)

Data: outputs/vs30_validation/jst_hvsr_vs_vs30_results.csv (pre-computed)
      data/site_vs30_search/jst_testing_station_inventory.csv (station metadata)
"""

from __future__ import annotations

import sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def partial_correlation(x, y, z):
    """Compute partial correlation r_xy.z controlling for matrix z (n_samples, n_controls)."""
    from scipy.stats import pearsonr
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    # Residualise x on z
    beta_x = np.linalg.lstsq(z, x, rcond=None)[0]
    resid_x = x - z @ beta_x
    # Residualise y on z
    beta_y = np.linalg.lstsq(z, y, rcond=None)[0]
    resid_y = y - z @ beta_y
    r, p = pearsonr(resid_x, resid_y)
    return r, p, resid_x, resid_y


def main():
    # ── Load data ─────────────────────────────────────────────────
    vs30_csv = Path("outputs/vs30_validation/jst_hvsr_vs_vs30_results.csv")

    # Get station metadata from test dataset conditions
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from JsT import SeismicWaveformDataset
    ds = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    conds = ds.conditions.iloc[ds.indices].copy()
    conds["station_id"] = (conds["station_network_code"].fillna("UNKNOWN").astype(str)
                           + "." + conds["station_code"].fillna("UNKNOWN").astype(str))

    # Build station metadata map (first occurrence per station)
    sta_meta = {}
    for _, row in conds.iterrows():
        sid = row["station_id"]
        if sid not in sta_meta:
            sta_meta[sid] = {
                "lat": float(row["station_latitude_deg"]),
                "lon": float(row["station_longitude_deg"]),
                "net": str(row["station_network_code"]),
            }

    df = pd.read_csv(vs30_csv)
    df["lat"] = df["station_id"].map(lambda s: sta_meta.get(s, {}).get("lat", np.nan))
    df["lon"] = df["station_id"].map(lambda s: sta_meta.get(s, {}).get("lon", np.nan))
    df["network"] = df["station_id"].map(lambda s: sta_meta.get(s, {}).get("net", "XX"))

    # Filter to proxy
    proxy = df[df["vs30_kind"] == "proxy"].copy()
    proxy = proxy.dropna(subset=["vs30", "mean_amp", "lon", "lat"])
    proxy["vs30_log"] = np.log10(proxy["vs30"].values.astype(float))

    xs = proxy["vs30"].values.astype(float)
    ys = proxy["mean_amp"].values.astype(float)
    lons = proxy["lon"].values.astype(float)
    lats = proxy["lat"].values.astype(float)
    nets = proxy["network"].values

    N = len(xs)
    rho_base, p_base = spearmanr(xs, ys)
    r_pearson, p_pearson = pearsonr(xs, ys)

    print(f"Baseline: ρ={rho_base:+.4f} N={N}")

    # ── 1. Partial correlation controls ───────────────────────────

    print("\n=== 1. Partial correlation ===")

    controls = []
    labels = []

    # (a) spatial only
    z_spatial = np.column_stack([lons, lats])
    z_spatial = (z_spatial - z_spatial.mean(0)) / z_spatial.std(0)
    r_part, p_part, _, _ = partial_correlation(xs, ys, z_spatial)
    controls.append({"label": "Spatial (lon, lat)", "r_partial": float(r_part), "p": float(p_part),
                     "r_baseline": float(r_pearson), "r_change": float(r_part - r_pearson)})
    print(f"  Spatial: r_partial={r_part:+.4f} (baseline r={r_pearson:+.4f}, Δ={r_part - r_pearson:+.4f})")

    # (b) spatial + network dummy
    major_nets = ["AK", "HV", "OK", "GS", "UU", "UW", "NN", "NM", "CI", "NC"]
    net_dummies = np.column_stack([(nets == n).astype(float) for n in major_nets if (nets == n).sum() >= 3])
    if net_dummies.shape[1] > 0:
        z_spatial_net = np.column_stack([z_spatial, net_dummies])
        r_part2, p_part2, _, _ = partial_correlation(xs, ys, z_spatial_net)
        controls.append({"label": "Spatial + network", "r_partial": float(r_part2), "p": float(p_part2),
                         "r_baseline": float(r_pearson), "r_change": float(r_part2 - r_pearson)})
        print(f"  Spatial+network: r_partial={r_part2:+.4f} (Δ={r_part2 - r_pearson:+.4f})")

    # (c) spatial + event_count (density control)
    if "event_count" in proxy.columns:
        n_events = proxy["event_count"].values.astype(float)
        z_density = np.column_stack([z_spatial, (n_events - n_events.mean()) / n_events.std()])
        r_part3, p_part3, _, _ = partial_correlation(xs, ys, z_density)
        controls.append({"label": "Spatial + event density", "r_partial": float(r_part3), "p": float(p_part3),
                         "r_baseline": float(r_pearson), "r_change": float(r_part3 - r_pearson)})
        print(f"  Spatial+density: r_partial={r_part3:+.4f} (Δ={r_part3 - r_pearson:+.4f})")

    # ── 2. Leave-one-network-out ──────────────────────────────────

    print("\n=== 2. Leave-one-network-out ===")
    lono = []
    net_counts = pd.Series(nets).value_counts()
    for net in sorted(net_counts[net_counts >= 5].index):
        mask = nets != net
        if mask.sum() < 20:
            continue
        rho_leave, p_leave = spearmanr(xs[mask], ys[mask])
        lono.append({"network": str(net), "n_removed": int((~mask).sum()),
                     "n_remaining": int(mask.sum()),
                     "rho": float(rho_leave), "p": float(p_leave),
                     "delta_from_baseline": float(rho_leave - rho_base)})
        print(f"  Drop {net}: ρ={rho_leave:+.4f} N={mask.sum()} (Δ={rho_leave - rho_base:+.4f})")

    # ── 3. Within-network ─────────────────────────────────────────

    print("\n=== 3. Within-network ===")
    within = []
    for net in sorted(net_counts[net_counts >= 8].index):
        mask = nets == net
        xs_net = xs[mask]
        ys_net = ys[mask]
        if len(xs_net) < 8:
            continue
        rho_net, p_net = spearmanr(xs_net, ys_net)
        within.append({"network": str(net), "n": int(mask.sum()),
                       "rho": float(rho_net), "p": float(p_net),
                       "direction": "negative (correct)" if rho_net < 0 else "positive"})
        print(f"  {net} (N={mask.sum()}): ρ={rho_net:+.4f} p={p_net:.4f}  {within[-1]['direction']}")

    # ── also test: within-network directional consistency ────────
    n_neg = sum(1 for w in within if w["rho"] < 0)
    n_pos = sum(1 for w in within if w["rho"] >= 0)
    direction_test = {
        "n_networks": len(within),
        "n_negative": n_neg,
        "n_positive": n_pos,
        "binomial_p": None,  # would need binomial test
    }
    if len(within) >= 5:
        from scipy.stats import binomtest
        direction_test["binomial_p"] = float(binomtest(n_neg, len(within), p=0.5, alternative="greater").pvalue)

    # ── 4. Bootstrap confidence intervals ─────────────────────────

    print("\n=== 4. Bootstrap CI ===")
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        rho_b, _ = spearmanr(xs[idx], ys[idx])
        boot_rhos.append(rho_b)
    rho_ci = np.percentile(boot_rhos, [2.5, 97.5])
    print(f"  ρ 95% CI: [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]")

    # ── Save ──────────────────────────────────────────────────────

    results = {
        "baseline": {"rho": float(rho_base), "p": float(p_base), "N": N,
                     "pearson_r": float(r_pearson), "pearson_p": float(p_pearson)},
        "partial_correlation": controls,
        "leave_one_network_out": lono,
        "within_network": within,
        "within_network_direction_test": direction_test,
        "bootstrap": {"rho_95ci": [float(rho_ci[0]), float(rho_ci[1])], "n_bootstrap": n_boot},
    }

    out_dir = Path("outputs/expA_vs30_controls")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Verdict ───────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")

    score = 0
    # Partial: did correlation survive spatial controls?
    if controls and controls[0]["r_partial"] < -0.25:
        print(f"  ✓ Partial correlation survives spatial control (r_partial={controls[0]['r_partial']:+.3f})")
        score += 1
    else:
        print(f"  ✗ Partial correlation weak under spatial control")

    # LONO: is any single network driving the result?
    if lono:
        min_rho = min(d["rho"] for d in lono)
        max_rho = max(d["rho"] for d in lono)
        if min_rho < -0.35:
            print(f"  ✓ LONO: ρ range [{min_rho:+.3f}, {max_rho:+.3f}] — no single network dominates")
            score += 1
        else:
            print(f"  ✗ LONO: weakest ρ={min_rho:+.3f}")

    # Within-network: directional consistency?
    if n_neg >= 0.7 * len(within):
        print(f"  ✓ Within-network: {n_neg}/{len(within)} networks negative (directional consistency)")
        score += 1
    else:
        print(f"  ✗ Within-network: only {n_neg}/{len(within)} negative")

    if score >= 2:
        print(f"\n  Vs30 correlation IS ROBUST to spatial/network controls (score {score}/3).")
    elif score == 1:
        print(f"\n  Vs30 correlation PARTIALLY robust (score {score}/3). Needs caveats.")
    else:
        print(f"\n  Vs30 correlation WEAK under controls (score {score}/3). Proxy may be confounded.")

    print(f"\nSaved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
