"""Standard HVSR vs measured Vs30: 656 KiK-net stations.

Context benchmark analysis: what correlation does standard multi-event
earthquake-HVSR achieve against profile-derived measured Vs30, given many
earthquake records? This places the JsT proxy-Vs30 result on an external
measured-Vs30 scale, but it is not a head-to-head JsT comparison.

Input: data/kiknet_measured_vs30_pwave_v1/hvsr_validation/
         kiknet_measured_vs30_hvsr_validation_manifest.csv (656 stations)

Key question: if a mature multi-event earthquake-HVSR product correlates with
measured Vs30 at ρ = X, then JsT single-event at ρ = -0.59 (proxy) should be
interpreted relative to that empirical scale, not to perfect ground truth.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "manuscript" / "figures"))
from nature_geo_style import SEMANTIC


def main():
    manifest_csv = Path(
        "data/kiknet_measured_vs30_pwave_v1/hvsr_validation/"
        "kiknet_measured_vs30_hvsr_validation_manifest.csv"
    )
    df = pd.read_csv(manifest_csv)
    n_stations = len(df)
    print(f"Stations: {n_stations}")

    # ── Vs30 ─────────────────────────────────────────────────────────
    vs30 = df["vs30_m_s"].values.astype(float)
    vs30_log = np.log10(np.maximum(vs30, 1.0))

    # ── HVSR peak metrics ────────────────────────────────────────────
    # First peak: frequency + amplitude
    f0 = df["hvsr_first_peak_frequency_hz"].values.astype(float)
    a0 = df["hvsr_first_peak_amplitude"].values.astype(float)

    # Predominant peak
    fp = df["hvsr_predominant_peak_frequency_hz"].values.astype(float)
    ap = df["hvsr_predominant_peak_amplitude"].values.astype(float)

    # Geometric mean amplitude (broadband)
    gmean = df["hvsr_geometric_mean_amplitude"].values.astype(float)

    # Peak count
    n_peaks = df["hvsr_significant_peak_count"].values.astype(int)

    # NEHRP class
    nehrp = df["nehrp_site_class"].values

    # ── Filter to stations with at least one significant peak ────────
    has_peak = (n_peaks >= 1) & (~np.isnan(f0))
    n_with_peak = has_peak.sum()
    print(f"Stations with ≥1 significant peak: {n_with_peak} ({n_with_peak/n_stations:.0%})")

    # ── 1. A0 vs Vs30 ────────────────────────────────────────────────
    mask = has_peak & (~np.isnan(a0)) & (~np.isnan(vs30))
    rho_a0, p_a0 = spearmanr(a0[mask], vs30[mask])
    r_a0, p_r_a0 = pearsonr(np.log10(np.maximum(a0[mask], 1.0)), vs30_log[mask])

    print(f"\n=== A0 (first peak amplitude) vs Vs30 ===")
    print(f"  Spearman ρ = {rho_a0:+.4f}  p = {p_a0:.2e}  N = {mask.sum()}")
    print(f"  Pearson (log-log) r = {r_a0:+.4f}  p = {p_r_a0:.2e}")

    # ── 2. F0 vs Vs30 ────────────────────────────────────────────────
    mask_f = has_peak & (~np.isnan(f0)) & (~np.isnan(vs30_log))
    rho_f0, p_f0 = spearmanr(f0[mask_f], vs30[mask_f])
    r_f0, p_r_f0 = pearsonr(np.log10(np.maximum(f0[mask_f], 0.05)), vs30_log[mask_f])
    print(f"\n=== F0 (first peak frequency) vs Vs30 ===")
    print(f"  Spearman ρ = {rho_f0:+.4f}  p = {p_f0:.2e}  N = {mask_f.sum()}")
    print(f"  Pearson (log-log) r = {r_f0:+.4f}  p = {p_r_f0:.2e}")

    # ── 3. Predominant peak amplitude vs Vs30 ────────────────────────
    mask_p = has_peak & (~np.isnan(ap)) & (~np.isnan(vs30))
    rho_ap, p_ap = spearmanr(ap[mask_p], vs30[mask_p])
    print(f"\n=== Ap (predominant peak amplitude) vs Vs30 ===")
    print(f"  Spearman ρ = {rho_ap:+.4f}  p = {p_ap:.2e}  N = {mask_p.sum()}")

    # ── 4. Geometric mean amplitude vs Vs30 ──────────────────────────
    mask_g = (~np.isnan(gmean)) & (~np.isnan(vs30))
    rho_gm, p_gm = spearmanr(gmean[mask_g], vs30[mask_g])
    print(f"\n=== Geometric-mean HVSR amplitude vs Vs30 ===")
    print(f"  Spearman ρ = {rho_gm:+.4f}  p = {p_gm:.2e}  N = {mask_g.sum()}")

    # ── 5. Per-NEHRP class ───────────────────────────────────────────

    print(f"\n=== Per-NEHRP class ===")
    per_class = {}
    for cls in ["A", "B", "C", "D", "E"]:
        mask_c = (nehrp == cls) & has_peak & (~np.isnan(a0)) & (~np.isnan(vs30))
        if mask_c.sum() < 3:
            print(f"  {cls}: N={mask_c.sum()} — too few")
            continue
        rho_c, p_c = spearmanr(a0[mask_c], vs30[mask_c])
        vs30_range = f"{vs30[mask_c].min():.0f}-{vs30[mask_c].max():.0f}"
        print(f"  {cls}: N={mask_c.sum()}  ρ={rho_c:+.4f}  p={p_c:.4f}  "
              f"Vs30 range=[{vs30_range}]  ã0 mean={a0[mask_c].mean():.2f}")
        per_class[cls] = {
            "N": int(mask_c.sum()),
            "spearman_rho": float(rho_c),
            "p_value": float(p_c),
            "vs30_range": vs30_range,
            "a0_mean": float(a0[mask_c].mean()),
            "a0_std": float(a0[mask_c].std()),
        }

    # ── 6. Peak count vs Vs30 significance ───────────────────────────

    # Do stations with 2+ peaks have different A0-Vs30 correlation?
    multi_peak = (n_peaks >= 2) & has_peak & (~np.isnan(a0)) & (~np.isnan(vs30))
    single_peak = (n_peaks == 1) & has_peak & (~np.isnan(a0)) & (~np.isnan(vs30))
    if multi_peak.sum() >= 5:
        rho_multi, p_multi = spearmanr(a0[multi_peak], vs30[multi_peak])
        rho_single, p_single = spearmanr(a0[single_peak], vs30[single_peak])
        print(f"\n=== Peak count stratification ===")
        print(f"  Single-peak (N={single_peak.sum()}): ρ={rho_single:+.4f} p={p_single:.4f}")
        print(f"  Multi-peak  (N={multi_peak.sum()}):  ρ={rho_multi:+.4f} p={p_multi:.4f}")

    # ── 7. Summary — external context benchmark ─────────────────────

    # Best metric: which HVSR quantity correlates most strongly with Vs30?
    metrics = {
        "A0 (first peak amplitude)": (rho_a0, p_a0, mask.sum()),
        "F0 (first peak frequency)": (rho_f0, p_f0, mask_f.sum()),
        "Ap (predominant peak amplitude)": (rho_ap, p_ap, mask_p.sum()),
        "Geometric mean amplitude": (rho_gm, p_gm, mask_g.sum()),
    }
    best = max(metrics, key=lambda k: abs(metrics[k][0]))

    print(f"\n{'='*65}")
    print(f"  CONTEXT BENCHMARK: Standard multi-event earthquake HVSR")
    print(f"  vs 656 KiK-net profile-derived measured Vs30")
    print(f"{'='*65}")
    print(f"  Best metric: {best}  ρ = {metrics[best][0]:+.4f}  N = {metrics[best][2]}")
    print(f"  For context:")
    print(f"    JsT-HVSR          vs proxy  Vs30: ρ = -0.59  (N=86)")
    print(f"    Standard HVSR A0  vs measured Vs30: ρ = {rho_a0:+.4f}  (N={mask.sum()})")
    print(f"    Standard HVSR F0  vs measured Vs30: ρ = {rho_f0:+.4f}  (N={mask_f.sum()})")

    # ── Save ────────────────────────────────────────────────────────

    out = Path("outputs/kiknet_hvsr_vs30_method_ceiling")
    out.mkdir(parents=True, exist_ok=True)

    results = {
        "source": "GFZ earthquake-HVSR (Zhu et al. 2020, doi:10.5880/GFZ.2.1.2020.006)",
        "stations_total": int(n_stations),
        "stations_with_peak": int(n_with_peak),
        "vs30_range": [float(vs30.min()), float(vs30.max())],
        "per_metric": {k: {"spearman_rho": float(v[0]), "p_value": float(v[1]), "N": int(v[2])}
                       for k, v in metrics.items()},
        "per_nehrp_class": per_class,
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {out / 'results.json'}")


if __name__ == "__main__":
    main()
