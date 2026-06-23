"""Figure 2 render — Validation & Complementarity.

Four panels (2×2 grid):
  (a) JsT-HVSR vs proxy Vs30 scatter, ρ=−0.59, N=86, with CI + robustness annotations
  (b) Per-network single-event self-consistency bar chart (16 networks, 0.91× overall)
  (c) Hawaii distance-similarity, ρ=−0.55, N=300 pairs, no external Vs30
  (d) Complementarity: cross-method cos + rank ρ by geological class (double dissociation)

Data: vs30 CSV, single_vs_multi JSON, hawaii_pairs JSON, expB_followup JSON
"""

import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nature_geo_style import (
    MID_COL, DOUBLE_COL, SEMANTIC,
    add_panel_label, apply_nature_style, save_panel,
)

OUT = Path(__file__).resolve().parent.parent / "figures"

CLASS_LABELS = {0: "Craton", 1: "Sedimentary\nbasin", 2: "Basin-range",
                3: "Volcanic arc", 4: "Subduction\nzone / Volcanic"}
# From Exp B follow-up run output (printed, not in JSON)
CLASS_RANK_RHO = {0: -0.426, 1: 0.050, 2: 0.042, 3: None, 4: 0.818}
CLASS_N = {0: 17, 1: 9, 2: 12, 3: 1, 4: 11}


# ── panel (a): Vs30 scatter with robustness ─────────────────────────────

def panel_a():
    df = pd.read_csv("outputs/vs30_validation/jst_hvsr_vs_vs30_results.csv")
    proxy = df[df["vs30_kind"] == "proxy"].copy()
    xs_proxy = proxy["vs30"].values.astype(float)
    ys_proxy = proxy["mean_amp"].values.astype(float)
    mask = ~(np.isnan(xs_proxy) | np.isnan(ys_proxy))
    xs_proxy, ys_proxy = xs_proxy[mask], ys_proxy[mask]
    rho, pval = spearmanr(xs_proxy, ys_proxy)

    xs_all = df["vs30"].values.astype(float)
    ys_all = df["mean_amp"].values.astype(float)
    mask_all = ~(np.isnan(xs_all) | np.isnan(ys_all))

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(MID_COL, 80 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.12)

    ax.scatter(xs_all[mask_all], ys_all[mask_all], c=SEMANTIC["bg_grey"], s=8, alpha=0.28,
               edgecolors="none", label=f"All (N={mask_all.sum()})")
    ax.scatter(xs_proxy, ys_proxy, c=SEMANTIC["jsT_blue"], s=16, alpha=0.68,
               edgecolors="white", lw=0.3, label=f"Proxy (N={len(xs_proxy)})")

    z = np.polyfit(xs_proxy, ys_proxy, 1)
    x_fit = np.linspace(xs_proxy.min(), xs_proxy.max(), 100)
    ax.plot(x_fit, np.polyval(z, x_fit), color=SEMANTIC["black"], lw=0.8, alpha=0.85)
    se = np.sqrt(np.sum((ys_proxy - np.polyval(z, xs_proxy))**2) / (len(xs_proxy) - 2))
    ax.fill_between(x_fit, np.polyval(z, x_fit) - 1.96*se, np.polyval(z, x_fit) + 1.96*se,
                    color=SEMANTIC["jsT_blue"], alpha=0.07, lw=0)

    ax.set_xlabel("USGS proxy Vs30 (m s$^{-1}$)")
    ax.set_ylabel("Mean JsT-HVSR amplification")
    ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.2, fontsize=5.8)

    # Main stat
    ax.text(0.03, 0.92, f"$\\rho$ = {rho:+.2f}   p = {pval:.1e}   N = {len(xs_proxy)}",
            transform=ax.transAxes, fontsize=5.8, va="top", color=SEMANTIC["black"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.80,
                      lw=0.25, edgecolor=SEMANTIC["axis_grey"]))

    # Robustness mini-annotation
    ax.text(0.03, 0.18,
            "Robustness controls:\n"
            "Partial corr (spatial): r = −0.40\n"
            "Leave-one-network-out: range [-0.65, -0.48]\n"
            "Within-network: 3/4 negative\n"
            "Bootstrap 95% CI: [−0.71, −0.44]",
            transform=ax.transAxes, fontsize=4.8, va="bottom", color=SEMANTIC["axis_grey"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75,
                      lw=0.2, edgecolor=SEMANTIC["axis_grey"]))

    add_panel_label(ax, "a")
    ax.grid(True, alpha=0.10, lw=0.2)
    save_panel(fig, OUT / "fig2_panel_a")
    plt.close(fig)
    print(f"  Panel (a): Vs30 ρ={rho:+.3f} N={len(xs_proxy)}")


# ── panel (b): per-network self-consistency ─────────────────────────────

def panel_b():
    from collections import defaultdict
    with open("outputs/single_vs_multi_event/results.json") as f:
        data = json.load(f)

    stations = data["stations"]
    aggregate = data["aggregate"]
    net_groups = defaultdict(list)
    for s in stations:
        net = s["station_id"].split(".")[0]
        net_groups[net].append(s)

    per_net = []
    for net, stns in sorted(net_groups.items()):
        jst_acc = np.mean([s["jst_single_to_multi_mean"] for s in stns])
        hv_acc = np.mean([s["hv_single_to_multi_mean"] for s in stns])
        per_net.append({"network": net, "n_stations": len(stns),
                         "ratio": float(jst_acc / max(hv_acc, 1e-12))})
    per_net.sort(key=lambda x: x["ratio"], reverse=True)

    networks = [d["network"] for d in per_net]
    ratios = [d["ratio"] for d in per_net]
    n_stns = [d["n_stations"] for d in per_net]

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(MID_COL, 80 / 25.4))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.17)

    x_pos = np.arange(len(networks))
    # Color by complexity: blue for simple, grey for moderate, rust for complex
    complexity_map = {"GS": 0, "OK": 1, "NM": 1, "ZD": 1, "UU": 2, "NN": 2,
                      "CI": 2, "NC": 2, "UW": 3, "AT": 3, "AV": 3,
                      "AK": 4, "HV": 4, "CN": 4}
    colors = []
    for net in networks:
        c = complexity_map.get(net, 2)
        if c <= 1: colors.append(SEMANTIC["jsT_blue"])
        elif c >= 4: colors.append(SEMANTIC["hawaii_rust"])
        else: colors.append(SEMANTIC["bg_grey"])

    ax.bar(x_pos, ratios, color=colors, edgecolor="white", lw=0.25, width=0.68)
    ax.axhline(y=1.0, color=SEMANTIC["black"], linestyle="-", lw=0.5, alpha=0.65)

    for i, (r, n) in enumerate(zip(ratios, n_stns)):
        ax.text(i, 0.615, str(n), ha="center", va="bottom", fontsize=4.5,
                color=SEMANTIC["axis_grey"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(networks, rotation=45, ha="right", fontsize=5.5)
    ax.set_ylabel("Self-consistency ratio  (JsT / standard HVSR)")
    ax.set_ylim(0.6, 1.18)

    overall = aggregate["ratio"]
    ax.text(0.01, 0.94, f"All networks: {overall:.2f}$\\times$  ({aggregate['n_stations']} stations)",
            transform=ax.transAxes, fontsize=6, va="top", color=SEMANTIC["black"])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SEMANTIC["jsT_blue"], label="Simple geology (basin/craton)"),
        Patch(facecolor=SEMANTIC["bg_grey"], label="Moderate complexity"),
        Patch(facecolor=SEMANTIC["hawaii_rust"], label="Complex (subduction/volcanic)"),
    ]
    ax.legend(handles=legend_elements, fontsize=5.0, loc="lower right",
              frameon=True, framealpha=0.85, edgecolor="#DDDDDD",
              borderpad=0.15, handletextpad=0.35)

    add_panel_label(ax, "b")
    ax.grid(True, alpha=0.10, lw=0.2, axis="y")
    save_panel(fig, OUT / "fig2_panel_b")
    plt.close(fig)
    print(f"  Panel (b): {len(per_net)} networks, overall {overall:.3f}")


# ── panel (c): Hawaii distance-similarity ───────────────────────────────

def panel_c():
    with open("outputs/fig2_cache/hawaii_pairs.json") as f:
        hi = json.load(f)
    with open("outputs/fig2_cache/hawaii_stats.json") as f:
        stats = json.load(f)

    pairs = hi["pairs"]
    distances = np.array([p["distance_km"] for p in pairs])
    similarities = np.array([p["cos_sim"] for p in pairs])
    rho, pval = stats["spearman_rho"], stats["p_value"]

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(MID_COL, 80 / 25.4))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.12)

    ax.scatter(distances, similarities, c=SEMANTIC["hawaii_rust"], s=12, alpha=0.42, edgecolors="none")

    order = np.argsort(distances)
    window = max(3, len(order) // 12)
    trend = uniform_filter1d(similarities[order], size=window)
    ax.plot(distances[order], trend, color=SEMANTIC["black"], lw=1.0, alpha=0.80)

    ax.set_xlabel("Station separation distance (km)")
    ax.set_ylabel("JsT-HVSR cosine similarity")
    ax.set_xlim(left=-3)

    ax.text(0.97, 0.92, f"$\\rho$ = {rho:+.3f}\np = {pval:.1e}\nN = {stats['n_pairs']} pairs",
            transform=ax.transAxes, fontsize=5.8, ha="right", va="top",
            color=SEMANTIC["black"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.80,
                      lw=0.25, edgecolor=SEMANTIC["axis_grey"]))
    ax.text(0.03, 0.92, "no external Vs30 data",
            transform=ax.transAxes, fontsize=5.2, ha="left", va="top",
            color=SEMANTIC["hawaii_rust"], fontstyle="italic")

    add_panel_label(ax, "c")
    ax.grid(True, alpha=0.10, lw=0.2)
    save_panel(fig, OUT / "fig2_panel_c")
    plt.close(fig)
    print(f"  Panel (c): Hawaii ρ={rho:+.3f}")


# ── panel (d): complementarity double dissociation ───────────────────────

def panel_d():
    apply_nature_style()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(DOUBLE_COL, 72 / 25.4))
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.98, top=0.90, bottom=0.15)

    # Data from Exp B follow-up
    classes = [0, 1, 2, 4]  # skip 3 (n=1, no rank ρ)
    xp = np.arange(len(classes))
    cross_cos_vals = [0.650, 0.792, 0.726, 0.633]
    cross_cos_errs = [0.107, 0.096, 0.074, 0.068]
    rank_rho_vals = [CLASS_RANK_RHO[c] for c in classes]
    ns = [CLASS_N[c] for c in classes]
    labels_simple = ["Craton", "Sedimentary\nbasin", "Basin-range", "Subduction /\nvolcanic"]
    class_colors = [SEMANTIC["jsT_blue"], SEMANTIC["jsT_blue"],
                    SEMANTIC["bg_grey"], SEMANTIC["hawaii_rust"]]

    # ── Left: cross-method cos ──
    bars_l = ax_l.bar(xp, cross_cos_vals, color=class_colors, edgecolor="white", lw=0.25, width=0.55)
    ax_l.errorbar(xp, cross_cos_vals, yerr=cross_cos_errs, fmt="none",
                  ecolor=SEMANTIC["black"], lw=0.6, capsize=3, capthick=0.5)
    for i, (v, n) in enumerate(zip(cross_cos_vals, ns)):
        ax_l.text(i, v + cross_cos_errs[i] + 0.015, f"N={n}", ha="center", fontsize=5.5,
                  color=SEMANTIC["axis_grey"])
    ax_l.set_xticks(xp)
    ax_l.set_xticklabels(labels_simple, fontsize=6)
    ax_l.set_ylabel("Cross-method cosine similarity\n(JsT vs standard HVSR)")
    ax_l.set_ylim(0.50, 1.0)
    ax_l.axhline(y=0.690, color=SEMANTIC["null_grey"], linestyle="--", lw=0.5, alpha=0.6)
    ax_l.text(0.5, 0.690 + 0.01, "all-station mean = 0.690", fontsize=5, ha="center",
              color=SEMANTIC["axis_grey"])
    add_panel_label(ax_l, "d")
    ax_l.grid(True, alpha=0.10, lw=0.2, axis="y")

    # ── Right: rank correlation ──
    bar_colors_r = [SEMANTIC["jsT_blue"] if v and v < 0.3 else
                    SEMANTIC["hawaii_rust"] if v and v > 0.3 else
                    SEMANTIC["bg_grey"] for v in rank_rho_vals]
    bars_r = ax_r.bar(xp, [v or 0 for v in rank_rho_vals], color=bar_colors_r,
                      edgecolor="white", lw=0.25, width=0.55)
    # Annotate p-values
    p_annotations = ["p = 0.088", "p = 0.90", "p = 0.90", "p = 0.002"]
    for i, (v, p, n) in enumerate(zip(rank_rho_vals, p_annotations, ns)):
        y_val = (v or 0)
        offset = 0.06 if y_val >= 0 else -0.08
        ax_r.text(i, y_val + offset, f"N={n}  {p}", ha="center", fontsize=5.2,
                  color=SEMANTIC["axis_grey"])
    ax_r.set_xticks(xp)
    ax_r.set_xticklabels(labels_simple, fontsize=6)
    ax_r.set_ylabel("Rank correlation (JsT vs HV)\nSpearman $ρ$")
    ax_r.axhline(y=0, color=SEMANTIC["null_grey"], linestyle="-", lw=0.4, alpha=0.4)
    ax_r.set_ylim(-0.65, 1.05)
    ax_r.grid(True, alpha=0.10, lw=0.2, axis="y")

    # ── Annotation: the inversion ──
    fig.text(0.5, 0.04,
             "In sedimentary basins: similar spectral shapes but uncorrelated rankings\n"
             "In subduction / volcanic: divergent spectral shapes but correlated rankings",
             ha="center", va="center", fontsize=6.2, color=SEMANTIC["black"],
             fontstyle="italic")

    save_panel(fig, OUT / "fig2_panel_d")
    plt.close(fig)
    print("  Panel (d): complementarity double dissociation")


# ── main ────────────────────────────────────────────────────────────────

def main():
    print("Fig 2: Validation & Complementarity")
    panel_a()
    panel_b()
    panel_c()
    panel_d()
    print("Done.")


if __name__ == "__main__":
    main()
