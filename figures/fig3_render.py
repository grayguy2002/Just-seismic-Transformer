"""Figure 3 render — Mechanism: emergent geological knowledge in token 7.

Three panels:
  (a) Token 7 PCA projection, 587 stations, key geological groups labelled directly
  (b) Intra-group cosine similarity bar chart, Δ=+0.098, Kilauea 0.994
  (c) Three-scale evidence: global (ECEF removal), local (Kilauea vs coastal),
      continental (OK basin vs GS basin) — geology, not geography

Data: outputs/fig3_cache/{pca_projection.npz, intra_summary.json},
      outputs/expC_token7_disentangle/results.json
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nature_geo_style import (
    MID_COL, DOUBLE_COL, SEMANTIC, GEO_COLORS, GEO_HIGHLIGHT,
    add_panel_label, apply_nature_style, save_panel,
)

OUT = Path(__file__).resolve().parent.parent / "figures"


# ── panel (a): Token 7 PCA projection ──────────────────────────────────

def panel_a():
    data = np.load("outputs/fig3_cache/pca_projection.npz", allow_pickle=True)
    X_pca_2d = data["X_pca_2d"]
    station_ids = data["station_ids"]
    geo_labels = data["geo_labels"]
    geo_set = sorted(set(geo_labels))
    group_sizes = {g: sum(1 for l in geo_labels if l == g) for g in geo_set}

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(MID_COL, 82 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.94, bottom=0.08)

    geo_sorted = sorted(geo_set, key=lambda g: group_sizes[g], reverse=True)
    for g in geo_sorted:
        mask = np.array([l == g for l in geo_labels])
        color = GEO_COLORS.get(g, "#C8C8C8")
        ax.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                   c=[color], s=7, alpha=0.52 if g in GEO_HIGHLIGHT else 0.28,
                   edgecolors="none", lw=0)

    KEY = ["Basalt_Kilauea", "Sedimentary_centralUS", "Sedimentary_basin",
           "Metamorphic_range"]
    for g in KEY:
        if g not in geo_set: continue
        mask = np.array([l == g for l in geo_labels])
        xy = np.median(X_pca_2d[mask], axis=0)
        ax.text(xy[0], xy[1], g.replace("_", " "), fontsize=5.2,
                color=GEO_COLORS.get(g, SEMANTIC["black"]),
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=1.0))

    ax.set_xlabel("Token 7 PC1")
    ax.set_ylabel("Token 7 PC2")
    ax.text(0.02, 0.97, f"{len(station_ids)} stations  {len(geo_set)} geological groups",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=5.5, color=SEMANTIC["axis_grey"])

    add_panel_label(ax, "a")
    ax.grid(True, alpha=0.08, lw=0.18)
    save_panel(fig, OUT / "fig3_panel_a")
    plt.close(fig)
    print("  Panel (a): Token 7 PCA")


# ── panel (b): intra-group cosine bar ──────────────────────────────────

def panel_b():
    with open("outputs/fig3_cache/intra_summary.json") as f:
        data = json.load(f)

    groups = sorted(data["groups"].items(), key=lambda x: x[1]["mean_cos"], reverse=True)
    names = [g[0] for g in groups]
    mean_cos = [g[1]["mean_cos"] for g in groups]
    n_stns = [g[1]["n_stations"] for g in groups]
    all_intra = data["all_intra"]
    all_inter = data["all_inter"]

    apply_nature_style()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 62 / 25.4))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.20)

    x_pos = np.arange(len(names))
    colors = []
    for g in names:
        if g == "Basalt_Kilauea": colors.append(SEMANTIC["hawaii_rust"])
        elif g == "Sedimentary_centralUS": colors.append(SEMANTIC["jsT_blue"])
        elif g == "Sedimentary_basin": colors.append(SEMANTIC["jsT_blue_light"])
        else: colors.append(SEMANTIC["bg_grey"])

    ax.bar(x_pos, mean_cos, color=colors, edgecolor="white", lw=0.25, width=0.68)

    for i, (g, mc, n) in enumerate(zip(names, mean_cos, n_stns)):
        if mc > 0.75:
            ax.text(i, mc + 0.008, str(n), ha="center", fontsize=4.5,
                    color=SEMANTIC["axis_grey"])

    ax.axhline(y=all_intra, color=SEMANTIC["jsT_blue"], linestyle="--", lw=0.7, alpha=0.55,
               label=f"All intra = {all_intra:.3f}")
    ax.axhline(y=all_inter, color=SEMANTIC["std_grey"], linestyle="--", lw=0.7, alpha=0.55,
               label=f"All inter = {all_inter:.3f}")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=45, ha="right", fontsize=5.0)
    ax.set_ylabel("Intra-group token 7 cosine similarity")
    ax.set_ylim(0, 1.08)

    ki = names.index("Basalt_Kilauea") if "Basalt_Kilauea" in names else None
    if ki is not None:
        ax.annotate(f"Kilauea  {mean_cos[ki]:.3f}",
                    xy=(ki, mean_cos[ki]),
                    xytext=(ki + 2.0, mean_cos[ki] + 0.14),
                    arrowprops=dict(arrowstyle="->", color=SEMANTIC["hawaii_rust"], lw=0.6),
                    fontsize=5.5, ha="center", color=SEMANTIC["hawaii_rust"],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.2))

    ax.legend(fontsize=5.5, loc="upper right", frameon=True, framealpha=0.85,
              edgecolor="#DDDDDD", borderpad=0.12, handletextpad=0.3)
    delta = all_intra - all_inter
    ax.text(0.02, 0.96, f"$\\Delta$ = {delta:+.3f} ({all_intra/all_inter:.1f}$\\times$)",
            transform=ax.transAxes, fontsize=5.8, va="top", color=SEMANTIC["black"])

    add_panel_label(ax, "b")
    ax.grid(True, alpha=0.10, lw=0.2, axis="y")
    save_panel(fig, OUT / "fig3_panel_b")
    plt.close(fig)
    print("  Panel (b): intra-group cosine")


# ── panel (c): three-scale geological evidence ──────────────────────────

def panel_c():
    with open("outputs/expC_token7_disentangle/results.json") as f:
        expc = json.load(f)

    apply_nature_style()
    fig = plt.figure(figsize=(DOUBLE_COL, 90 / 25.4))
    gs = fig.add_gridspec(3, 1, left=0.10, right=0.98, top=0.94, bottom=0.06,
                          hspace=0.55, height_ratios=[1, 1, 1])

    # ── Row 1: Global — ECEF removal ──
    ax1 = fig.add_subplot(gs[0])
    t1 = expc["test1_coordinate_removed"]
    delta_orig = t1["original_delta"]
    delta_ecef = t1["ecef_removed_delta"]
    retained = t1["delta_retained_ratio"]
    xp = [0, 1]
    bars1 = ax1.bar(xp, [delta_orig, delta_ecef],
                    color=[SEMANTIC["jsT_blue"], SEMANTIC["token_green"]],
                    edgecolor="white", lw=0.25, width=0.40)
    ax1.set_xticks(xp)
    ax1.set_xticklabels(["Original\ntoken 7", "ECEF coordinates\nremoved"], fontsize=6.5)
    ax1.set_ylabel("$\\Delta$ (intra $-$ inter)")
    ax1.set_title("Global scale: geological clustering survives coordinate removal",
                  fontsize=7, color=SEMANTIC["black"], pad=3, loc="left")
    ax1.text(0.5, 0.92, f"Δ retained = {retained:.0%}",
             transform=ax1.transAxes, ha="center", fontsize=6.5,
             color=SEMANTIC["token_green"], fontweight="bold",
             bbox=dict(facecolor="white", alpha=0.75, lw=0.25, pad=1.5,
                       edgecolor=SEMANTIC["token_green"]))
    ax1.text(0.99, 0.92,
             f"ECEF explains\n< 1% of token 7\nvariance",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=5.2, color=SEMANTIC["axis_grey"])
    ax1.grid(True, alpha=0.10, lw=0.2, axis="y")

    # ── Row 2: Local — Kilauea vs coastal within Hawaii ──
    ax2 = fig.add_subplot(gs[1])
    t2 = expc["test2_hawaii_same_region"]
    ki_intra = t2["kilauea_intra"]
    ki_coastal = t2["kilauea_vs_coastal"]
    ki_flank = t2["kilauea_vs_flank"]
    labels2 = ["Within\nKilauea basalt", "Kilauea vs\ncoastal sediment", "Kilauea vs\nflank basalt"]
    vals2 = [ki_intra, ki_coastal, ki_flank]
    colors2 = [SEMANTIC["hawaii_rust"], SEMANTIC["bg_grey_warm"], SEMANTIC["bg_grey"]]
    ax2.bar([0, 1, 2], vals2, color=colors2, edgecolor="white", lw=0.25, width=0.45)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(labels2, fontsize=6.5)
    ax2.set_ylabel("Token 7 cosine\nsimilarity")
    ax2.set_ylim(0.65, 1.02)
    ax2.set_title("Intra-island scale (Hawaii, all < 100 km): geology resolves within single island",
                  fontsize=7, color=SEMANTIC["black"], pad=3, loc="left")
    # Annotate deltas
    ax2.annotate("", xy=(1, ki_coastal), xytext=(0, ki_intra),
                 arrowprops=dict(arrowstyle="<->", color=SEMANTIC["black"], lw=0.5))
    ax2.text(0.5, (ki_intra + ki_coastal) / 2,
             f"$\\Delta$ = {ki_intra - ki_coastal:+.3f}", ha="center", va="center",
             fontsize=5.8, color=SEMANTIC["black"],
             bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=0.5))
    ax2.grid(True, alpha=0.10, lw=0.2, axis="y")

    # ── Row 3: Continental — cross-region same geology ──
    ax3 = fig.add_subplot(gs[2])
    t3 = expc["test3_cross_region_basins"]
    within = t3["within_basin"]
    cross = t3["cross_basin"]
    groups3 = ["OK basin\n(within)", "GS centralUS\n(within)", "OK$-$GS\n(cross 1000+ km)"]
    vals3 = [within["Sedimentary_basin"], within["Sedimentary_centralUS"],
             cross["Sedimentary_basin_vs_Sedimentary_centralUS"]]
    colors3 = [SEMANTIC["jsT_blue"], SEMANTIC["jsT_blue"], SEMANTIC["token_green"]]
    ax3.bar([0, 1, 2], vals3, color=colors3, edgecolor="white", lw=0.25, width=0.45)
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(groups3, fontsize=6.5)
    ax3.set_ylabel("Token 7 cosine\nsimilarity")
    ax3.set_ylim(0, 1.05)
    ax3.set_title(
        "Inter-continental scale: same geology clusters across > 1,000 km",
        fontsize=7, color=SEMANTIC["black"], pad=3, loc="left")
    ax3.grid(True, alpha=0.10, lw=0.2, axis="y")

    add_panel_label(fig.axes[0], "c")
    save_panel(fig, OUT / "fig3_panel_c")
    plt.close(fig)
    print("  Panel (c): three-scale geological evidence")


# ── main ────────────────────────────────────────────────────────────────

def main():
    print("Fig 3: Geological knowledge in representation")
    panel_a()
    panel_b()
    panel_c()
    print("Done.")


if __name__ == "__main__":
    main()
