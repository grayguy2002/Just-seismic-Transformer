"""Figure 1 render — Phenomenon: the residual as a reproducible site signature.

Four panels (2×2 grid for manual assembly):
  (a) Cartopy US map — stations coloured by network, circle area ∝ event count, Hawaii inset
  (b) 150-station overview scatter — mean JsT-HVSR by geological group, key groups highlighted
  (c) Three representative stations — multi-event JsT-HVSR curves overlaid (OK.CROK, HV.PAUD, AK.RC01)
  (d) Intra vs inter Pearson r histogram + null distribution

All from cache: outputs/fig1_cache/{station_hvsr.json, stats.json}
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nature_geo_style import (
    MID_COL, DOUBLE_COL, SEMANTIC, GEO_COLORS, GEO_HIGHLIGHT, NETWORK_COLORS,
    add_panel_label, apply_nature_style, save_panel,
)

CACHE = Path("outputs/fig1_cache")
OUT = Path(__file__).resolve().parent.parent / "figures"

FREQ_EDGES = np.logspace(np.log10(0.3), np.log10(15.0), 41)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])

REPRESENTATIVE = {"OK.CROK": "Sedimentary basin", "HV.PAUD": "Kilauea basalt", "AK.RC01": "Subduction zone"}
SITE_COLORS = [SEMANTIC["jsT_blue"], SEMANTIC["hawaii_rust"], SEMANTIC["std_grey"]]


# ── helpers ──────────────────────────────────────────────────────────────

def loess_smooth(x, y, frac=0.2):
    x, y = np.asarray(x), np.asarray(y)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    window = max(3, int(n * frac))
    smoothed = np.zeros(n)
    for i in range(n):
        left, right = max(0, i - window // 2), min(n, i - window // 2 + window)
        left = max(0, right - window)
        xi, yi = xs[left:right], ys[left:right]
        dist = np.abs(xi - xs[i])
        std = dist.std()
        if std < 1e-12:
            smoothed[i] = yi.mean()
        else:
            w = np.exp(-0.5 * (dist / std) ** 2)
            smoothed[i] = np.dot(w, yi) / (w.sum() + 1e-12)
    return smoothed[np.argsort(order)]


# ── panel (a): Cartopy map ──────────────────────────────────────────────

def panel_a(hvsr_data):
    try:
        import cartopy.crs as ccrs
        from nature_geo_style import add_cartopy_base
    except ImportError:
        print("  Panel (a) SKIPPED — cartopy not installed")
        return

    by_network = {}
    for sid, d in hvsr_data.items():
        net = d["net"]
        by_network.setdefault(net, ([], [], []))
        by_network[net][0].append(d["lon"])
        by_network[net][1].append(d["lat"])
        by_network[net][2].append(9 + 1.25 * np.sqrt(d["n_events"]))

    major_nets = [n for n in ["AK", "HV", "OK", "GS", "UU", "UW", "NN", "NM",
                               "CI", "NC", "JP", "TW"] if n in by_network]

    apply_nature_style()
    fig = plt.figure(figsize=(MID_COL, 88 / 25.4))
    gs = fig.add_gridspec(1, 1, left=0.06, right=0.98, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())

    add_cartopy_base(ax, [-172, -64, 17, 72], land_scale="110m")

    for net in major_nets:
        lons, lats, sizes = by_network[net]
        mask = np.array([not (18 <= la <= 23 and -160 <= lo <= -154)
                         for lo, la in zip(lons, lats)])
        if mask.sum() > 0:
            ax.scatter(np.array(lons)[mask], np.array(lats)[mask],
                       s=np.array(sizes)[mask], c=NETWORK_COLORS.get(net, SEMANTIC["bg_grey"]),
                       alpha=0.74, edgecolors="white", lw=0.2,
                       transform=ccrs.PlateCarree(), label=f"{net} ({int(mask.sum())})", zorder=3)

    # Hawaii inset axes
    ax_inset = fig.add_axes([0.60, 0.55, 0.18, 0.35], projection=ccrs.PlateCarree())
    add_cartopy_base(ax_inset, [-156.2, -154.65, 18.85, 20.35], land_scale="10m")
    if "HV" in by_network:
        lons, lats, sizes = by_network["HV"]
        mask = np.array([18 <= la <= 23 and -160 <= lo <= -154 for lo, la in zip(lons, lats)])
        if mask.sum() > 0:
            ax_inset.scatter(np.array(lons)[mask], np.array(lats)[mask],
                             s=np.array(sizes)[mask], c=NETWORK_COLORS["HV"], alpha=0.78,
                             edgecolors="white", lw=0.2, transform=ccrs.PlateCarree(), zorder=3)
    ax_inset.set_title("Hawaii", fontsize=5.5, color=SEMANTIC["hawaii_rust"], pad=2)

    add_panel_label(ax, "a", x=-0.04, y=1.02)

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w",
                       markerfacecolor=NETWORK_COLORS.get(n, SEMANTIC["bg_grey"]),
                       markersize=4.5, lw=0, label=n) for n in major_nets[:10]]
    ax.legend(handles=handles, fontsize=4.5, loc="lower left",
              bbox_to_anchor=(0.008, 0.012), ncol=3,
              frameon=True, framealpha=0.90, facecolor="white",
              edgecolor="#DDDDDD", handletextpad=0.2, columnspacing=0.5,
              borderpad=0.15, markerscale=0.7)

    ax.text(0.0, -0.06, f"{len(hvsr_data)} stations  |  circle area ~ event count  |  colour = network",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.5, color=SEMANTIC["axis_grey"])

    save_panel(fig, OUT / "fig1_panel_a")
    plt.close(fig)
    print(f"  Panel (a): {len(hvsr_data)} stations on US map")


# ── panel (b): station overview scatter ─────────────────────────────────

def panel_b(station_ids, geo_labels, hvsr_amps, td_stats):
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(MID_COL, 88 / 25.4))
    fig.subplots_adjust(left=0.10, right=0.78, top=0.93, bottom=0.12)

    geo_set = sorted(set(geo_labels))
    station_order = []
    for g in geo_set:
        group = [s for s, gl in zip(station_ids, geo_labels) if gl == g]
        group.sort(key=lambda s: hvsr_amps[s])
        station_order.extend(group)

    x_pos = np.arange(len(station_order))
    y_amps = np.array([hvsr_amps[s] for s in station_order])
    colors = [GEO_COLORS.get(geo_labels[station_ids.index(s)], "#C8C8C8") for s in station_order]
    alpha_map = {g: 0.80 if g in GEO_HIGHLIGHT else 0.45 for g in geo_set}
    alphas = [alpha_map[geo_labels[station_ids.index(s)]] for s in station_order]

    ax.scatter(x_pos, y_amps, c=colors, s=9, alpha=alphas, edgecolors="none", lw=0)
    smooth = loess_smooth(x_pos.astype(float), y_amps, frac=0.15)
    ax.plot(x_pos, smooth, color=SEMANTIC["black"], lw=0.8, alpha=0.75)
    ax.axhline(y=0, color=SEMANTIC["null_grey"], linestyle="--", lw=0.4, alpha=0.5)
    ax.set_xlabel("Station index (grouped by geological context)")
    ax.set_ylabel("Mean JsT-HVSR amplification")
    ax.tick_params(labelsize=5.8)

    ax.text(0.01, 0.96,
            f"intra r = {td_stats['intra']:.3f}    inter r = {td_stats['inter']:.3f}    "
            f"$\\Delta$ = {td_stats['delta']:+.3f} ({td_stats['ratio']:.1f}$\\times$)",
            transform=ax.transAxes, fontsize=5.5, va="top", color=SEMANTIC["black"])

    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=GEO_COLORS.get(g, "#C8C8C8"),
                       markersize=4, lw=0) for g in geo_set]
    labels = [g.replace("_", " ") for g in geo_set]
    ax.legend(handles, labels, fontsize=4.0, loc="upper left", bbox_to_anchor=(1.005, 1),
              ncol=1, frameon=True, framealpha=0.85, edgecolor="#DDDDDD",
              borderpad=0.12, handletextpad=0.25, markerscale=0.9)

    add_panel_label(ax, "b")
    save_panel(fig, OUT / "fig1_panel_b")
    plt.close(fig)
    print("  Panel (b): 150-station overview")


# ── panel (c): representative stations ──────────────────────────────────

def panel_c(hvsr_data):
    apply_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 62 / 25.4))
    fig.subplots_adjust(wspace=0.30, left=0.05, right=0.98, top=0.88, bottom=0.20)

    for ax_idx, (sta_id, label) in enumerate(REPRESENTATIVE.items()):
        ax = axes[ax_idx]
        if sta_id not in hvsr_data or len(hvsr_data[sta_id]["curves"]) < 2:
            ax.text(0.5, 0.5, "insufficient events", ha="center", va="center",
                    fontsize=7, color=SEMANTIC["axis_grey"], transform=ax.transAxes)
            ax.set_title(label, fontsize=7, pad=3)
            continue

        curves = np.array(hvsr_data[sta_id]["curves"])
        n = len(curves)
        for ci in range(n):
            alpha_val = 0.28 + 0.28 * (ci / max(n - 1, 1))
            ax.plot(FREQ_CENTERS, curves[ci], lw=0.45, alpha=alpha_val,
                    color=SITE_COLORS[ax_idx], drawstyle="steps-mid")
        mean_curve = curves.mean(axis=0)
        ax.plot(FREQ_CENTERS, mean_curve, color=SEMANTIC["black"], lw=1.0)
        ax.axhline(y=0, color=SEMANTIC["null_grey"], linestyle="--", lw=0.35, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        if ax_idx == 0:
            ax.set_ylabel("JsT-HVSR  log$_{10}$(res / pred)")
        ax.tick_params(labelsize=5.8)
        ax.set_title(f"{sta_id}  ({label})", fontsize=6.5, color=SEMANTIC["black"], pad=3)
        ax.grid(True, alpha=0.12, lw=0.2)

        all_r = [pearsonr(curves[i], curves[j])[0] for i in range(n) for j in range(i+1, n)]
        intra_r = np.mean(all_r) if all_r else 0
        ax.text(0.97, 0.88, f"intra r = {intra_r:.2f}\nn = {n}",
                transform=ax.transAxes, fontsize=5.2, ha="right", va="top",
                color=SEMANTIC["black"],
                bbox=dict(facecolor="white", alpha=0.72, lw=0.25, pad=1.2,
                          edgecolor=SEMANTIC["axis_grey"]))

    add_panel_label(axes[0], "c", x=-0.10)
    save_panel(fig, OUT / "fig1_panel_c")
    plt.close(fig)
    print("  Panel (c): 3 representative stations")


# ── panel (d): intra/inter histogram ────────────────────────────────────

def panel_d(stats):
    apply_nature_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(MID_COL, 88 / 25.4),
        gridspec_kw={"height_ratios": [2.0, 1.0]})
    fig.subplots_adjust(hspace=0.28, left=0.13, right=0.97, top=0.93, bottom=0.09)

    bins = np.linspace(-0.4, 1.0, 50)
    ax_top.hist(stats["intra_vals"], bins=bins, alpha=0.55, color=SEMANTIC["jsT_blue"],
                label="Same station (intra)", density=True)
    ax_top.hist(stats["inter_vals"], bins=bins, alpha=0.45, color=SEMANTIC["bg_grey"],
                label="Different station (inter)", density=True)
    ax_top.axvline(stats["intra"], color=SEMANTIC["jsT_blue"], lw=0.9)
    ax_top.axvline(stats["inter"], color=SEMANTIC["std_grey"], lw=0.9)
    ax_top.legend(fontsize=5.8, loc="upper left", frameon=True, framealpha=0.88,
                  edgecolor="#DDDDDD", borderpad=0.2, handletextpad=0.4)
    ax_top.set_ylabel("Density")
    ax_top.tick_params(labelsize=5.8)
    ax_top.text(0.98, 0.90,
                f"intra = {stats['intra']:.3f}\ninter = {stats['inter']:.3f}\n"
                f"$\\Delta$ = {stats['delta']:+.3f} ({stats['ratio']:.1f}$\\times$)",
                transform=ax_top.transAxes, fontsize=5.5, ha="right", va="top",
                color=SEMANTIC["black"],
                bbox=dict(facecolor="white", alpha=0.80, lw=0.3, pad=1.5,
                          edgecolor=SEMANTIC["axis_grey"]))

    bins_null = np.linspace(-0.25, 0.25, 30)
    ax_bot.hist(stats["null_intra_vals"], bins=bins_null, alpha=0.50, color=SEMANTIC["bg_grey"],
                label="Null intra (shuffled)", density=True)
    ax_bot.hist(stats["null_inter_vals"], bins=bins_null, alpha=0.35, color=SEMANTIC["bg_grey_warm"],
                label="Null inter", density=True)
    ax_bot.axvline(stats["null_intra"], color=SEMANTIC["std_grey"], lw=0.7)
    ax_bot.axvline(stats["null_inter"], color=SEMANTIC["null_grey"], lw=0.7)
    ax_bot.legend(fontsize=5.5, loc="upper right", frameon=True, framealpha=0.88,
                  edgecolor="#DDDDDD", borderpad=0.15, handletextpad=0.3)
    ax_bot.set_xlabel("Time-domain residual Pearson r")
    ax_bot.set_ylabel("Density")
    ax_bot.tick_params(labelsize=5.8)
    ax_bot.text(0.98, 0.85, f"null $\\Delta$ = {stats['null_delta']:+.4f}",
                transform=ax_bot.transAxes, fontsize=5.2, ha="right", va="top",
                color=SEMANTIC["black"])

    add_panel_label(ax_top, "d")
    save_panel(fig, OUT / "fig1_panel_d")
    plt.close(fig)
    print("  Panel (d): intra/inter histogram")


# ── main ────────────────────────────────────────────────────────────────

def main():
    with open(CACHE / "station_hvsr.json") as f:
        hvsr_data = json.load(f)
    with open(CACHE / "stats.json") as f:
        td_stats = json.load(f)

    hvsr_amps = {sid: float(np.mean(np.array(d["curves"]))) for sid, d in hvsr_data.items()}
    station_ids = sorted(hvsr_data.keys())
    geo_labels = [hvsr_data[s]["geology"] for s in station_ids]

    print(f"Fig 1: {len(hvsr_data)} stations, intra={td_stats['intra']:.3f} inter={td_stats['inter']:.3f}")
    panel_a(hvsr_data)
    panel_b(station_ids, geo_labels, hvsr_amps, td_stats)
    panel_c(hvsr_data)
    panel_d(td_stats)
    print("Done.")


if __name__ == "__main__":
    main()
