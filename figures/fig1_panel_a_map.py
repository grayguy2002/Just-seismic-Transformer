"""Fig 1 panel (a) — Cartopy US map coloured by seismic network.

Requires: cartopy (local-only — lab54 has no cartopy)
Input:    outputs/fig1_global_phenomenon/hvsr_map_data.json (from lab54 GPU run)
Output:   manuscript/figures/fig1_panel_a.{svg,pdf,png}
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nature_geo_style import (
    DOUBLE_COL, SEMANTIC, NETWORK_COLORS, add_panel_label, apply_nature_style, save_panel,
)


def main():
    import cartopy.crs as ccrs
    from nature_geo_style import add_cartopy_base

    data_path = Path("outputs/fig1_global_phenomenon/hvsr_map_data.json")
    with open(data_path) as f:
        hvsr_data = json.load(f)

    # Group stations by network
    by_network = {}
    for sid, d in hvsr_data.items():
        net = d["net"]
        by_network.setdefault(net, ([], [], []))  # lons, lats, sizes
        by_network[net][0].append(d["lon"])
        by_network[net][1].append(d["lat"])
        by_network[net][2].append(9 + 1.25 * np.sqrt(d["n_events"]))

    sta_cont = {net: (lons, lats, sizes) for net, (lons, lats, sizes) in by_network.items()
                if not (18 <= np.mean(lats) <= 23 and -160 <= np.mean(lons) <= -154)}
    # Hawaii: only HV
    sta_hawaii = {}
    if "HV" in by_network:
        sta_hawaii["HV"] = by_network["HV"]

    # Determine which networks to show on main map vs other
    major_nets = [n for n in ["AK", "HV", "OK", "GS", "UU", "UW", "NN", "NM",
                               "CI", "NC", "JP", "TW"] if n in by_network]
    other_nets = [n for n in sorted(by_network) if n not in major_nets]

    apply_nature_style()
    fig = plt.figure(figsize=(DOUBLE_COL, 92 / 25.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[2.25, 1.0],
        left=0.048, right=0.992, top=0.93, bottom=0.09, wspace=0.14,
    )
    ax_us = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_hi = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    add_cartopy_base(ax_us, [-172, -64, 17, 72], land_scale="110m")
    add_cartopy_base(ax_hi, [-156.2, -154.65, 18.85, 20.35], land_scale="10m")

    def scatter_net(ax, lons, lats, sizes, color, alpha, label, zorder=3):
        ax.scatter(lons, lats, s=sizes, c=color, alpha=alpha,
                   edgecolors="white", lw=0.2,
                   transform=ccrs.PlateCarree(), label=label, zorder=zorder)

    # Major networks on main map
    for net in major_nets:
        if net in sta_cont:
            lons, lats, sizes = sta_cont[net]
            scatter_net(ax_us, lons, lats, sizes, NETWORK_COLORS.get(net, SEMANTIC["bg_grey"]),
                        0.72, f"{net} ({len(lons)})")

    # Other networks in grey
    for net in other_nets:
        if net in sta_cont:
            lons, lats, sizes = sta_cont[net]
            scatter_net(ax_us, lons, lats, sizes, SEMANTIC["bg_grey"],
                        0.30, None)

    # Hawaii inset
    if sta_hawaii:
        for net, (lons, lats, sizes) in sta_hawaii.items():
            scatter_net(ax_hi, lons, lats, sizes, NETWORK_COLORS.get(net, SEMANTIC["hawaii_rust"]),
                        0.78, None)

    ax_hi.set_title("Hawaii", loc="left", pad=3, fontsize=6.5,
                    color=SEMANTIC["hawaii_rust"])
    add_panel_label(ax_us, "a", x=-0.04, y=1.02)

    n_sta = len(hvsr_data)
    ax_us.text(
        0.0, -0.09,
        f"{n_sta} stations  |  circle area = event count  |  colour = seismic network",
        transform=ax_us.transAxes, ha="left", va="top",
        fontsize=5.8, color=SEMANTIC["axis_grey"],
    )

    # Legend
    legend_handles = []
    for net in major_nets:
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], marker="o", color="w",
                                     markerfacecolor=NETWORK_COLORS.get(net, SEMANTIC["bg_grey"]),
                                     markersize=5, lw=0, label=f"{net}"))
    ax_us.legend(handles=legend_handles, fontsize=5.0, loc="lower left",
                 bbox_to_anchor=(0.012, 0.018), ncol=2,
                 frameon=True, framealpha=0.92, facecolor="white",
                 edgecolor="#DDDDDD", handletextpad=0.22, columnspacing=0.65,
                 borderpad=0.2, markerscale=0.72)

    out_dir = Path(__file__).resolve().parent
    save_panel(fig, out_dir / "fig1_panel_a")
    plt.close(fig)
    print(f"Panel (a) map saved: {n_sta} stations, {len(major_nets)} networks")


if __name__ == "__main__":
    main()
