"""Shared Nature Geoscience figure helpers for JsT manuscript panels.

Design rules (from figure-design-brief.md):
  - This is a measurement paper, not an ML benchmark.
  - Color is semantic. Do not map the same concept to different colours across figures.
  - Reserve saturated colour for the data subset that carries the claim.
  - Remove top/right spines, large titles, decorative boxes, slide-style conclusions.
  - Every panel exposes its sample size, statistic, or direct visual evidence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


# ── Nature physical sizes ────────────────────────────────────────────────

MM_TO_IN = 1.0 / 25.4
SINGLE_COL = 89 * MM_TO_IN      #  89 mm — single-column
MID_COL    = 120 * MM_TO_IN      # 120 mm — medium
DOUBLE_COL = 183 * MM_TO_IN      # 183 mm — full double-column


# ── Semantic colour grammar ──────────────────────────────────────────────

SEMANTIC = {
    # Primary signal: JsT prediction, residual, measurement
    "jsT_blue":       "#1B5A8C",
    "jsT_blue_light": "#7BA3C4",

    # Reference / standard method
    "std_grey":       "#555555",
    "std_grey_light": "#AAAAAA",

    # Background / all-data context (never the claim-carrier)
    "bg_grey":        "#C4C4C4",
    "bg_grey_warm":   "#D5D0CB",

    # Volcanic/geological accent, retained for legacy render scripts.
    "volcanic_rust":  "#C5634B",
    "volcanic_light": "#E0A394",
    "hawaii_rust":    "#C5634B",
    "hawaii_light":   "#E0A394",

    # Receiver-token mechanism / geological representation
    "token_green":    "#4F7D6B",
    "token_green_light": "#98B7A8",

    # Negative / underperforming (used sparingly)
    "neg_red":        "#B94A48",
    "neg_red_light":  "#D99392",

    # Threshold / null reference
    "null_grey":      "#888888",

    # Text / structure
    "black":          "#222222",
    "axis_grey":      "#999999",
}

# Geological group colours — muted palette covering all 17 geological groups.
# Key geological groups get stronger saturation.
# Others use low-saturation variants so no station is grey.
GEO_COLORS = {
    "Basalt_Kilauea":        SEMANTIC["volcanic_rust"],
    "Basalt_coastal":        "#E8C5BA",
    "Basalt_flank":          "#D49E8E",
    "Basalt_weathered":      "#C0806A",
    "Sedimentary_centralUS": SEMANTIC["jsT_blue"],
    "Sedimentary_basin":     SEMANTIC["jsT_blue_light"],
    "Sedimentary_coastal":   "#8CADC8",
    "Sedimentary_embayment": "#B0C5D5",
    "Metamorphic_interior":  "#9B8CAC",
    "Metamorphic_range":     "#BDB0CB",
    "Active_margin":         "#A0B0A0",
    "Passive_margin_east":   "#C5CFC0",
    "Basin_range":           "#C0B8A8",
    "Craton_north":          "#C8C0B8",
    "Interior_platform":     "#D0CCC4",
    "Volcanic_arc":          "#C8A898",
    "Volcanic_cascades":     "#A8B898",
}

# Backward-compatible alias: key groups get stronger alpha in scatter plots
GEO_HIGHLIGHT = {
    "Basalt_Kilauea":        SEMANTIC["volcanic_rust"],
    "Sedimentary_centralUS": SEMANTIC["jsT_blue"],
    "Sedimentary_basin":     SEMANTIC["jsT_blue_light"],
}

GEO_LOW = {"default": "#D8D8D8"}

# Per-network colours — kept desaturated; used only where network identity
# matters (e.g. Fig 2b bar chart or supplementary map).
NETWORK_COLORS = {
    "AK": "#6B7B9A",
    "HV": SEMANTIC["volcanic_rust"],
    "OK": "#9B8D6B",
    "GS": "#8C9C8C",
    "UU": "#8B7B9E",
    "UW": "#6B8B7E",
    "NN": "#7C8C6B",
    "NM": "#9B7D6B",
    "CI": "#6B7D9E",
    "NC": "#7B8D9E",
    "JP": "#A07B6B",
    "TW": "#9B7C8E",
}


def geo_group_color(group_name: str) -> str:
    """Return highlight colour for key geological groups, low-grey otherwise."""
    return GEO_HIGHLIGHT.get(group_name, GEO_LOW["default"])


# ── Matplotlib global style ──────────────────────────────────────────────

def apply_nature_style(base_size: float = 6.6) -> None:
    """Apply compact Nature-style Matplotlib settings with editable SVG text."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": base_size,
        "axes.titlesize": base_size + 0.4,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.4,
        "ytick.labelsize": base_size - 0.4,
        "legend.fontsize": base_size - 0.6,
        "axes.linewidth": 0.45,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ── Panel annotation helpers ─────────────────────────────────────────────

def add_panel_label(ax, label: str, x: float = -0.08, y: float = 1.03) -> None:
    """Bold lowercase panel letter near upper-left edge."""
    ax.text(
        x, y, label,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=7.2, fontweight="bold", color=SEMANTIC["black"],
    )


def add_stat_annotation(ax, text: str, x: float = 0.02, y: float = 0.95,
                        ha: str = "left", va: str = "top") -> None:
    """Small data-adjacent statistic annotation."""
    ax.text(
        x, y, text,
        transform=ax.transAxes, fontsize=5.8,
        ha=ha, va=va, color=SEMANTIC["black"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  alpha=0.85, lw=0.4, edgecolor=SEMANTIC["axis_grey"]),
    )


def save_panel(fig, output_base: Path | str, dpi: int = 600) -> None:
    """Save editable vector plus high-resolution raster outputs."""
    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


# ── Cartopy helpers ──────────────────────────────────────────────────────

def cartopy_available() -> bool:
    try:
        import cartopy.crs   # noqa: F401
        import cartopy.feature  # noqa: F401
        return True
    except Exception:
        return False


def add_cartopy_base(ax, extent, land_scale: str = "110m") -> None:
    """Add restrained map context to a Cartopy GeoAxes."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND.with_scale(land_scale),
                   facecolor="#F2F1EC", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale(land_scale),
                   facecolor="#EEF3F6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale(land_scale),
                   linewidth=0.35, edgecolor="#777777", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale(land_scale),
                   linewidth=0.25, edgecolor="#A0A0A0", zorder=1)
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.2, color="#B8B8B8", alpha=0.7, linestyle="-",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 5.6, "color": "#555555"}
    gl.ylabel_style = {"size": 5.6, "color": "#555555"}
