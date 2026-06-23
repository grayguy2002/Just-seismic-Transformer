"""Bottom-to-top neural-network architecture diagram for JsT.

This figure is a pure architecture schematic: inputs/embeddings at the bottom,
the repeated JsT block in the middle and the waveform head at the top.

Run from the project root:
  python3 manuscript/figures/fig4a_jst_bottom_up_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from nature_geo_style import SEMANTIC, apply_nature_style


FIG_DIR = Path(__file__).resolve().parent
OUT_BASE = FIG_DIR / "jst_bottom_up_architecture"

BLUE = SEMANTIC["jsT_blue"]
BLUE_LIGHT = SEMANTIC["jsT_blue_light"]
GREEN = SEMANTIC["token_green_light"]
BLACK = SEMANTIC["black"]
GREY = SEMANTIC["axis_grey"]
EDGE = "#C9CDD0"
PANEL = "#F7F8F8"
SOURCE = "#8CADC8"
PATH = "#C5CFC0"
ZEROED = "#D8D8D8"
PATCH = "#D6D300"
PATCH_DARK = "#C8C000"


def txt(ax, x, y, s, fs=5.8, color=BLACK, ha="center", va="center", weight="normal"):
    ax.text(
        x,
        y,
        s,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        fontweight=weight,
        linespacing=0.94,
        zorder=6,
    )


def box(
    ax,
    x,
    y,
    w,
    h,
    s="",
    face="#FFFFFF",
    edge=EDGE,
    fs=5.8,
    lw=0.72,
    r=0.010,
    weight="normal",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={r}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=1,
    )
    ax.add_patch(patch)
    if s:
        txt(ax, x + w / 2, y + h / 2, s, fs=fs, weight=weight)
    return patch


def arrow(ax, x0, y0, x1, y1, color="#8F8F8F", lw=0.74, ms=6.2, rad=0.0):
    patch = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
        zorder=4,
    )
    ax.add_patch(patch)
    return patch


def token_strip(ax, x, y, w, h):
    colors = [SOURCE] * 3 + [PATH] * 4 + [GREEN] + [ZEROED] * 3
    gap = 0.003
    tw = (w - 10 * gap) / 11
    for i, color in enumerate(colors):
        xi = x + i * (tw + gap)
        box(ax, xi, y, tw, h, "", color, edge="white", lw=0.30, r=0.002)
        txt(ax, xi + tw / 2, y + h / 2, str(i), fs=3.75, weight="bold")
        if i >= 8:
            ax.plot(
                [xi + 0.004, xi + tw - 0.004],
                [y + 0.006, y + h - 0.006],
                transform=ax.transAxes,
                color="#777777",
                lw=0.35,
            )


def patch_strip(ax, x, y, w, h):
    labels = ["0", "1", "2", "...", "47", "48", "49"]
    gap = 0.004
    tw = (w - 6 * gap) / 7
    for i, label in enumerate(labels):
        xi = x + i * (tw + gap)
        face = PATCH_DARK if i in (0, 1, 2, 5, 6) else PATCH
        box(ax, xi, y, tw, h, "", face, edge="white", lw=0.30, r=0.002)
        txt(ax, xi + tw / 2, y + h / 2, label, fs=3.75, weight="bold")


def draw_jst_stack(ax, x=0.170, y=0.405, w=0.660, h=0.305):
    box(ax, x, y, w, h, "", "#FFFFFF", edge=EDGE, lw=0.82, r=0.012)
    txt(ax, x + w / 2, y + h - 0.028, "JsTBlock x8", fs=6.65, weight="bold", va="top")

    ix = x + 0.080 * w
    iw = 0.840 * w
    layer_h = 0.046
    ys = [0.607, 0.545, 0.483, 0.421]
    layers = [
        ("FiLM condition modulation", "#EEF3F6", GREEN),
        ("RMSNorm + multi-head attention\n1-D RoPE on q,k", "#F9FAFA", BLUE_LIGHT),
        ("RMSNorm + SwiGLU feed-forward", "#F9FAFA", "#B8C4AF"),
        ("residual gates", PANEL, EDGE),
    ]

    for i, (label, face, edge) in enumerate(layers):
        box(ax, ix, ys[i], iw, layer_h, label, face, edge=edge, fs=4.82, lw=0.58, r=0.006)
        if i < len(layers) - 1:
            arrow(
                ax,
                ix + iw / 2,
                ys[i] - 0.006,
                ix + iw / 2,
                ys[i + 1] + layer_h + 0.006,
                color="#AFAFAF",
                lw=0.50,
                ms=4.4,
            )

    return x, y, w, h


def main() -> int:
    apply_nature_style(base_size=5.8)
    fig = plt.figure(figsize=(3.75, 6.05), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    cx = 0.500

    # Bottom encoders: same size, symmetric about the centre.
    cond = (0.080, 0.085, 0.365, 0.120)
    patch = (0.555, 0.085, 0.365, 0.120)
    box(ax, *cond, "", PANEL, edge=EDGE, lw=0.74, r=0.010)
    txt(ax, cond[0] + cond[2] / 2, cond[1] + cond[3] - 0.032, "ConditionEncoder v3", fs=5.25)
    token_strip(ax, cond[0] + 0.040, cond[1] + 0.042, cond[2] - 0.080, 0.029)
    txt(ax, cond[0] + cond[2] / 2, cond[1] + 0.022, "11 slots, 8 active", fs=4.10, color=GREY)

    box(ax, *patch, "", "#EEF3F6", edge=BLUE_LIGHT, lw=0.74, r=0.010)
    txt(ax, patch[0] + patch[2] / 2, patch[1] + patch[3] - 0.032, "PatchEmbed1D", fs=5.25)
    patch_strip(ax, patch[0] + 0.050, patch[1] + 0.042, patch[2] - 0.100, 0.029)
    txt(ax, patch[0] + patch[2] / 2, patch[1] + 0.022, "50 patches", fs=4.10, color=GREY)

    # Merge into the transformer token sequence.
    seq = (0.235, 0.285, 0.530, 0.060)
    arrow(ax, cond[0] + cond[2] * 0.62, cond[1] + cond[3] + 0.010, seq[0] + seq[2] * 0.34, seq[1] - 0.004, color="#8F8F8F", lw=0.70, ms=5.8)
    arrow(ax, patch[0] + patch[2] * 0.38, patch[1] + patch[3] + 0.010, seq[0] + seq[2] * 0.66, seq[1] - 0.004, color="#8F8F8F", lw=0.70, ms=5.8)
    box(ax, *seq, "token sequence\ncondition tokens + waveform patches", "#FFFFFF", edge=EDGE, fs=4.95, lw=0.74, r=0.010)

    # Main stack.
    arrow(ax, cx, seq[1] + seq[3] + 0.006, cx, 0.397, color="#8F8F8F", lw=0.74, ms=6.0)
    jst_x, jst_y, jst_w, jst_h = draw_jst_stack(ax)

    # Timestep side modulation, aligned to the block centre and kept compact.
    tbox = (0.852, 0.532, 0.110, 0.060)
    box(ax, *tbox, "Timestep\nEmbedder", PANEL, edge=EDGE, fs=4.40, lw=0.66, r=0.008)
    arrow(ax, tbox[0] - 0.004, tbox[1] + tbox[3] / 2, jst_x + jst_w + 0.004, tbox[1] + tbox[3] / 2, color=BLUE_LIGHT, lw=0.70, ms=5.6)
    txt(ax, (tbox[0] + jst_x + jst_w) / 2, tbox[1] + tbox[3] / 2 + 0.021, "adaLN", fs=4.00, color=BLUE)

    # Output head.
    keep = (0.332, 0.748, 0.336, 0.046)
    arrow(ax, cx, jst_y + jst_h + 0.006, cx, keep[1] - 0.004, color="#8F8F8F", lw=0.74, ms=6.0)
    box(ax, *keep, "keep patch tokens", PANEL, edge=EDGE, fs=4.85, lw=0.68, r=0.008)

    final = (0.250, 0.825, 0.500, 0.068)
    arrow(ax, cx, keep[1] + keep[3] + 0.006, cx, final[1] - 0.004, color="#8F8F8F", lw=0.74, ms=6.0)
    box(ax, *final, "FinalLayer\nRMSNorm + linear + unpatchify", "#EEF3F6", edge=BLUE_LIGHT, fs=4.92, lw=0.74, r=0.010)

    out = (0.310, 0.932, 0.380, 0.050)
    arrow(ax, cx, final[1] + final[3] + 0.006, cx, out[1] - 0.004, color=BLUE, lw=0.86, ms=6.6)
    box(ax, *out, "output waveform", "#FFFFFF", edge=EDGE, fs=5.15, lw=0.74, r=0.010)

    OUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
