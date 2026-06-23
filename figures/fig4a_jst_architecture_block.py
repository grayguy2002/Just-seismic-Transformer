"""Draw a standalone block diagram of the JsT code architecture.

The diagram is derived from:
  - JsT/condition_encoder.py
  - JsT/model.py
  - JsT/denoiser.py
  - outputs/run036/args.json

Run from the project root:
  python3 manuscript/figures/fig4a_jst_architecture_block.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from nature_geo_style import SEMANTIC, apply_nature_style


FIG_DIR = Path(__file__).resolve().parent
OUT_BASE = FIG_DIR / "jst_code_architecture_block"
OUT_CLEAN_BASE = FIG_DIR / "jst_code_architecture_block_clean"

SOURCE = "#8CADC8"
PATH = "#C5CFC0"
RECEIVER = SEMANTIC["token_green_light"]
ZEROED = "#D8D8D8"
PATCH = "#D6D300"
PATCH_DARK = "#C8C000"
BLUE = SEMANTIC["jsT_blue"]
BLUE_LIGHT = SEMANTIC["jsT_blue_light"]
BLACK = SEMANTIC["black"]
GREY = SEMANTIC["axis_grey"]
PANEL = "#F7F8F8"
EDGE = "#C9CDD0"
WARM = "#F2F1EC"


def box(
    ax,
    x,
    y,
    w,
    h,
    text,
    face,
    edge=EDGE,
    fs=6.0,
    weight="normal",
    color=BLACK,
    lw=0.75,
    r=0.010,
    linespacing=0.95,
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
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fs,
            color=color,
            fontweight=weight,
            linespacing=linespacing,
        )
    return patch


def label(ax, x, y, text, fs=5.4, color=GREY, ha="center", va="center", weight="normal"):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        fontweight=weight,
        linespacing=0.95,
    )


def arrow(ax, a, b, color="#8F8F8F", lw=0.85, ms=8, rad=0.0):
    p = FancyArrowPatch(
        a,
        b,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(p)
    return p


def mini_wave(ax, x, y, w, h, color=BLUE, lw=1.05):
    t = np.linspace(0, 1, 260)
    env = np.exp(-3.4 * t)
    sig = (np.sin(2 * np.pi * 7.0 * t) + 0.35 * np.sin(2 * np.pi * 15.0 * t + 0.7)) * env
    ax.plot(
        x + w * t,
        y + h * (0.5 + 0.42 * sig),
        transform=ax.transAxes,
        color=color,
        lw=lw,
        solid_capstyle="round",
    )


def token_strip(ax, x, y, w, h):
    labels = [str(i) for i in range(11)]
    colors = [SOURCE] * 3 + [PATH] * 4 + [RECEIVER] + [ZEROED] * 3
    gap = 0.0032
    tw = (w - gap * 10) / 11
    for i, (lab, col) in enumerate(zip(labels, colors)):
        xi = x + i * (tw + gap)
        box(ax, xi, y, tw, h, lab, col, edge="white", fs=5.3, weight="bold", lw=0.45, r=0.003)
        if i >= 8:
            ax.plot(
                [xi + 0.006, xi + tw - 0.006],
                [y + 0.010, y + h - 0.010],
                transform=ax.transAxes,
                color="#777777",
                lw=0.55,
            )
    label(ax, x, y + h + 0.018, "0-2 source", fs=4.9, color=SOURCE, ha="left", va="bottom", weight="bold")
    label(ax, x + 0.275 * w, y + h + 0.018, "3-6 path", fs=4.9, color="#7A866F", ha="left", va="bottom", weight="bold")
    label(ax, x + 0.655 * w, y + h + 0.018, "7 receiver", fs=4.9, color=RECEIVER, ha="left", va="bottom", weight="bold")
    label(ax, x + 0.800 * w, y + h + 0.018, "8-10 zeroed", fs=4.9, color="#777777", ha="left", va="bottom", weight="bold")


def patch_strip(ax, x, y, w, h):
    labels = ["0", "1", "2", "...", "47", "48", "49"]
    gap = 0.006
    tw = (w - gap * 6) / 7
    for i, lab in enumerate(labels):
        box(
            ax,
            x + i * (tw + gap),
            y,
            tw,
            h,
            lab,
            PATCH_DARK if i in (0, 1, 2, 5, 6) else PATCH,
            edge="white",
            fs=5.2,
            weight="bold",
            lw=0.45,
            r=0.003,
        )


def draw_jst_block(ax, x, y, w, h):
    box(ax, x, y, w, h, "", "#FFFFFF", edge=EDGE, lw=0.95, r=0.012)
    label(ax, x + w / 2, y + h - 0.030, "JsTBlock x8", fs=7.2, color=BLACK, weight="bold", va="top")
    label(ax, x + w / 2, y + h - 0.058, "width 512 | 8 heads | MLP ratio 4", fs=4.8, color=GREY, va="top")

    sx = x + 0.105 * w
    sw = 0.790 * w
    sh = 0.102 * h
    gap = 0.018
    top = y + h - 0.155
    layers = [
        ("FiLM entry modulation", "#EEF3F6", RECEIVER),
        ("RMSNorm + MHA\n1-D RoPE on q,k", "#F9FAFA", BLUE_LIGHT),
        ("RMSNorm + SwiGLU\nfeed-forward", "#F9FAFA", "#B8C4AF"),
        ("residual gates", PANEL, EDGE),
    ]
    centers = []
    for i, (text, face, edge) in enumerate(layers):
        yy = top - i * (sh + gap)
        centers.append((sx + sw / 2, yy + sh / 2))
        box(ax, sx, yy, sw, sh, text, face, edge=edge, fs=5.25, lw=0.65, r=0.006)

    # Skip/gate hints inside the block.
    ax.plot(
        [sx - 0.020 * w, sx - 0.020 * w, sx + sw + 0.018 * w],
        [centers[1][1], centers[-1][1], centers[-1][1]],
        transform=ax.transAxes,
        color="#D5D5D5",
        lw=1.2,
        solid_capstyle="round",
    )
    arrow(ax, (sx - 0.020 * w, centers[1][1]), (sx + 0.020 * w, centers[1][1]), color="#CFCFCF", lw=0.8, ms=5)
    arrow(ax, (sx + 0.020 * w, centers[-1][1]), (sx + sw * 0.97, centers[-1][1]), color="#CFCFCF", lw=0.8, ms=5)

    # Side channels: condition FiLM and timestep adaLN-Zero.
    label(ax, x + 0.50 * w, y + 0.080 * h, "FiLM from condition groups | adaLN-Zero from t", fs=4.35, color=GREY)

    # Attention-like marks are schematic, not measured attention weights.
    rng = np.random.default_rng(7)
    for _ in range(8):
        cx = x + rng.uniform(0.18, 0.84) * w
        cy = y + rng.uniform(0.24, 0.78) * h
        rr = rng.uniform(0.004, 0.009)
        ax.add_patch(plt.Circle((cx, cy), rr, transform=ax.transAxes, facecolor=BLUE_LIGHT, edgecolor="none", alpha=0.14))


def draw(clean: bool = False):
    apply_nature_style(base_size=6.4)
    fig = plt.figure(figsize=(9.2, 4.9), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    if not clean:
        label(
            ax,
            0.035,
            0.955,
            "Just seismic Transformer (JsT) architecture",
            fs=9.0,
            color=BLACK,
            ha="left",
            va="top",
            weight="bold",
        )
        label(
            ax,
            0.035,
            0.912,
            "Block schematic traced from the condition encoder, denoiser and JsT model code",
            fs=5.6,
            color=GREY,
            ha="left",
            va="top",
        )

    # Condition branch.
    row_y = 0.740
    box(ax, 0.040, row_y, 0.120, 0.080, "seismic\nmetadata", WARM, edge="#D8D4CC", fs=5.4)
    box(ax, 0.200, row_y, 0.140, 0.080, "SeismicCondition\nEncoder v3", PANEL, edge=EDGE, fs=5.25)
    arrow(ax, (0.165, row_y + 0.040), (0.195, row_y + 0.040))
    arrow(ax, (0.345, row_y + 0.040), (0.375, row_y + 0.040))
    token_strip(ax, 0.380, row_y + 0.010, 0.340, 0.055)
    label(ax, 0.550, row_y - 0.016, "11 slots; tokens 8-10 zeroed in the measurement encoder", fs=4.55)

    # Flow-state / patch branch.
    row_y = 0.535
    box(ax, 0.040, row_y, 0.120, 0.084, "flow state z(t)\nor initial noise", WARM, edge="#D8D4CC", fs=5.25)
    box(ax, 0.200, row_y, 0.140, 0.084, "PatchEmbed1D\nConv1d", "#EEF3F6", edge=BLUE_LIGHT, fs=5.25)
    arrow(ax, (0.165, row_y + 0.042), (0.195, row_y + 0.042))
    arrow(ax, (0.345, row_y + 0.042), (0.375, row_y + 0.042))
    patch_strip(ax, 0.380, row_y + 0.021, 0.195, 0.047)
    label(ax, 0.478, row_y - 0.010, "3 x 3200 samples -> 50 patches; patch size 64", fs=4.75)

    # Timestep branch.
    row_y = 0.345
    box(ax, 0.040, row_y, 0.120, 0.070, "timestep t", WARM, edge="#D8D4CC", fs=5.45)
    box(ax, 0.200, row_y, 0.140, 0.070, "sinusoidal t\nMLP embed", PANEL, edge=EDGE, fs=5.10)
    arrow(ax, (0.165, row_y + 0.035), (0.195, row_y + 0.035))

    # Main Transformer core.
    draw_jst_block(ax, 0.625, 0.255, 0.225, 0.455)
    arrow(ax, (0.720, 0.770), (0.625, 0.632), color="#909090", lw=0.85, ms=7, rad=0.04)
    arrow(ax, (0.575, 0.558), (0.625, 0.525), color="#909090", lw=0.85, ms=7, rad=-0.02)
    arrow(ax, (0.340, 0.380), (0.625, 0.425), color=BLUE_LIGHT, lw=0.90, ms=7, rad=-0.08)

    # Output branch.
    arrow(ax, (0.852, 0.482), (0.880, 0.482), color=BLUE, lw=0.95, ms=8)
    box(ax, 0.885, 0.422, 0.080, 0.120, "FinalLayer\nRMSNorm\nlinear\nunpatchify", "#EEF3F6", edge=BLUE_LIGHT, fs=4.85)
    arrow(ax, (0.925, 0.419), (0.925, 0.335), color=BLUE, lw=0.95, ms=8)
    box(ax, 0.885, 0.218, 0.080, 0.092, "", "#FFFFFF", edge="#D0D0D0", fs=5.0)
    mini_wave(ax, 0.897, 0.243, 0.056, 0.050)
    label(ax, 0.925, 0.190, "predicted\nclean waveform", fs=5.0, color=BLUE, va="top")

    # Flow wrapper. It is intentionally separate from the network blocks.
    box(ax, 0.040, 0.075, 0.430, 0.140, "", "#FFFFFF", edge=EDGE, fs=5.2)
    label(ax, 0.060, 0.185, "Flow-matching denoiser wrapper", fs=6.1, color=BLACK, ha="left", va="top", weight="bold")
    label(ax, 0.060, 0.145, "training", fs=5.05, color=GREY, ha="left", va="top")
    label(ax, 0.220, 0.145, r"$z=t\,x+(1-t)\epsilon$", fs=5.8, color=BLACK, ha="left", va="top")
    label(ax, 0.060, 0.102, "generation", fs=5.05, color=GREY, ha="left", va="top")
    label(ax, 0.220, 0.102, "Heun ODE integration, 50 steps", fs=5.6, color=BLACK, ha="left", va="top")

    # Compact code-derived constants.
    box(ax, 0.505, 0.075, 0.285, 0.140, "", "#FFFFFF", edge=EDGE, fs=5.0)
    label(ax, 0.525, 0.185, "Run-time dimensions", fs=6.0, color=BLACK, ha="left", va="top", weight="bold")
    label(ax, 0.525, 0.145, "input/output: 3 components x 3200 samples at 40 Hz", fs=5.0, color=GREY, ha="left", va="top")
    label(ax, 0.525, 0.130, "network slots: 11 condition tokens + 50 waveform patches", fs=4.9, color=GREY, ha="left", va="top")
    label(ax, 0.525, 0.101, "condition slots 8-10 are zeroed for the measurement encoder", fs=4.9, color=GREY, ha="left", va="top")
    label(ax, 0.525, 0.072, "observed waveform is used after generation to form the residual", fs=4.9, color=GREY, ha="left", va="bottom")

    return fig


def save(fig, output_base: Path):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def main() -> int:
    fig = draw(clean=False)
    save(fig, OUT_BASE)
    plt.close(fig)
    fig = draw(clean=True)
    save(fig, OUT_CLEAN_BASE)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
