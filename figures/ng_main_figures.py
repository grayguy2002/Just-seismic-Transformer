"""Render Nature Geoscience-style main figures from existing JsT caches.

This script intentionally separates figure rendering from GPU inference.  It
reuses the cache/results files produced by the current Fig. 1--3 pipeline and
writes one Methods figure plus three Results figures:

  methods_ng -- selected condition encoder and residual measurement
  fig1_ng    -- residual site signature
  fig2_ng    -- measured-profile validation hierarchy
  fig3_ng    -- receiver-token geological manifold

The cross-method complementarity panel remains available as an optional
supplement-style render, but it is not part of the main NG figure set.

Run from the project root:
  python3 manuscript/figures/ng_main_figures.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.ndimage import uniform_filter1d
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nature_geo_style import (  # noqa: E402
    DOUBLE_COL,
    SEMANTIC,
    GEO_COLORS,
    add_cartopy_base,
    add_panel_label,
    apply_nature_style,
    save_panel,
)


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "ng_main_figures"

FIG1_CACHE = ROOT / "outputs" / "fig1_cache"
FIG2_CACHE = ROOT / "outputs" / "fig2_cache"
FIG3_CACHE = ROOT / "outputs" / "fig3_cache"
VS30_CSV = ROOT / "outputs" / "vs30_validation" / "jst_hvsr_vs_vs30_results.csv"
VS30_STATS = ROOT / "outputs" / "vs30_validation" / "correlation_results.json"
EXPA = ROOT / "outputs" / "expA_vs30_controls" / "results.json"
SINGLE_MULTI = ROOT / "outputs" / "single_vs_multi_event" / "results.json"
EXPB_CROSS = ROOT / "outputs" / "expB_cross_method" / "results.json"
EXPB_FOLLOWUP = ROOT / "outputs" / "expB_followup" / "results.json"
EXPC = ROOT / "outputs" / "expC_token7_disentangle" / "results.json"
KIKNET_EXP_F = ROOT / "outputs" / "expF_kiknet_jst_inference" / "per_station_all_events.csv"
KIKNET_EXP_F_QC_MIN2 = ROOT / "outputs" / "expF_kiknet_jst_inference" / "per_station_qc_arrival_window_min2.csv"
KIKNET_RESULTS = ROOT / "outputs" / "expF_kiknet_jst_inference" / "results.json"
KIKNET_STATIONS = ROOT / "data" / "kiknet_measured_vs30_pwave_v1" / "kiknet_measured_vs30_station_manifest.csv"
KIKNET_EXP_H_DIR = ROOT / "outputs" / "expH_kiknet_dense_arrival_qc_events"
KIKNET_EXP_H_POST = ROOT / "outputs" / "expH_kiknet_dense_arrival_qc_events_postprocess"
KIKNET_EXP_H_RECORDS = KIKNET_EXP_H_DIR / "per_station_event_records.csv"
KIKNET_EXP_H_QC_MIN2 = KIKNET_EXP_H_DIR / "per_station_qc_arrival_window_min2.csv"
KIKNET_EXP_H_RESULTS = KIKNET_EXP_H_DIR / "results.json"
KIKNET_EXP_H_FREQ = KIKNET_EXP_H_POST / "jst_frequency_correlations.csv"
KIKNET_EXP_H_STD_FREQ = KIKNET_EXP_H_POST / "standard_hvsr_frequency_correlations.csv"
KIKNET_EXP_I = ROOT / "outputs" / "expI_kiknet_validation_controls" / "results.json"

FREQ_EDGES = np.logspace(np.log10(0.3), np.log10(15.0), 41)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
KIKNET_HVSR_BINS = [f"hvsr_bin_{i:02d}" for i in range(40)]
KIKNET_BANDS = {
    "0.5-5Hz": [i for i, f in enumerate(FREQ_CENTERS) if 0.5 <= f < 5.0],
    "1-3Hz": [i for i, f in enumerate(FREQ_CENTERS) if 1.0 <= f < 3.0],
    "1-10Hz": [i for i, f in enumerate(FREQ_CENTERS) if 1.0 <= f < 10.0],
    "3-10Hz": [i for i, f in enumerate(FREQ_CENTERS) if 3.0 <= f < 10.0],
}

AMP_CMAP = LinearSegmentedColormap.from_list(
    "jst_amp",
    ["#EEF2F4", "#B9CDDC", SEMANTIC["jsT_blue_light"], SEMANTIC["jsT_blue"]],
)
FINGERPRINT_CMAP = LinearSegmentedColormap.from_list(
    "jst_fingerprint",
    ["#F8F8F5", "#DCE8F0", "#9DBAD0", SEMANTIC["jsT_blue"], "#0B335A"],
)
FINGERPRINT_CMAP.set_bad("#FFFFFF")
FINGERPRINT_ANOM_CMAP = LinearSegmentedColormap.from_list(
    "jst_fingerprint_anomaly",
    ["#9A9A9A", "#D9D9D6", "#FAFAF7", "#9DBAD0", SEMANTIC["jsT_blue"], "#0B335A"],
)
FINGERPRINT_ANOM_CMAP.set_bad("#FFFFFF")

SOURCE_COLOR = "#8CADC8"
PATH_COLOR = "#C5CFC0"
RECEIVER_COLOR = SEMANTIC["token_green_light"]

TOKEN_ROWS = [
    (0, "source_size", "source", SOURCE_COLOR),
    (1, "source_location_depth", "source", SOURCE_COLOR),
    (2, "source_radiation_proxy", "source", SOURCE_COLOR),
    (3, "path_geometry", "path", PATH_COLOR),
    (4, "path_travel_time", "path", PATH_COLOR),
    (5, "selected_phase_label", "path", PATH_COLOR),
    (6, "path_region_proxy", "path", PATH_COLOR),
    (7, "receiver_site", "receiver", RECEIVER_COLOR),
]

ABLATION_ROWS = [
    ("remove token 8", -0.036, "station identity"),
    ("remove token 9", -0.039, "instrument"),
    ("remove token 10", -0.037, "receiver orientation"),
    ("remove 8+9+10", -0.004, "selected encoder"),
]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def jitter(n: int, width: float = 0.06, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, size=n)


def station_network(station_id: str) -> str:
    return station_id.split(".")[0]


def load_kiknet_station_results(path: Path = KIKNET_EXP_F) -> tuple[pd.DataFrame, dict]:
    """Return KiK-net Exp F station results joined to profile-derived station metadata."""
    result = pd.read_csv(path)
    manifest = pd.read_csv(KIKNET_STATIONS)
    result["station_code"] = result["station_id"].str.split(".").str[-1]
    keep = [
        "station_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "vs30_m_s",
        "nehrp_site_class",
    ]
    merged = result.merge(manifest[keep], on="station_code", how="left", validate="many_to_one")
    missing = merged["station_latitude_deg"].isna().sum()
    if missing:
        raise ValueError(f"Missing KiK-net station coordinates for {missing} stations")
    return merged, load_json(KIKNET_RESULTS)


def load_kiknet_dense_station_results(path: Path = KIKNET_EXP_H_QC_MIN2) -> tuple[pd.DataFrame, dict]:
    """Return dense KiK-net Exp H station results with profile-derived Vs30 metadata."""
    result = pd.read_csv(path)
    result = result.rename(columns={"vs30": "vs30_m_s", "nehrp": "nehrp_site_class"})
    return result, load_json(KIKNET_EXP_H_RESULTS)


def load_kiknet_dense_records() -> pd.DataFrame:
    """Return dense KiK-net Exp H station-event records with QC flags and frequency bands."""
    df = pd.read_csv(KIKNET_EXP_H_RECORDS)
    df["arrival_qc"] = (
        (df["arrival_sample"] >= 0)
        & (df["arrival_sample"] < df["trace_n_samples"])
        & (df["left_pad"] <= 800)
        & (df["right_pad"] <= 0)
    )
    df["full_pre_qc"] = (
        (df["arrival_sample"] >= 800)
        & (df["arrival_sample"] < df["trace_n_samples"])
        & (df["left_pad"] == 0)
        & (df["right_pad"] <= 0)
    )
    for name, idx in KIKNET_BANDS.items():
        df[name] = df[[KIKNET_HVSR_BINS[i] for i in idx]].mean(axis=1)
    return df


def metric_row(path: Path, scope: str, metric: str, event_id: str = "pooled") -> dict:
    rows = pd.read_csv(path)
    row = rows[(rows["scope"] == scope) & (rows["metric"] == metric) & (rows["event_id"].astype(str) == str(event_id))]
    if row.empty:
        raise ValueError(f"Missing metric row: {path} {scope=} {metric=} {event_id=}")
    return row.iloc[0].to_dict()


def bootstrap_one_record_rhos(df: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 123) -> np.ndarray:
    """Sample one record per station and compute Spearman rho to profile Vs30."""
    sub = df[["station_id", "vs30", metric]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in sub.groupby("station_id", sort=True)]
    values = sub[metric].to_numpy(float)
    vs30 = sub["vs30"].to_numpy(float)
    rhos = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = np.array([rng.choice(g) for g in groups], dtype=int)
        rhos[i] = spearmanr(values[idx], vs30[idx]).statistic
    return rhos


def select_quantile_stations(values: pd.Series | dict, n: int = 16) -> list[str]:
    """Select stations evenly across the station-mean residual-amplification range."""
    if isinstance(values, dict):
        series = pd.Series(values, dtype=float)
    else:
        series = pd.Series(values, dtype=float).dropna()
    series = series.sort_values(kind="mergesort")
    if len(series) <= n:
        return list(series.index)
    positions = np.linspace(0, len(series) - 1, n).round().astype(int)
    positions = np.unique(positions)
    while len(positions) < n:
        missing = [i for i in range(len(series)) if i not in set(positions)]
        positions = np.sort(np.r_[positions, missing[len(missing) // 2]])
    return list(series.index[positions[:n]])


def fingerprint_matrix_from_groups(
    grouped_curves: dict[str, np.ndarray],
    selected: list[str],
    max_events: int,
) -> tuple[np.ndarray, list[tuple[str, int, int]], np.ndarray]:
    """Build a station-event heatmap matrix with blank separator rows between stations."""
    rows: list[np.ndarray] = []
    spans: list[tuple[str, int, int]] = []
    station_means: list[float] = []
    n_freq = len(FREQ_CENTERS)
    blank = np.full(n_freq, np.nan)
    for station_id in selected:
        curves = np.asarray(grouped_curves[station_id], dtype=float)
        curves = curves[np.isfinite(curves).all(axis=1)]
        if curves.size == 0:
            continue
        mean_curve = np.nanmean(curves, axis=0)
        order = np.argsort([pearsonr(row, mean_curve)[0] if np.nanstd(row) > 0 else -np.inf for row in curves])[::-1]
        use = curves[order[:max_events]]
        start = len(rows)
        rows.extend(use)
        if len(use) < max_events:
            rows.extend([blank.copy() for _ in range(max_events - len(use))])
        end = len(rows) - 1
        spans.append((station_id, start, end))
        station_means.append(float(np.nanmean(curves)))
        rows.append(blank.copy())
    if rows:
        rows = rows[:-1]
    return np.vstack(rows), spans, np.asarray(station_means, dtype=float)


def robust_frequency_anomaly(matrix: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Remove the frequency-wise common trend and return robust z-like anomalies."""
    matrix = np.asarray(matrix, dtype=float)
    ref = matrix if reference is None else np.asarray(reference, dtype=float)
    center = np.nanmedian(ref, axis=0)
    mad = np.nanmedian(np.abs(ref - center), axis=0)
    scale = 1.4826 * mad
    fallback = np.nanstd(ref, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    out = (matrix - center) / scale
    out[~np.isfinite(matrix)] = np.nan
    return out


def row_center_anomaly(matrix: np.ndarray) -> np.ndarray:
    """Remove record-level offsets so heatmaps emphasise spectral shape."""
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix.copy()
    valid_rows = np.isfinite(matrix).any(axis=1)
    centered[valid_rows] = matrix[valid_rows] - np.nanmedian(matrix[valid_rows], axis=1, keepdims=True)
    centered[~np.isfinite(matrix)] = np.nan
    return centered


def group_for_complementarity(station_id: str) -> str:
    net = station_network(station_id)
    if net in {"OK", "GS", "NM"}:
        return "sedimentary"
    if net in {"AK", "HV", "AV", "AT"}:
        return "complex"
    return "other"


def save_stats(stats: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "figure_stats.json"
    old = {}
    if path.exists():
        old = load_json(path)
    old.update(stats)
    with path.open("w") as f:
        json.dump(old, f, indent=2)


def _axis_box(ax, x, y, w, h, text, face, edge="#FFFFFF", lw=0.5, fontsize=5.8, weight="normal"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=SEMANTIC["black"],
        fontweight=weight,
    )
    return patch


def _axis_arrow(ax, start, end, color=None, lw=0.65, rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=lw,
        color=color or SEMANTIC["axis_grey"],
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    return arrow


def _axis_wave(ax, x0, y0, w, h, color, lw=0.9, phase=0.0, amp=0.32, noise=0.0):
    xs = np.linspace(0, 1, 160)
    y = (
        0.5
        + amp * np.sin(2 * np.pi * (2.6 * xs + phase)) * np.exp(-1.4 * xs)
        + 0.10 * np.sin(2 * np.pi * (8.4 * xs + 0.2 + phase)) * np.exp(-0.7 * xs)
    )
    if noise:
        y += noise * np.sin(2 * np.pi * (17 * xs + phase))
    ax.plot(x0 + xs * w, y0 + y * h, transform=ax.transAxes, color=color, lw=lw, clip_on=False)


def draw_methods() -> dict:
    """Methods figure: selected condition encoder and residual measurement."""
    apply_nature_style(6.2)
    fig = plt.figure(figsize=(DOUBLE_COL, 132 / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.26, 0.88],
        width_ratios=[1.18, 1.0],
        left=0.045,
        right=0.985,
        top=0.965,
        bottom=0.075,
        hspace=0.33,
        wspace=0.28,
    )

    # a: model-to-measurement schematic.
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    add_panel_label(ax, "a", x=-0.025, y=1.02)

    _axis_box(
        ax,
        0.018,
        0.55,
        0.128,
        0.24,
        "station-event\nmetadata",
        "#F2F1EC",
        edge="#D8D4CC",
        fontsize=6.1,
    )
    ax.text(
        0.018,
        0.43,
        "source, path, receiver\ncoordinates and phase labels",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.1,
        color=SEMANTIC["axis_grey"],
    )

    token_x0 = 0.205
    token_w = 0.044
    token_gap = 0.008
    token_y = 0.55
    token_h = 0.23
    for i, (idx, name, group, color) in enumerate(TOKEN_ROWS):
        x = token_x0 + i * (token_w + token_gap)
        _axis_box(ax, x, token_y, token_w, token_h, f"{idx}", color, edge="white", fontsize=7.0, weight="bold")
        ax.text(
            x + token_w / 2,
            token_y - 0.045,
            name.replace("_", "\n"),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=4.35,
            color=SEMANTIC["axis_grey"],
            linespacing=0.90,
        )

    group_specs = [
        ("source x3", 0, 2, SOURCE_COLOR),
        ("path x4", 3, 6, PATH_COLOR),
        ("receiver x1", 7, 7, RECEIVER_COLOR),
    ]
    for label, start, end, color in group_specs:
        x0 = token_x0 + start * (token_w + token_gap)
        x1 = token_x0 + end * (token_w + token_gap) + token_w
        ax.plot([x0, x1], [0.845, 0.845], transform=ax.transAxes, color=color, lw=1.2)
        ax.text(
            (x0 + x1) / 2,
            0.872,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5.7,
            color=SEMANTIC["black"],
        )

    _axis_arrow(ax, (0.151, 0.67), (0.197, 0.67))

    _axis_box(
        ax,
        0.665,
        0.55,
        0.105,
        0.24,
        "flow-matching\ntransformer",
        "#F6F6F6",
        edge="#D0D0D0",
        fontsize=5.8,
    )
    _axis_arrow(ax, (0.626, 0.67), (0.657, 0.67))

    # Waveform measurement block.
    _axis_box(ax, 0.830, 0.70, 0.122, 0.145, "", "#FFFFFF", edge="#D0D0D0", fontsize=5.6)
    ax.text(0.846, 0.815, "observed", transform=ax.transAxes, fontsize=5.5, ha="left", va="center", color=SEMANTIC["black"])
    _axis_wave(ax, 0.850, 0.715, 0.088, 0.062, SEMANTIC["black"], lw=0.78, phase=0.02, noise=0.035)
    _axis_box(ax, 0.830, 0.49, 0.122, 0.145, "", "#FFFFFF", edge="#D0D0D0", fontsize=5.6)
    ax.text(0.846, 0.605, "predicted", transform=ax.transAxes, fontsize=5.5, ha="left", va="center", color=SEMANTIC["jsT_blue"])
    _axis_wave(ax, 0.850, 0.505, 0.088, 0.062, SEMANTIC["jsT_blue"], lw=0.78, phase=0.09)
    _axis_arrow(ax, (0.772, 0.67), (0.825, 0.57), color=SEMANTIC["jsT_blue"], rad=-0.10)
    ax.text(0.965, 0.663, "-", transform=ax.transAxes, fontsize=10, ha="center", va="center", color=SEMANTIC["axis_grey"])

    _axis_box(
        ax,
        0.823,
        0.155,
        0.142,
        0.165,
        "",
        "#EEF3F6",
        edge="#C7D8E4",
        fontsize=5.7,
        weight="bold",
    )
    ax.text(0.894, 0.284, "residual", transform=ax.transAxes, fontsize=5.8, ha="center", va="center", color=SEMANTIC["black"], fontweight="bold")
    ax.text(0.894, 0.255, "observed - predicted", transform=ax.transAxes, fontsize=5.2, ha="center", va="center", color=SEMANTIC["black"])
    _axis_wave(ax, 0.842, 0.166, 0.104, 0.052, SEMANTIC["jsT_blue"], lw=0.85, phase=0.22, noise=0.025)
    _axis_arrow(ax, (0.890, 0.485), (0.890, 0.325), color=SEMANTIC["axis_grey"])
    ax.text(
        0.745,
        0.215,
        "site-effect observable\ncomputed as JsT-HVSR",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=5.6,
        color=SEMANTIC["black"],
    )
    _axis_arrow(ax, (0.750, 0.215), (0.818, 0.215), color=SEMANTIC["jsT_blue"])

    ax.text(
        0.205,
        0.205,
        "The selected encoder removes station identity,\n"
        "instrument and receiver-orientation tokens while\n"
        "retaining source, path and receiver terms.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color=SEMANTIC["axis_grey"],
    )

    # b: token ablation support.
    ax = fig.add_subplot(gs[1, 0])
    labels = [r[0] for r in ABLATION_ROWS]
    vals = np.array([r[1] for r in ABLATION_ROWS], dtype=float)
    y = np.arange(len(labels))
    cols = [SEMANTIC["bg_grey"], SEMANTIC["bg_grey"], SEMANTIC["bg_grey"], SEMANTIC["jsT_blue"]]
    ax.barh(y, vals, color=cols, edgecolor="white", lw=0.25, height=0.58)
    ax.axvline(-0.05, color=SEMANTIC["null_grey"], ls="--", lw=0.75)
    ax.axvline(0, color=SEMANTIC["black"], lw=0.45, alpha=0.55)
    for yi, val, (_, _, note) in zip(y, vals, ABLATION_ROWS):
        ax.text(val - 0.003, yi, f"{val:+.3f}", ha="right", va="center", fontsize=5.4, color=SEMANTIC["black"])
        ax.text(0.004, yi, note, ha="left", va="center", fontsize=5.2, color=SEMANTIC["axis_grey"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.064, 0.030)
    ax.set_xlabel("gen_peak ratio delta at 600 epochs")
    ax.set_title("Ablation identifies expendable receiver-side tokens", loc="left", fontsize=6.4, pad=2.0)
    ax.text(-0.050, len(labels) - 0.10, "decision threshold", rotation=90, fontsize=5.0, color=SEMANTIC["axis_grey"], va="top", ha="right")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.10, lw=0.2)
    add_panel_label(ax, "b", x=-0.09, y=1.03)

    # c: compact model comparison as graphical evidence; details are in the manuscript table.
    ax = fig.add_subplot(gs[1, 1])
    add_panel_label(ax, "c", x=-0.08, y=1.03)
    ax.set_title("The selected encoder preserves scale with fewer tokens", loc="left", fontsize=6.4, pad=2.0)

    metrics = [
        ("gen_peak\nratio", 0.898, 0.876, "-2.4 pp", SEMANTIC["axis_grey"]),
        ("gen_peak\nstd", 0.389, 0.643, "+65%", SEMANTIC["token_green"]),
        ("encoder\nparams (M)", 7.2, 5.3, "-26%", SEMANTIC["jsT_blue"]),
    ]
    y = np.arange(len(metrics))
    old = np.array([m[1] for m in metrics], dtype=float)
    new = np.array([m[2] for m in metrics], dtype=float)
    scaled_old = old / np.maximum(old, new)
    scaled_new = new / np.maximum(old, new)
    height = 0.28
    ax.barh(y + height / 1.8, scaled_old, height=height, color=SEMANTIC["bg_grey"], edgecolor="white", lw=0.25)
    ax.barh(y - height / 1.8, scaled_new, height=height, color=SEMANTIC["jsT_blue"], edgecolor="white", lw=0.25)
    for yi, (label, old_val, new_val, change, color), so, sn in zip(y, metrics, scaled_old, scaled_new):
        ax.text(1.04, yi, change, ha="left", va="center", fontsize=5.7, color=color, fontweight="bold")
        ax.text(so - 0.015, yi + height / 1.8, f"{old_val:g}", ha="right", va="center", fontsize=4.9, color=SEMANTIC["std_grey"])
        ax.text(sn - 0.015, yi - height / 1.8, f"{new_val:g}", ha="right", va="center", fontsize=4.9, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in metrics])
    ax.set_xlim(0, 1.23)
    ax.set_xlabel("Metric scaled to row maximum")
    ax.text(
        0.995,
        1.015,
        "blue: 8-token encoder; grey: 11-token candidate",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=SEMANTIC["axis_grey"],
    )
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.10, lw=0.2)
    ax.text(
        0.01,
        -0.28,
        "The 8-token no-patch condition encoder is used for all Results figures.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color=SEMANTIC["axis_grey"],
    )

    save_panel(fig, FIG_DIR / "methods_ng")
    plt.close(fig)
    return {
        "methods": {
            "n_tokens_selected_encoder": 8,
            "ablations": {label: delta for label, delta, _ in ABLATION_ROWS},
            "candidate_11token_gen_peak_ratio": 0.898,
            "selected_encoder_gen_peak_ratio": 0.876,
            "candidate_11token_gp_std": 0.389,
            "selected_encoder_gp_std": 0.643,
            "candidate_11token_encoder_params_m": 7.2,
            "selected_encoder_params_m": 5.3,
        }
    }


def draw_fig1() -> dict:
    """Fig. 1: JsT residuals behave as reproducible site signatures."""
    hvsr = load_json(FIG1_CACHE / "station_hvsr.json")
    stats = load_json(FIG1_CACHE / "stats.json")

    station_ids = list(hvsr.keys())
    amps = np.array([float(np.mean(hvsr[s]["curves"])) for s in station_ids])
    lons = np.array([float(hvsr[s]["lon"]) for s in station_ids])
    lats = np.array([float(hvsr[s]["lat"]) for s in station_ids])
    events = np.array([int(hvsr[s]["n_events"]) for s in station_ids])
    sizes = 7.0 + 2.8 * np.sqrt(events)
    kiknet, kiknet_results = load_kiknet_dense_station_results()
    jp_amp = kiknet["mean_amp"].to_numpy(float)
    color_vmin = float(np.nanpercentile(np.r_[amps, jp_amp], 5))
    color_vmax = float(np.nanpercentile(np.r_[amps, jp_amp], 95))
    validation_controls = load_json(KIKNET_EXP_I)
    j_station_min2 = validation_controls["spatial_block_controls"]["station_mean_min2"]
    kiknet_records = load_kiknet_dense_records()

    apply_nature_style(6.2)
    fig = plt.figure(figsize=(DOUBLE_COL, 176 / 25.4))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.16, 0.70, 0.96],
        width_ratios=[1.48, 1.0],
        left=0.055,
        right=0.985,
        top=0.970,
        bottom=0.065,
        hspace=0.42,
        wspace=0.25,
    )

    # a: spatial residual-amplification landscape.
    try:
        import cartopy.crs as ccrs

        map_gs = gs[0, :].subgridspec(1, 2, width_ratios=[3.10, 1.25], wspace=0.07)
        ax_map = fig.add_subplot(map_gs[0, 0], projection=ccrs.PlateCarree())
        ax_jp = fig.add_subplot(map_gs[0, 1], projection=ccrs.PlateCarree())
        add_cartopy_base(ax_map, [-172, -64, 24, 72], land_scale="110m")
        sc = ax_map.scatter(
            lons,
            lats,
            c=amps,
            s=sizes,
            cmap=AMP_CMAP,
            vmin=color_vmin,
            vmax=color_vmax,
            alpha=0.74,
            edgecolors="white",
            lw=0.32,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        add_cartopy_base(ax_jp, [128.5, 146.5, 30.2, 45.9], land_scale="50m")
        jp_sizes = 6.5 + 2.4 * np.sqrt(kiknet["n_events"].to_numpy(float))
        ax_jp.scatter(
            kiknet["station_longitude_deg"].to_numpy(float),
            kiknet["station_latitude_deg"].to_numpy(float),
            c=jp_amp,
            s=jp_sizes,
            cmap=AMP_CMAP,
            vmin=color_vmin,
            vmax=color_vmax,
            alpha=0.76,
            edgecolors="white",
            lw=0.30,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        ax_map.set_title("MLAAPDE testing stations", loc="left", pad=2, fontsize=6.0, color=SEMANTIC["axis_grey"])
        ax_jp.set_title("Japan KiK-net profile Vs30", loc="left", pad=2, fontsize=6.0, color=SEMANTIC["jsT_blue"])
        ax_jp.text(
            0.03,
            0.035,
            f"QC $\\geq$2 records, N = {j_station_min2['N']}\n"
            f"$\\rho$ = {j_station_min2['rho']:+.2f}, p = {j_station_min2['p']:.1e}",
            transform=ax_jp.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.25,
            color=SEMANTIC["black"],
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=1.6),
            zorder=7,
        )
        cb = fig.colorbar(sc, ax=[ax_map, ax_jp], orientation="horizontal", fraction=0.055, pad=0.085, aspect=46)
        cb.set_label("Mean JsT-HVSR amplification", fontsize=5.8)
        cb.ax.tick_params(labelsize=5.2, length=2)
        ax_map.text(
            0.01,
            0.03,
            f"{len(station_ids)} test stations in analysis; map view shows training-domain coverage",
            transform=ax_map.transAxes,
            fontsize=5.7,
            color=SEMANTIC["axis_grey"],
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.8),
        )
        add_panel_label(ax_map, "a", x=-0.03, y=1.02)
    except Exception:
        ax_map = fig.add_subplot(gs[0, :])
        sc = ax_map.scatter(lons, lats, c=amps, s=sizes, cmap=AMP_CMAP, edgecolors="white", lw=0.25)
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        fig.colorbar(sc, ax=ax_map, orientation="horizontal", fraction=0.05, pad=0.12)
        add_panel_label(ax_map, "a")

    # b: station-ranked residual landscape with standard-HVSR benchmark reference.
    us_color = SEMANTIC["jsT_blue"]
    rank_items = []
    for sid in station_ids:
        curves = np.asarray(hvsr[sid]["curves"], dtype=float)
        station_mean = float(np.nanmean(curves))
        event_vals = np.nanmean(curves, axis=1)
        rank_items.append((sid, station_mean, event_vals, hvsr[sid].get("geology", ""), hvsr[sid].get("net", "")))
    rank_items.sort(key=lambda item: item[1])
    rank_ids = [item[0] for item in rank_items]
    rank_means = np.array([item[1] for item in rank_items], dtype=float)
    x_rank = np.arange(len(rank_ids))

    svm = load_json(SINGLE_MULTI)
    svm_df = pd.DataFrame(svm["stations"]).set_index("station_id")
    hv_ref = pd.Series(index=rank_ids, dtype=float)
    cross_ref = pd.Series(index=rank_ids, dtype=float)
    for sid in rank_ids:
        if sid in svm_df.index:
            hv_ref.loc[sid] = float(svm_df.loc[sid, "hv_single_to_multi_mean"])
            cross_ref.loc[sid] = float(svm_df.loc[sid, "jst_multi_vs_hv_multi"])
    hv_x = np.array([i for i, sid in enumerate(rank_ids) if np.isfinite(hv_ref.loc[sid])], dtype=float)
    hv_y = np.array([hv_ref.loc[sid] for sid in rank_ids if np.isfinite(hv_ref.loc[sid])], dtype=float)
    cross_y = np.array([cross_ref.loc[sid] for sid in rank_ids if np.isfinite(cross_ref.loc[sid])], dtype=float)
    hv_rank_rho = float(spearmanr(hv_x, hv_y).statistic) if len(hv_x) > 2 else np.nan
    cross_rank_rho = float(spearmanr(hv_x, cross_y).statistic) if len(hv_x) > 2 else np.nan
    cross_mean = float(np.nanmean(cross_y)) if len(cross_y) else np.nan

    rank_gs = gs[1, :].subgridspec(2, 1, height_ratios=[0.76, 0.24], hspace=0.06)
    ax_rank = fig.add_subplot(rank_gs[0, 0])
    ax_ref = fig.add_subplot(rank_gs[1, 0], sharex=ax_rank)

    bar_cols = np.full(len(rank_ids), SEMANTIC["jsT_blue_light"], dtype=object)
    bar_alpha = np.full(len(rank_ids), 0.52, dtype=float)
    highlight = {
        "AK.BWN": ("low-amplification", SEMANTIC["token_green"]),
        "AK.RC01": ("subduction", SEMANTIC["std_grey"]),
        "OK.CROK": ("sedimentary basin", SEMANTIC["jsT_blue"]),
    }
    for sid, (_, col) in highlight.items():
        if sid in rank_ids:
            idx = rank_ids.index(sid)
            bar_cols[idx] = col
            bar_alpha[idx] = 0.95

    for i, (mean_val, col, alpha) in enumerate(zip(rank_means, bar_cols, bar_alpha)):
        ax_rank.vlines(i, 0, mean_val, color=col, alpha=alpha, lw=0.95, zorder=2)

    rng = np.random.default_rng(24)
    for i, (_, mean_val, event_vals, _, _) in enumerate(rank_items):
        jitter_x = i + rng.uniform(-0.20, 0.20, size=len(event_vals))
        ax_rank.scatter(
            jitter_x,
            event_vals,
            s=3.8,
            color=SEMANTIC["black"],
            alpha=0.16,
            linewidths=0,
            zorder=3,
        )

    for sid, (label, col) in highlight.items():
        if sid not in rank_ids:
            continue
        idx = rank_ids.index(sid)
        y = rank_means[idx]
        ax_rank.scatter(idx, y, s=18, marker="o", color=col, edgecolors="white", lw=0.35, zorder=5)
        dy = 0.075 if sid != "AK.BWN" else 0.055
        ax_rank.annotate(
            sid,
            xy=(idx, y),
            xytext=(idx + (6 if sid != "OK.CROK" else -10), y + dy),
            ha="left" if sid != "OK.CROK" else "right",
            va="bottom",
            fontsize=5.35,
            color=col,
            arrowprops=dict(arrowstyle="-", lw=0.45, color=col, shrinkA=1, shrinkB=2),
            zorder=6,
        )

    ax_rank.set_title("Station-ranked residual amplification across the testing domain", loc="left", fontsize=6.5, pad=2.0)
    ax_rank.set_ylabel("Mean JsT-HVSR\namplification")
    ax_rank.set_xlim(-1, len(rank_ids))
    ax_rank.set_ylim(0, max(0.92, float(np.nanmax(rank_means)) + 0.08))
    ax_rank.set_xticks([])
    ax_rank.grid(axis="y", alpha=0.12, lw=0.25)
    ax_rank.text(
        0.01,
        0.95,
        "150 stations ordered by station mean; pale dots are 8 earthquake records per station",
        transform=ax_rank.transAxes,
        ha="left",
        va="top",
        fontsize=5.35,
        color=SEMANTIC["axis_grey"],
    )
    ax_rank.text(0.00, -0.08, "low", transform=ax_rank.transAxes, ha="left", va="top", fontsize=5.2, color=SEMANTIC["axis_grey"])
    ax_rank.text(1.00, -0.08, "high", transform=ax_rank.transAxes, ha="right", va="top", fontsize=5.2, color=SEMANTIC["axis_grey"])
    add_panel_label(ax_rank, "b", x=-0.035, y=1.02)

    ax_ref.axhline(cross_mean, color=SEMANTIC["jsT_blue"], lw=0.65, ls="-", alpha=0.72, zorder=1)
    ax_ref.scatter(
        hv_x,
        cross_y,
        s=13,
        color=SEMANTIC["std_grey"],
        alpha=0.74,
        edgecolors="white",
        lw=0.25,
        zorder=3,
        label="benchmark stations",
    )
    ax_ref.set_ylabel("HVSR\nref.")
    ax_ref.set_xlabel("Testing-domain stations ordered by JsT-HVSR station mean")
    ax_ref.set_ylim(0.45, 1.00)
    ax_ref.set_yticks([0.5, 0.75, 1.0])
    ax_ref.grid(axis="y", alpha=0.12, lw=0.25)
    ax_ref.tick_params(axis="x", length=0, labelbottom=False)
    ax_ref.text(
        0.01,
        0.86,
        f"standard HVSR benchmark available for {len(hv_x)} stations; "
        f"mean cosine = {cross_mean:.2f}",
        transform=ax_ref.transAxes,
        ha="left",
        va="top",
        fontsize=5.25,
        color=SEMANTIC["axis_grey"],
    )

    # c: representative multi-event spectral residual curves.
    sub = gs[2, 0].subgridspec(1, 3, wspace=0.22)
    reps = [
        ("OK.CROK", "Sedimentary basin", SEMANTIC["jsT_blue"]),
        ("AK.BWN", "Low-amplification site", SEMANTIC["token_green"]),
        ("AK.RC01", "Subduction setting", SEMANTIC["std_grey"]),
    ]
    for i, (sid, label, color) in enumerate(reps):
        ax = fig.add_subplot(sub[0, i])
        curves = np.asarray(hvsr.get(sid, {}).get("curves", []), dtype=float)
        if curves.size:
            for k, curve in enumerate(curves):
                ax.plot(
                    FREQ_CENTERS,
                    curve,
                    color=color,
                    alpha=0.20 + 0.25 * k / max(len(curves) - 1, 1),
                    lw=0.45,
                    drawstyle="steps-mid",
                )
            ax.plot(FREQ_CENTERS, curves.mean(axis=0), color=SEMANTIC["black"], lw=1.0)
            if len(curves) > 1:
                vals = [pearsonr(curves[a], curves[b])[0] for a in range(len(curves)) for b in range(a + 1, len(curves))]
                txt = f"r = {np.mean(vals):.2f}\nn = {len(curves)}"
            else:
                txt = f"n = {len(curves)}"
            ax.text(
                0.96,
                0.92,
                txt,
                transform=ax.transAxes,
                fontsize=5.2,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.78, lw=0.25, pad=1.1),
            )
        ax.axhline(0, color=SEMANTIC["null_grey"], ls="--", lw=0.35, alpha=0.55)
        ax.set_xscale("log")
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("JsT-HVSR\nlog$_{10}$(res / pred)")
            add_panel_label(ax, "c", x=-0.22, y=1.05)
        ax.set_title(f"{sid}\n{label}", fontsize=5.9, pad=2.0)
        ax.grid(True, alpha=0.10, lw=0.2)

    # d: residual-correlation distributions and label-shuffled effect-size control.
    subc = gs[2, 1].subgridspec(2, 1, height_ratios=[1.16, 0.58], hspace=0.52)
    ax = fig.add_subplot(subc[0, 0])
    ax_eff = fig.add_subplot(subc[1, 0])

    bins = np.linspace(-0.38, 0.82, 82)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def _ridge(values, baseline, color, fill_alpha, label, mean_value):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        n_total = len(values)
        values = values[(values >= bins[0]) & (values <= bins[-1])]
        dens, _ = np.histogram(values, bins=bins, density=True)
        dens = uniform_filter1d(dens, size=3, mode="nearest")
        dens = dens / max(np.nanmax(dens), 1e-12) * 0.36
        ax.fill_between(centers, baseline, baseline + dens, color=color, alpha=fill_alpha, lw=0)
        ax.plot(centers, baseline + dens, color=color, lw=0.9)
        ax.plot([mean_value, mean_value], [baseline, baseline + 0.40], color=color, lw=1.05)
        ax.text(
            bins[0] + 0.015,
            baseline + 0.28,
            f"{label}\n$n$ = {n_total:,}",
            ha="left",
            va="center",
            fontsize=5.5,
            color=SEMANTIC["black"],
        )
        ax.text(
            mean_value + 0.012,
            baseline + 0.39,
            f"mean $r$ = {mean_value:.3f}",
            ha="left",
            va="bottom",
            fontsize=5.2,
            color=color,
        )

    _ridge(stats["inter_vals"], 0.12, SEMANTIC["std_grey_light"], 0.34, "different station", stats["inter"])
    _ridge(stats["intra_vals"], 0.70, SEMANTIC["jsT_blue"], 0.30, "same station", stats["intra"])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.02, 1.16)
    ax.set_xlabel("Time-domain residual Pearson $r$", labelpad=1.5)
    ax.set_ylabel("Pair class")
    ax.set_yticks([0.12, 0.70])
    ax.set_yticklabels(["different", "same"])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.grid(axis="x", alpha=0.10, lw=0.25)
    ax.text(
        0.98,
        0.96,
        f"$\\Delta r$ = {stats['delta']:+.3f}\n{stats['ratio']:.1f}$\\times$ higher",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.88, lw=0.25, pad=1.4),
    )
    add_panel_label(ax, "d", x=-0.13, y=1.05)

    effect_labels = ["observed\nlabels", "shuffled\nlabels"]
    effect_vals = np.array([stats["delta"], stats["null_delta"]], dtype=float)
    effect_cols = [SEMANTIC["jsT_blue"], SEMANTIC["std_grey_light"]]
    x_eff = np.arange(2)
    ax_eff.axhline(0, color=SEMANTIC["null_grey"], lw=0.55, ls="--", zorder=1)
    for x0, val, col, alpha in zip(x_eff, effect_vals, effect_cols, [0.88, 0.70]):
        ax_eff.bar(x0, val, width=0.48, color=col, alpha=alpha, edgecolor="white", lw=0.35, zorder=2)
    for x0, val, col in zip(x_eff, effect_vals, effect_cols):
        va = "bottom" if val >= 0 else "top"
        dy = 0.004 if val >= 0 else -0.004
        ax_eff.text(x0, val + dy, f"{val:+.3f}", ha="center", va=va, fontsize=5.4, color=SEMANTIC["black"])
    ax_eff.set_xticks(x_eff)
    ax_eff.set_xticklabels(effect_labels)
    ax_eff.set_ylabel("$\\Delta r$")
    ax_eff.set_xlabel("")
    ax_eff.set_ylim(-0.018, 0.086)
    ax_eff.set_yticks([0.00, 0.04, 0.08])
    ax_eff.grid(axis="y", alpha=0.14, lw=0.25)
    ax_eff.text(
        0.98,
        0.95,
        "station-label control",
        transform=ax_eff.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=SEMANTIC["axis_grey"],
    )

    save_panel(fig, FIG_DIR / "fig1_ng")
    plt.close(fig)
    return {
        "fig1": {
            "n_stations": len(station_ids),
            "n_events": int(np.sum(events)),
            "intra": stats["intra"],
            "inter": stats["inter"],
            "delta": stats["delta"],
            "ratio": stats["ratio"],
            "null_delta": stats["null_delta"],
            "ranked_station_count": len(rank_ids),
            "ranked_station_min_amp": float(np.nanmin(rank_means)),
            "ranked_station_max_amp": float(np.nanmax(rank_means)),
            "standard_hvsr_reference_stations": int(len(hv_x)),
            "standard_hvsr_reference_rank_rho": hv_rank_rho,
            "jst_standard_cross_method_rank_rho": cross_rank_rho,
            "kiknet_station_min2_rho": j_station_min2["rho"],
            "kiknet_station_min2_ci95": j_station_min2["block_bootstrap_ci95"],
        }
    }


def draw_fig2() -> dict:
    """Fig. 2: measured-site validation of JsT-HVSR."""
    df = pd.read_csv(VS30_CSV)
    corr = load_json(VS30_STATS)["correlations"]
    expa = load_json(EXPA)
    expi = load_json(KIKNET_EXP_I)
    _, kiknet_results = load_kiknet_dense_station_results()
    kiknet_records = load_kiknet_dense_records()
    kiknet_qc_records = kiknet_records[kiknet_records["arrival_qc"]].copy()
    kiknet_full_records = kiknet_records[kiknet_records["full_pre_qc"]].copy()
    kiknet_all = metric_row(KIKNET_EXP_H_FREQ, "all", "1-10Hz")
    kiknet_pooled = metric_row(KIKNET_EXP_H_FREQ, "arrival_qc", "1-10Hz")
    kiknet_full = metric_row(KIKNET_EXP_H_FREQ, "full_pre_qc", "1-10Hz")
    kiknet_std = metric_row(KIKNET_EXP_H_STD_FREQ, "arrival_qc", "1-3Hz")
    kiknet_boot = bootstrap_one_record_rhos(kiknet_qc_records, "1-10Hz")
    kiknet_event = pd.read_csv(KIKNET_EXP_H_FREQ)
    kiknet_event = kiknet_event[
        (kiknet_event["scope"] == "arrival_qc")
        & (kiknet_event["metric"] == "1-10Hz")
        & (kiknet_event["event_id"].astype(str) != "pooled")
    ].copy()

    proxy = df[df["vs30_kind"] == "proxy"].copy()
    x_proxy, y_proxy = finite_xy(proxy["vs30"], proxy["mean_amp"])
    x_all, y_all = finite_xy(df["vs30"], df["mean_amp"])

    apply_nature_style(6.2)
    fig = plt.figure(figsize=(DOUBLE_COL, 165 / 25.4))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[0.92, 1.08],
        height_ratios=[1.48, 0.82, 0.82],
        left=0.066,
        right=0.985,
        top=0.965,
        bottom=0.088,
        hspace=0.46,
        wspace=0.34,
    )

    # a: direct single-record measured-Vs30 validation in Japan.
    ax = fig.add_subplot(gs[0, :])
    x_jp_all, y_jp_all = finite_xy(kiknet_records["vs30"], kiknet_records["1-10Hz"])
    x_jp, y_jp = finite_xy(kiknet_qc_records["vs30"], kiknet_qc_records["1-10Hz"])
    x_jp_full, y_jp_full = finite_xy(kiknet_full_records["vs30"], kiknet_full_records["1-10Hz"])
    ax.scatter(
        x_jp_all,
        y_jp_all,
        c=SEMANTIC["bg_grey"],
        s=8.5,
        alpha=0.17,
        edgecolors="none",
        label=f"all records (N={int(kiknet_all['N'])})",
        zorder=1,
    )
    ax.scatter(
        x_jp,
        y_jp,
        c=SEMANTIC["token_green"],
        s=12,
        alpha=0.28,
        edgecolors="white",
        lw=0.10,
        label=f"arrival/window QC (N={int(kiknet_pooled['N'])})",
        zorder=2,
    )
    ax.scatter(
        x_jp_full,
        y_jp_full,
        c=SEMANTIC["jsT_blue"],
        s=19,
        alpha=0.68,
        edgecolors="white",
        lw=0.25,
        label=f"full-pre QC (N={int(kiknet_full['N'])})",
        zorder=3,
    )
    coef = np.polyfit(np.log10(x_jp), y_jp, 1)
    x_fit = np.logspace(np.log10(x_jp.min()), np.log10(x_jp.max()), 180)
    ax.plot(x_fit, np.polyval(coef, np.log10(x_fit)), color=SEMANTIC["black"], lw=1.05, zorder=4)
    fit_label_x = 1180
    ax.text(
        fit_label_x,
        np.polyval(coef, np.log10(fit_label_x)) - 0.035,
        "log-linear fit",
        ha="left",
        va="top",
        fontsize=5.1,
        color=SEMANTIC["black"],
        rotation=-7,
        rotation_mode="anchor",
    )
    bins = pd.qcut(np.log10(kiknet_qc_records["vs30"]), q=8, duplicates="drop")
    binned = (
        kiknet_qc_records.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(
            x=("vs30", "median"),
            y=("1-10Hz", "median"),
            q25=("1-10Hz", lambda v: np.percentile(v, 25)),
            q75=("1-10Hz", lambda v: np.percentile(v, 75)),
            n=("1-10Hz", "size"),
        )
    )
    ax.errorbar(
        binned["x"],
        binned["y"],
        yerr=[binned["y"] - binned["q25"], binned["q75"] - binned["y"]],
        fmt="o",
        ms=4.8,
        lw=0.75,
        capsize=2.3,
        color=SEMANTIC["jsT_blue"],
        mfc="white",
        mec=SEMANTIC["jsT_blue"],
        mew=1.0,
        zorder=6,
        label="log-Vs30 bins",
    )
    station_mean = (
        kiknet_qc_records.groupby("station_id", as_index=False)
        .agg(vs30=("vs30", "first"), band=("1-10Hz", "mean"), n_events=("event_id", "nunique"))
    )
    station_mean_min2 = station_mean[station_mean["n_events"] >= 2].copy()
    station_mean_min2_r = spearmanr(station_mean_min2["band"], station_mean_min2["vs30"])
    station_mean_min2_rho = float(station_mean_min2_r.statistic)
    station_mean_min2_p = float(station_mean_min2_r.pvalue)
    ax.set_xscale("log")
    ax.set_xticks([120, 200, 360, 760, 1500])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(95, 2300)
    ymin = max(-0.08, float(np.nanpercentile(y_jp_all, 0.4)) - 0.03)
    ymax = min(1.38, float(np.nanpercentile(y_jp_all, 99.6)) + 0.05)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("KiK-net profile Vs30 (m s$^{-1}$)")
    ax.set_ylabel("Single-record JsT residual band (1--10 Hz)")
    ax.set_title("Dense KiK-net waveforms provide the measured-profile test", loc="left", fontsize=6.8, pad=2.4)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=5.6,
        frameon=False,
        handletextpad=0.35,
        borderaxespad=0.1,
    )
    ax.text(
        0.018,
        0.93,
        f"7 earthquakes; {int(kiknet_pooled['N'])} QC records\n"
        f"$\\rho$ = {kiknet_pooled['rho']:+.2f}; p = {kiknet_pooled['p']:.1e}\n"
        f"spatial partial r = {kiknet_pooled['partial_latlon_r']:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.88, lw=0.25, pad=1.4),
    )
    ax.text(
        0.985,
        0.075,
        f"station means, QC $\\geq$2:\n$\\rho$ = {station_mean_min2_rho:+.2f}; N = {len(station_mean_min2)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=SEMANTIC["axis_grey"],
    )
    ax.grid(True, alpha=0.10, lw=0.2)
    add_panel_label(ax, "a", x=-0.042, y=1.02)

    # b: proxy Vs30 validation as a broad U.S. anchor.
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(
        x_all,
        y_all,
        c=SEMANTIC["bg_grey"],
        s=10,
        alpha=0.23,
        edgecolors="none",
        label=f"all matched (N={corr['ALL']['N']})",
    )
    ax.scatter(
        x_proxy,
        y_proxy,
        c=SEMANTIC["jsT_blue"],
        s=17,
        alpha=0.72,
        edgecolors="white",
        lw=0.25,
        label=f"proxy Vs30 (N={corr['proxy']['N']})",
    )
    logx = np.log10(x_proxy)
    coef = np.polyfit(logx, y_proxy, 1)
    x_fit = np.logspace(np.log10(x_proxy.min()), np.log10(x_proxy.max()), 120)
    y_fit = np.polyval(coef, np.log10(x_fit))
    ax.plot(x_fit, y_fit, color=SEMANTIC["black"], lw=0.95)
    fit_label_x = 1030
    ax.text(
        fit_label_x,
        np.polyval(coef, np.log10(fit_label_x)) - 0.020,
        "log-linear fit",
        ha="left",
        va="top",
        fontsize=5.0,
        color=SEMANTIC["black"],
        rotation=-14,
        rotation_mode="anchor",
    )
    ax.set_xscale("log")
    ax.set_xticks([200, 400, 800, 1600])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("USGS proxy Vs30 (m s$^{-1}$)")
    ax.set_ylabel("Mean JsT-HVSR amplification")
    ax.set_title("Broad U.S. proxy anchor", loc="left", fontsize=6.4, pad=2.0)
    ax.legend(loc="upper right", fontsize=5.1, frameon=False, handletextpad=0.3)
    ax.text(
        0.04,
        0.95,
        f"$\\rho$ = {corr['proxy']['rho']:+.2f}; p = {corr['proxy']['p']:.1e}; N = {corr['proxy']['N']}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.0,
        bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.82, lw=0.25, pad=1.4),
    )
    lono = expa["leave_one_network_out"]
    lono_vals = [d["rho"] for d in lono]
    boot = expa["bootstrap"]["rho_95ci"]
    spatial = next(d for d in expa["partial_correlation"] if d["label"] == "Spatial (lon, lat)")
    ax.text(
        0.04,
        0.08,
        f"spatial partial r = {spatial['r_partial']:+.2f}\n"
        f"leave-one-network-out $\\rho$ [{min(lono_vals):+.2f}, {max(lono_vals):+.2f}]",
        transform=ax.transAxes,
        va="bottom",
        fontsize=5.4,
        color=SEMANTIC["axis_grey"],
    )
    ax.grid(True, alpha=0.10, lw=0.2)
    add_panel_label(ax, "b", x=-0.12, y=1.04)

    # c: compact QC and frequency gradient.
    ax = fig.add_subplot(gs[1, 1])
    meta = expi["metadata_controls"]
    spatial = expi["spatial_block_controls"]["station_mean_min2"]
    screening = expi["screening_utility"]["station_means_min2"]
    event_fixed = expi["event_fixed_controls"]["event_fixed_rank"]
    event_partial = expi["event_fixed_controls"]["event_fixed_rank_spatial_residual"]
    boot = kiknet_boot[np.isfinite(kiknet_boot)]
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    boot_median = float(np.median(boot))
    freq_rows = []
    for scope, label in [
        ("all", "all records"),
        ("arrival_qc", "arrival/window QC"),
        ("full_pre_qc", "full-pre QC"),
    ]:
        for metric in ["0.5-5Hz", "1-10Hz", "3-10Hz"]:
            row = metric_row(KIKNET_EXP_H_FREQ, scope, metric)
            freq_rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "metric": metric,
                    "rho": row["rho"],
                    "N": int(row["N"]),
                }
            )
    band_cols = {
        "0.5-5Hz": SEMANTIC["token_green"],
        "1-10Hz": SEMANTIC["jsT_blue"],
        "3-10Hz": "#7C6F9B",
    }
    scope_y = {"all": 0, "arrival_qc": 1, "full_pre_qc": 2}
    scope_offset = {"0.5-5Hz": -0.13, "1-10Hz": 0.0, "3-10Hz": 0.13}
    ax.axvline(0, color=SEMANTIC["null_grey"], lw=0.65, ls="--", zorder=0)
    for metric in ["0.5-5Hz", "1-10Hz", "3-10Hz"]:
        xs = [r["rho"] for r in freq_rows if r["metric"] == metric]
        ys = [scope_y[r["scope"]] + scope_offset[metric] for r in freq_rows if r["metric"] == metric]
        ax.plot(xs, ys, color=band_cols[metric], lw=0.65, alpha=0.42, zorder=1)
    for r in freq_rows:
        y0 = scope_y[r["scope"]] + scope_offset[r["metric"]]
        ax.plot(
            r["rho"],
            y0,
            "o",
            color=band_cols[r["metric"]],
            ms=4.2 if r["metric"] == "1-10Hz" else 3.7,
            alpha=0.86,
            mec="white",
            mew=0.25,
            zorder=3,
        )
    for scope, label in [("all", "all records"), ("arrival_qc", "arrival/window QC"), ("full_pre_qc", "full-pre QC")]:
        row = metric_row(KIKNET_EXP_H_FREQ, scope, "1-10Hz")
        ax.text(
            0.012,
            scope_y[scope],
            f"N={int(row['N'])}",
            ha="left",
            va="center",
            fontsize=5.0,
            color=SEMANTIC["axis_grey"],
        )
    for i, metric in enumerate(["0.5-5Hz", "1-10Hz", "3-10Hz"]):
        x0 = 0.50 + 0.15 * i
        ax.plot([x0], [0.94], "o", transform=ax.transAxes, color=band_cols[metric], ms=4.0, mec="white", mew=0.2, clip_on=False)
        ax.text(
            x0 + 0.015,
            0.94,
            metric.replace("Hz", " Hz"),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=5.1,
            color=SEMANTIC["axis_grey"],
        )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["all records", "arrival/window QC", "full-pre QC"])
    ax.set_ylim(-0.45, 2.45)
    ax.set_xlim(-0.39, 0.035)
    ax.set_xlabel("Spearman $\\rho$ to profile Vs30")
    ax.set_title("Window quality sharpens the site ranking", loc="left", fontsize=6.4, pad=2.0)
    ax.grid(True, axis="x", alpha=0.10, lw=0.2)
    add_panel_label(ax, "c", x=-0.12, y=1.04)

    # d: station-ranked KiK-net measurement landscape.
    ax = fig.add_subplot(gs[2, :])
    station_rank = station_mean_min2.sort_values("vs30", kind="mergesort").reset_index(drop=True).copy()
    station_rank["rank"] = np.arange(len(station_rank), dtype=float)
    rank_map = dict(zip(station_rank["station_id"], station_rank["rank"]))
    rec_rank = kiknet_qc_records[kiknet_qc_records["station_id"].isin(rank_map)].copy()
    rec_rank["rank"] = rec_rank["station_id"].map(rank_map).astype(float)
    rng = np.random.default_rng(42)
    rec_rank["_x"] = rec_rank["rank"] + rng.uniform(-0.18, 0.18, size=len(rec_rank))
    soft_idx = np.searchsorted(station_rank["vs30"].to_numpy(float), 360.0, side="right") - 0.5
    nehrp_b_idx = np.searchsorted(station_rank["vs30"].to_numpy(float), 760.0, side="right") - 0.5
    ax.axvspan(-0.5, max(soft_idx, -0.5), color="#EEF3F6", alpha=0.68, zorder=0)
    if nehrp_b_idx < len(station_rank) - 0.5:
        ax.axvspan(nehrp_b_idx, len(station_rank) - 0.5, color="#F5F3EE", alpha=0.72, zorder=0)
    if soft_idx > -0.5:
        ax.axvline(soft_idx, color=SEMANTIC["axis_grey"], lw=0.42, alpha=0.34, zorder=1)
    if nehrp_b_idx < len(station_rank) - 0.5:
        ax.axvline(nehrp_b_idx, color=SEMANTIC["axis_grey"], lw=0.42, alpha=0.34, zorder=1)
    ax.scatter(
        rec_rank["_x"],
        rec_rank["1-10Hz"],
        s=5.5,
        color=SEMANTIC["token_green"],
        alpha=0.20,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        station_rank["rank"],
        station_rank["band"],
        s=8.5,
        color=SEMANTIC["jsT_blue"],
        alpha=0.74,
        edgecolors="white",
        lw=0.15,
        zorder=3,
    )
    smooth = station_rank["band"].rolling(27, center=True, min_periods=7).median()
    ax.plot(
        station_rank["rank"],
        smooth,
        color=SEMANTIC["black"],
        lw=1.05,
        zorder=4,
    )
    ax.scatter([], [], s=12, color=SEMANTIC["token_green"], alpha=0.45, edgecolors="none", label="single records")
    ax.scatter([], [], s=12, color=SEMANTIC["jsT_blue"], alpha=0.80, edgecolors="white", lw=0.15, label="station means")
    ax.plot([], [], color=SEMANTIC["black"], lw=1.05, label="rolling median")
    q = np.linspace(0, len(station_rank) - 1, 5).round().astype(int)
    ax.set_xticks(q)
    ax.set_xticklabels([f"{station_rank.loc[i, 'vs30']:.0f}" for i in q])
    ax.set_xlim(-0.5, len(station_rank) - 0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("KiK-net stations ordered by profile Vs30 (tick labels: m s$^{-1}$)")
    ax.set_ylabel("Single-record JsT band\n(1--10 Hz)")
    ax.set_title("Station-ranked residuals form a soft-to-hard site measurement landscape", loc="left", fontsize=6.4, pad=2.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=3,
        fontsize=5.0,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    ax.text(
        0.012,
        0.93,
        f"{len(station_rank)} stations with QC $\\geq$2\n"
        f"station mean $\\rho$ = {station_mean_min2_rho:+.2f}\n"
        f"one-record median $\\rho$ = {boot_median:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.84, lw=0.25, pad=1.2),
    )
    ax.text(
        0.985,
        0.92,
        f"soft-site screen: AUC {screening['auc_vs30_lt_360']:.2f};\n"
        f"top quartile {screening['top_score_quartile_soft_enrichment']:.1f}$\\times$ enriched",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.2,
        color=SEMANTIC["black"],
        bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.82, lw=0.20, pad=1.0),
    )
    ax.text(
        max(soft_idx, 0) / max(len(station_rank) - 1, 1),
        0.06,
        "Vs30 < 360",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color=SEMANTIC["axis_grey"],
    )
    if nehrp_b_idx < len(station_rank) - 0.5:
        ax.text(
            nehrp_b_idx / max(len(station_rank) - 1, 1),
            0.06,
            "Vs30 > 760",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.0,
            color=SEMANTIC["axis_grey"],
        )
    ax.grid(True, axis="y", alpha=0.10, lw=0.2)
    add_panel_label(ax, "d", x=-0.042, y=1.04)

    save_panel(fig, FIG_DIR / "fig2_ng")
    plt.close(fig)
    return {
        "fig2": {
            "proxy_rho": corr["proxy"]["rho"],
            "proxy_p": corr["proxy"]["p"],
            "proxy_n": corr["proxy"]["N"],
            "kiknet_single_record_rho": kiknet_pooled["rho"],
            "kiknet_single_record_p": kiknet_pooled["p"],
            "kiknet_single_record_n": int(kiknet_pooled["N"]),
            "kiknet_single_record_partial_r": kiknet_pooled["partial_latlon_r"],
            "kiknet_station_qc_min2_rho": station_mean_min2_rho,
            "kiknet_station_qc_min2_p": station_mean_min2_p,
            "kiknet_station_qc_min2_n": len(station_mean_min2),
            "kiknet_one_record_bootstrap_median_rho": boot_median,
            "kiknet_one_record_bootstrap_ci95": [ci_low, ci_high],
            "kiknet_standard_hvsr_same_window_rho": kiknet_std["rho"],
            "kiknet_full_pre_qc_rho": kiknet_full["rho"],
            "metadata_residualized_r": meta["metadata_residualized_jst"]["value"],
            "metadata_only_score_vs_vs30_rho": meta["metadata_only_score_vs_vs30"]["value"],
            "spatial_block_bootstrap_median": spatial["block_bootstrap_median"],
            "spatial_block_bootstrap_ci95": spatial["block_bootstrap_ci95"],
            "event_fixed_rank_r": event_fixed["value"],
            "event_fixed_spatial_residual_r": event_partial["value"],
            "station_min2_soft_site_auc": screening["auc_vs30_lt_360"],
            "station_min2_soft_site_enrichment": screening["top_score_quartile_soft_enrichment"],
        }
    }


def draw_complementarity() -> dict:
    """Optional supplement-style figure: JsT-HVSR and standard HVSR complementarity."""
    single = load_json(SINGLE_MULTI)
    cross = load_json(EXPB_CROSS)
    follow = load_json(EXPB_FOLLOWUP)
    stations = single["stations"]

    jst_vals = np.array([s["jst_single_to_multi_mean"] for s in stations], dtype=float)
    hv_vals = np.array([s["hv_single_to_multi_mean"] for s in stations], dtype=float)
    cross_vals = np.array([s["jst_single_vs_hv_multi"] for s in stations], dtype=float)
    groups = np.array([group_for_complementarity(s["station_id"]) for s in stations])
    group_colors = {
        "sedimentary": SEMANTIC["jsT_blue"],
        "complex": SEMANTIC["neg_red_light"],
        "other": SEMANTIC["bg_grey"],
    }

    apply_nature_style(6.2)
    fig = plt.figure(figsize=(DOUBLE_COL, 125 / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.24, 1.0],
        height_ratios=[1.02, 1.0],
        left=0.07,
        right=0.985,
        top=0.955,
        bottom=0.115,
        hspace=0.40,
        wspace=0.34,
    )

    # a: within- and cross-method distributions.
    ax = fig.add_subplot(gs[:, 0])
    data = [jst_vals, hv_vals, cross_vals]
    colors = [SEMANTIC["jsT_blue"], SEMANTIC["std_grey"], SEMANTIC["token_green"]]
    labels = ["JsT\nsingle→multi", "standard HVSR\nsingle→multi", "JsT→standard\ncross-method"]
    for i, vals in enumerate(data):
        x = np.full(len(vals), i, dtype=float) + jitter(len(vals), 0.075, seed=30 + i)
        ax.scatter(x, vals, s=14, color=colors[i], alpha=0.42, edgecolors="none", zorder=2)
        mean = float(np.mean(vals))
        sd = float(np.std(vals))
        ax.errorbar(i, mean, yerr=sd, fmt="o", color=SEMANTIC["black"], ms=4.2, lw=0.85, capsize=3.5, zorder=4)
        ax.text(i, mean + sd + 0.018, f"{mean:.3f}", ha="center", va="bottom", fontsize=5.6, color=SEMANTIC["black"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0.38, 1.01)
    ax.text(
        0.03,
        0.05,
        f"cross / JsT within = {cross['cross_vs_jst_within_ratio']:.2f}$\\times$\n"
        f"cross / standard within = {cross['cross_vs_hv_within_ratio']:.2f}$\\times$",
        transform=ax.transAxes,
        fontsize=5.7,
        color=SEMANTIC["axis_grey"],
    )
    ax.grid(True, axis="y", alpha=0.11, lw=0.2)
    add_panel_label(ax, "a", x=-0.09, y=1.02)

    # b: global decoupling of station-level method behaviour.
    ax = fig.add_subplot(gs[0, 1])
    for g in ["other", "sedimentary", "complex"]:
        mask = groups == g
        ax.scatter(jst_vals[mask], hv_vals[mask], s=18, color=group_colors[g], alpha=0.62 if g != "other" else 0.32, edgecolors="white", lw=0.2, label=g)
    rho, pval = spearmanr(jst_vals, hv_vals)
    ax.set_xlabel("JsT single-event\nself-consistency")
    ax.set_ylabel("standard HVSR\nself-consistency")
    ax.text(
        0.05,
        0.93,
        f"$\\rho$ = {rho:+.3f}; p = {pval:.2f}\nN = {len(stations)} stations",
        transform=ax.transAxes,
        va="top",
        fontsize=5.6,
        bbox=dict(facecolor="white", edgecolor=SEMANTIC["axis_grey"], alpha=0.82, lw=0.25, pad=1.3),
    )
    ax.legend(loc="lower right", fontsize=5.0, frameon=False, handletextpad=0.2)
    ax.grid(True, alpha=0.10, lw=0.2)
    add_panel_label(ax, "b", x=-0.15, y=1.04)

    # c: geological endmember dependence of cross-method agreement.
    ax = fig.add_subplot(gs[1, 1])
    class_labels = {
        "0": "craton",
        "1": "sedimentary\nbasin",
        "2": "basin-range /\nactive margin",
        "3": "volcanic\narc",
        "4": "subduction /\nvolcanic",
    }
    class_items = [(k, follow["per_class"][k]) for k in ["0", "1", "2", "3", "4"] if k in follow["per_class"]]
    y = np.arange(len(class_items))
    vals = np.array([d["cross_cos_mean"] for _, d in class_items], dtype=float)
    errs = np.array([d["cross_cos_std"] for _, d in class_items], dtype=float)
    ns = [d["n"] for _, d in class_items]
    cols = [
        SEMANTIC["jsT_blue"] if k == "1" else SEMANTIC["neg_red_light"] if k == "4" else SEMANTIC["bg_grey"]
        for k, _ in class_items
    ]
    ax.barh(y, vals, xerr=errs, color=cols, alpha=0.88, edgecolor="white", lw=0.25, height=0.58)
    for yi, val, n in zip(y, vals, ns):
        ax.text(val + 0.018, yi, f"N={n}", va="center", fontsize=5.2, color=SEMANTIC["axis_grey"])
    ax.axvline(cross["cross_method_mean"], color=SEMANTIC["std_grey"], ls="--", lw=0.65, alpha=0.68)
    ax.text(cross["cross_method_mean"] + 0.008, len(y) - 0.15, "global mean", fontsize=5.0, color=SEMANTIC["axis_grey"], rotation=90, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels([class_labels[k] for k, _ in class_items])
    ax.set_xlabel("JsT-to-standard cosine")
    ax.set_xlim(0.48, 0.92)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.10, lw=0.2)
    add_panel_label(ax, "c", x=-0.15, y=1.04)

    save_panel(fig, FIG_DIR / "complementarity_ng")
    plt.close(fig)
    return {
        "complementarity": {
            "cross_method_mean": cross["cross_method_mean"],
            "cross_method_std": cross["cross_method_std"],
            "jst_within_mean": cross["jst_within_mean"],
            "standard_within_mean": cross["hv_within_mean"],
            "rank_spearman": cross["rank_spearman"],
            "rank_p_value": cross["rank_p_value"],
            "basin_cross_mean": follow["basin_cross_mean"],
            "non_basin_cross_mean": follow["non_basin_cross_mean"],
        }
    }


def draw_fig3() -> dict:
    """Fig. 3: receiver-site token geological manifold."""
    pca = np.load(FIG3_CACHE / "pca_projection.npz", allow_pickle=True)
    X = pca["X_pca_2d"]
    station_ids = pca["station_ids"]
    geo_labels = pca["geo_labels"]
    intra = load_json(FIG3_CACHE / "intra_summary.json")
    expc = load_json(EXPC)

    key_groups = ["Sedimentary_centralUS", "Sedimentary_basin", "Metamorphic_range", "Active_margin"]
    key_colors = {
        "Sedimentary_centralUS": SEMANTIC["jsT_blue"],
        "Sedimentary_basin": SEMANTIC["jsT_blue_light"],
        "Metamorphic_range": "#9B8CAC",
        "Active_margin": SEMANTIC["token_green_light"],
    }

    apply_nature_style(6.2)
    fig = plt.figure(figsize=(DOUBLE_COL, 138 / 25.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.36, 1.0],
        height_ratios=[0.92, 1.16],
        left=0.065,
        right=0.985,
        top=0.955,
        bottom=0.085,
        hspace=0.34,
        wspace=0.30,
    )

    # a: token 7 PCA manifold.
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(X[:, 0], X[:, 1], s=8, color=SEMANTIC["bg_grey"], alpha=0.30, edgecolors="none")
    for g in key_groups:
        mask = geo_labels == g
        if not np.any(mask):
            continue
        ax.scatter(X[mask, 0], X[mask, 1], s=14, color=key_colors[g], alpha=0.78, edgecolors="white", lw=0.18)
        center = np.median(X[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            g.replace("_", " "),
            fontsize=5.7,
            color=key_colors[g],
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        )
    ax.set_xlabel("Token 7 PC1")
    ax.set_ylabel("Token 7 PC2")
    ax.text(
        0.03,
        0.96,
        f"{len(station_ids)} stations; {intra['n_geo_groups']} geological groups\n"
        f"30 PCs explain {float(pca['pca_explained_30d']):.1%}; 2D view explains {float(pca['pca_explained_2d']):.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=5.5,
        color=SEMANTIC["axis_grey"],
    )
    ax.grid(True, alpha=0.09, lw=0.2)
    add_panel_label(ax, "a", x=-0.09, y=1.02)

    # b: intra-group token consistency.
    ax = fig.add_subplot(gs[0, 1])
    groups = [
        item
        for item in sorted(intra["groups"].items(), key=lambda item: item[1]["mean_cos"], reverse=True)
        if not item[0].startswith("Basalt_")
    ]
    # Keep the right panel compact: top geological groups plus two broad controls.
    show_names = []
    for name, _ in groups:
        if len(show_names) < 8 or name in {"Sedimentary_basin", "Metamorphic_range"}:
            show_names.append(name)
        if len(show_names) >= 9:
            break
    vals = [intra["groups"][g]["mean_cos"] for g in show_names]
    ns = [intra["groups"][g]["n_stations"] for g in show_names]
    pair_weights = np.array([max(intra["groups"][g]["n_pairs"], 1) for g in show_names], dtype=float)
    shown_mean = float(np.average(vals, weights=pair_weights))
    y = np.arange(len(show_names))
    cols = [key_colors.get(g, SEMANTIC["bg_grey"]) for g in show_names]
    ax.barh(y, vals, color=cols, edgecolor="white", lw=0.25, height=0.58)
    ax.axvline(shown_mean, color=SEMANTIC["std_grey"], ls="--", lw=0.65, alpha=0.70)
    for yi, val, n in zip(y, vals, ns):
        ax.text(min(val + 0.018, 1.035), yi, f"N={n}", va="center", fontsize=4.9, color=SEMANTIC["axis_grey"])
    ax.set_yticks(y)
    ax.set_yticklabels([g.replace("_", " ") for g in show_names], fontsize=5.2)
    ax.set_xlabel("Intra-group token 7 cosine")
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.text(
        0.04,
        0.08,
        f"shown-group mean = {shown_mean:.3f}\n"
        f"global $\\Delta$ = {intra['delta']:+.3f}",
        transform=ax.transAxes,
        fontsize=5.2,
        color=SEMANTIC["black"],
    )
    ax.grid(True, axis="x", alpha=0.10, lw=0.2)
    add_panel_label(ax, "b", x=-0.16, y=1.04)

    # c: controls at three non-local scales.
    sub = gs[1, 1].subgridspec(3, 1, hspace=0.66)
    t1 = expc["test1_coordinate_removed"]
    t3 = expc["test3_cross_region_basins"]

    ax1 = fig.add_subplot(sub[0, 0])
    vals1 = [t1["original_delta"], t1["ecef_removed_delta"]]
    ax1.bar([0, 1], vals1, color=[SEMANTIC["jsT_blue"], SEMANTIC["token_green"]], edgecolor="white", lw=0.25, width=0.55)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["original", "ECEF\nremoved"], fontsize=5.4)
    ax1.set_ylabel("$\\Delta$")
    ax1.text(
        0.98,
        0.88,
        f"geological $\\Delta$ retained {t1['delta_retained_ratio']:.0%}\n"
        f"ECEF variance explained {t1['ecef_variance_explained']:.1%}",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=5.0,
        color=SEMANTIC["axis_grey"],
    )
    ax1.set_title("global coordinate control", loc="left", fontsize=5.9, pad=1.5)
    ax1.grid(True, axis="y", alpha=0.10, lw=0.2)
    add_panel_label(ax1, "c", x=-0.16, y=1.20)

    ax2 = fig.add_subplot(sub[1, 0])
    vals2 = [
        t3["within_basin"]["Sedimentary_basin"],
        t3["within_basin"]["Sedimentary_centralUS"],
        t3["cross_basin"]["Sedimentary_basin_vs_Sedimentary_centralUS"],
    ]
    ax2.bar([0, 1, 2], vals2, color=[SEMANTIC["jsT_blue_light"], SEMANTIC["jsT_blue"], SEMANTIC["token_green"]], edgecolor="white", lw=0.25, width=0.56)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["OK\nwithin", "GS\nwithin", "OK-GS\ncross"], fontsize=5.2)
    ax2.set_ylabel("cosine")
    ax2.text(0.98, 0.86, f"cross = {vals2[2]:.3f}", transform=ax2.transAxes, ha="right", fontsize=5.1)
    ax2.set_title("same geology across >1,000 km", loc="left", fontsize=5.9, pad=1.5)
    ax2.grid(True, axis="y", alpha=0.10, lw=0.2)

    ax3 = fig.add_subplot(sub[2, 0])
    vals3 = [t3["mean_within"], t3["mean_cross"]]
    ax3.bar([0, 1], vals3, color=[SEMANTIC["jsT_blue"], SEMANTIC["bg_grey"]], edgecolor="white", lw=0.25, width=0.54)
    ax3.set_ylim(0, 1.0)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["within\nbasin groups", "between\nbasin groups"], fontsize=5.2)
    ax3.set_ylabel("cosine")
    ax3.text(0.98, 0.86, f"$\\Delta$ = {t3['delta']:+.3f}", transform=ax3.transAxes, ha="right", fontsize=5.1)
    ax3.set_title("basin-family consistency", loc="left", fontsize=5.9, pad=1.5)
    ax3.grid(True, axis="y", alpha=0.10, lw=0.2)

    save_panel(fig, FIG_DIR / "fig3_ng")
    plt.close(fig)
    return {
        "fig3": {
            "n_stations": intra["n_stations"],
            "n_geo_groups": intra["n_geo_groups"],
            "all_intra": intra["all_intra"],
            "all_inter": intra["all_inter"],
            "delta": intra["delta"],
            "shown_nonvolcanic_mean": shown_mean,
            "ecef_delta_retained_ratio": t1["delta_retained_ratio"],
            "ecef_variance_explained": t1["ecef_variance_explained"],
            "cross_region_basin_cos": t3["cross_basin"]["Sedimentary_basin_vs_Sedimentary_centralUS"],
            "basin_family_delta": t3["delta"],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Render the NG Methods figure and three Results figures.")
    parser.add_argument(
        "--fig",
        choices=["methods", "1", "2", "3", "complementarity"],
        help="Render a single figure. Complementarity is optional and not included in --all.",
    )
    args = parser.parse_args()

    if not args.all and not args.fig:
        parser.error("Use --all or --fig {methods,1,2,3,complementarity}.")

    stats = {}
    if args.all or args.fig == "methods":
        print("Rendering methods_ng...")
        stats.update(draw_methods())
    if args.all or args.fig == "1":
        print("Rendering fig1_ng...")
        stats.update(draw_fig1())
    if args.all or args.fig == "2":
        print("Rendering fig2_ng...")
        stats.update(draw_fig2())
    if args.all or args.fig == "3":
        print("Rendering fig3_ng...")
        stats.update(draw_fig3())
    if args.fig == "complementarity":
        print("Rendering complementarity_ng...")
        stats.update(draw_complementarity())
    save_stats(stats)
    print(f"Saved stats: {OUT_DIR / 'figure_stats.json'}")


if __name__ == "__main__":
    main()
