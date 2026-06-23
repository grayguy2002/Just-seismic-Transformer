"""Render a real KiK-net waveform plus its JsT site-effect measurement.

This is an extraction element for Fig. 4a artwork, not a manuscript result
panel.  The sample is a full-pre-arrival-QC KiK-net record with a strong
positive JsT residual/predicted spectral measurement.

Run from the project root:
  python3 manuscript/figures/fig4a_real_jst_sample.py
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from nature_geo_style import SEMANTIC, apply_nature_style


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent

EVENT_ID = "20140314020700"
STATION_CODE = "TTRH02"
STATION_ID = f"KIKNET.{STATION_CODE}"
EVENT_LABEL = "M6.2, 14 Mar 2014"

RAW_ZIP = ROOT / "data" / "kiknet_measured_vs30_pwave_v1" / "raw_zips" / f"{EVENT_ID}_ascii.zip"
EVENT_RECORDS = ROOT / "outputs" / "expH_kiknet_dense_arrival_qc_events" / "per_station_event_records.csv"

OUT_BASE = FIG_DIR / "fig4a_real_waveform_jst_measurement"
CLEAN_OUT_BASE = FIG_DIR / "fig4a_real_waveform_jst_measurement_clean"
MODEL_OUT_BASE = FIG_DIR / "fig4a_observed_predicted_jst_measurement"
MODEL_CLEAN_OUT_BASE = FIG_DIR / "fig4a_observed_predicted_jst_measurement_clean"
SOURCE_BASE = FIG_DIR / "fig4a_real_waveform_jst_measurement_source"
MODEL_SOURCE_BASE = FIG_DIR / "fig4a_observed_predicted_jst_measurement_source"
MODEL_SAMPLE_NPZ = ROOT / "outputs" / "fig4a_jst_waveform_sample" / "fig4a_jst_waveform_sample.npz"
MODEL_SAMPLE_SUMMARY = ROOT / "outputs" / "fig4a_jst_waveform_sample" / "fig4a_jst_waveform_sample_summary.json"

MODEL_SAMPLE_RATE_HZ = 40.0
RAW_COMPONENTS = {
    "N": "NS1",
    "E": "EW1",
    "Z": "UD1",
}

F_MIN, F_MAX, N_FREQ_BINS = 0.3, 15.0, 40
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
BAND_1_10 = np.array([1.0 <= f < 10.0 for f in FREQ_CENTERS])


def _parse_ascii_channel(path_in_zip: str) -> tuple[dict[str, str], np.ndarray, float]:
    """Parse a NIED ASCII channel into header, acceleration in gal, and Hz."""
    with zipfile.ZipFile(RAW_ZIP) as archive:
        lines = archive.read(path_in_zip).decode("ascii", errors="replace").splitlines()

    header: dict[str, str] = {}
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("Memo."):
            data_start = i + 1
            break
        key = line[:18].strip()
        value = line[18:].strip()
        if key:
            header[key] = value

    scale_text = header.get("Scale Factor", "")
    scale_match = re.search(r"([-+]?\d+(?:\.\d+)?)\(gal\)/([-+]?\d+(?:\.\d+)?)", scale_text)
    if not scale_match:
        raise ValueError(f"Cannot parse scale factor from {path_in_zip}: {scale_text!r}")
    scale = float(scale_match.group(1)) / float(scale_match.group(2))

    sample_text = header.get("Sampling Freq(Hz)", "")
    sample_match = re.search(r"([-+]?\d+(?:\.\d+)?)", sample_text)
    if not sample_match:
        raise ValueError(f"Cannot parse sample rate from {path_in_zip}: {sample_text!r}")
    sample_rate_hz = float(sample_match.group(1))

    values: list[int] = []
    for line in lines[data_start:]:
        values.extend(int(item) for item in re.findall(r"[-+]?\d+", line))
    raw = np.asarray(values, dtype=float)
    return header, raw * scale, sample_rate_hz


def load_waveform_window(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return time in seconds from P arrival and normalized N/E/Z waveforms."""
    raw_rate = None
    channels = []
    for label, suffix in RAW_COMPONENTS.items():
        member = f"{EVENT_ID}/kik/ascii/{STATION_CODE}{EVENT_ID[2:12]}.{suffix}"
        _, signal_gal, sample_rate_hz = _parse_ascii_channel(member)
        raw_rate = sample_rate_hz if raw_rate is None else raw_rate
        channels.append(signal_gal)

    if raw_rate is None:
        raise RuntimeError("No waveform channels were loaded")

    arrival_raw = float(row["arrival_sample"]) * raw_rate / MODEL_SAMPLE_RATE_HZ
    pre_s = 20.0
    post_s = 60.0
    crop_start = int(round(arrival_raw - pre_s * raw_rate))
    crop_stop = crop_start + int(round((pre_s + post_s) * raw_rate))
    if crop_start < 0 or crop_stop > min(len(ch) for ch in channels):
        raise ValueError(
            f"Requested crop [{crop_start}, {crop_stop}) outside raw trace length "
            f"{min(len(ch) for ch in channels)}"
        )

    cropped = []
    for signal_gal in channels:
        window = np.asarray(signal_gal[crop_start:crop_stop], dtype=float)
        pre_n = int(pre_s * raw_rate)
        window = window - np.median(window[:pre_n])
        window = resample_poly(window, up=2, down=5)
        cropped.append(window)
    waves = np.vstack(cropped)
    target_n = int((pre_s + post_s) * MODEL_SAMPLE_RATE_HZ)
    waves = waves[:, :target_n]

    global_scale = np.percentile(np.abs(waves), 99.5)
    if not np.isfinite(global_scale) or global_scale <= 0:
        global_scale = np.max(np.abs(waves))
    waves = waves / max(global_scale, 1e-12)
    times = np.arange(waves.shape[1], dtype=float) / MODEL_SAMPLE_RATE_HZ - pre_s
    return times, waves


def load_measurement_row() -> pd.Series:
    records = pd.read_csv(EVENT_RECORDS)
    subset = records[
        (records["station_id"].astype(str) == STATION_ID)
        & (records["event_id"].astype(str) == EVENT_ID)
    ]
    if subset.empty:
        raise ValueError(f"Missing Exp H record for {STATION_ID} {EVENT_ID}")
    return subset.iloc[0]


def save_source_data(times: np.ndarray, waves: np.ndarray, row: pd.Series, band_score: float) -> None:
    wave_df = pd.DataFrame(
        {
            "time_s_from_p": times,
            "north_normalized_acc": waves[0],
            "east_normalized_acc": waves[1],
            "vertical_normalized_acc": waves[2],
        }
    )
    wave_df.to_csv(SOURCE_BASE.with_name(SOURCE_BASE.name + "_waveform.csv"), index=False)

    hvsr = np.asarray([float(row[f"hvsr_bin_{i:02d}"]) for i in range(N_FREQ_BINS)], dtype=float)
    spec_df = pd.DataFrame(
        {
            "frequency_hz": FREQ_CENTERS,
            "jst_log10_residual_over_prediction": hvsr,
            "in_1_10_hz_measurement_band": BAND_1_10,
        }
    )
    spec_df.to_csv(SOURCE_BASE.with_name(SOURCE_BASE.name + "_spectrum.csv"), index=False)

    meta = pd.DataFrame(
        [
            {
                "station_id": STATION_ID,
                "event_id": EVENT_ID,
                "event_label": EVENT_LABEL,
                "profile_vs30_m_s": float(row["vs30"]),
                "nehrp": str(row["nehrp"]),
                "jst_1_10_hz_log10_score": band_score,
                "arrival_sample_in_raw_40hz_trace": float(row["arrival_sample"]),
                "source_magnitude": float(row["source_magnitude"]),
                "source_depth_km": float(row["source_depth_km"]),
                "path_ep_distance_km": float(row["path_ep_distance_km"]),
            }
        ]
    )
    meta.to_csv(SOURCE_BASE.with_name(SOURCE_BASE.name + "_metadata.csv"), index=False)


def load_model_sample() -> dict[str, np.ndarray] | None:
    if not MODEL_SAMPLE_NPZ.exists():
        return None
    sample = np.load(MODEL_SAMPLE_NPZ)
    return {key: np.asarray(sample[key]) for key in sample.files}


def save_model_source_data(model_sample: dict[str, np.ndarray], row: pd.Series, band_score: float) -> None:
    time = np.asarray(model_sample["time_s"], dtype=float)
    observed = np.asarray(model_sample["observed"], dtype=float)
    predicted = np.asarray(model_sample["predicted"], dtype=float)
    residual = np.asarray(model_sample["residual"], dtype=float)
    wave_df = pd.DataFrame(
        {
            "time_s_from_p": time,
            "observed_n": observed[0],
            "observed_e": observed[1],
            "observed_z": observed[2],
            "predicted_n": predicted[0],
            "predicted_e": predicted[1],
            "predicted_z": predicted[2],
            "residual_n": residual[0],
            "residual_e": residual[1],
            "residual_z": residual[2],
        }
    )
    wave_df.to_csv(MODEL_SOURCE_BASE.with_name(MODEL_SOURCE_BASE.name + "_waveforms.csv"), index=False)

    spec_df = pd.DataFrame(
        {
            "frequency_hz": np.asarray(model_sample["frequency_hz"], dtype=float),
            "jst_log10_residual_over_prediction": np.asarray(model_sample["hvsr"], dtype=float),
            "in_1_10_hz_measurement_band": BAND_1_10,
        }
    )
    spec_df.to_csv(MODEL_SOURCE_BASE.with_name(MODEL_SOURCE_BASE.name + "_spectrum.csv"), index=False)

    meta = pd.DataFrame(
        [
            {
                "station_id": STATION_ID,
                "event_id": EVENT_ID,
                "event_label": EVENT_LABEL,
                "profile_vs30_m_s": float(row["vs30"]),
                "nehrp": str(row["nehrp"]),
                "jst_1_10_hz_log10_score": band_score,
                "source_magnitude": float(row["source_magnitude"]),
                "source_depth_km": float(row["source_depth_km"]),
                "path_ep_distance_km": float(row["path_ep_distance_km"]),
                "waveform_source": str(MODEL_SAMPLE_NPZ),
            }
        ]
    )
    meta.to_csv(MODEL_SOURCE_BASE.with_name(MODEL_SOURCE_BASE.name + "_metadata.csv"), index=False)


def _draw_element(row: pd.Series, times: np.ndarray, waves: np.ndarray, hvsr: np.ndarray, band_score: float, clean: bool):
    """Draw either the annotated extraction element or a cleaner artwork element."""
    fig = plt.figure(figsize=(6.2, 2.35 if not clean else 2.05), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.55, 1.0],
        left=0.055,
        right=0.985,
        bottom=0.20 if not clean else 0.12,
        top=0.76 if not clean else 0.92,
        wspace=0.22,
    )
    ax_w = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[0, 1])

    if not clean:
        fig.text(
            0.055,
            0.955,
            "Real KiK-net earthquake record",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=SEMANTIC["black"],
        )
        fig.text(
            0.055,
            0.895,
            f"{STATION_CODE} | {EVENT_LABEL} | profile Vs30={float(row['vs30']):.0f} m s$^{{-1}}$",
            ha="left",
            va="top",
            fontsize=6.1,
            color="#4A4A4A",
        )
        fig.text(
            0.640,
            0.955,
            "JsT site-effect measurement",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=SEMANTIC["black"],
        )

    wave_color = "#2A2A2A"
    offsets = np.array([1.45, 0.0, -1.45])
    labels = ["N", "E", "Z"]
    for i, (offset, label) in enumerate(zip(offsets, labels)):
        ax_w.plot(times, waves[i] + offset, color=wave_color, lw=0.62, solid_capstyle="round")
        ax_w.text(-22.2, offset, label, ha="right", va="center", fontsize=6.7, color=SEMANTIC["black"])
    ax_w.axvline(0, color=SEMANTIC["jsT_blue"], lw=0.75)
    ax_w.text(0.7, 2.15, "P", ha="left", va="center", fontsize=6.2, color=SEMANTIC["jsT_blue"])
    ax_w.set_xlim(-20, 60)
    ax_w.set_ylim(-2.45, 2.45)
    ax_w.set_yticks([])
    ax_w.set_xticks([-20, 0, 20, 40, 60] if not clean else [])
    ax_w.tick_params(axis="x", length=2.2, width=0.45)
    ax_w.spines["bottom"].set_visible(not clean)
    if not clean:
        ax_w.set_xlabel("Time from P arrival (s)", labelpad=1.5)

    ax_s.set_xscale("log")
    ax_s.axhline(0, color="#B9B9B9", lw=0.5, zorder=0)
    ax_s.axvspan(1, 10, color=SEMANTIC["jsT_blue_light"], alpha=0.12, lw=0, zorder=0)
    ax_s.plot(FREQ_CENTERS, hvsr, color=SEMANTIC["jsT_blue"], lw=1.55, solid_capstyle="round")
    ax_s.fill_between(
        FREQ_CENTERS,
        0,
        np.maximum(hvsr, 0),
        where=BAND_1_10,
        color=SEMANTIC["jsT_blue_light"],
        alpha=0.26,
        interpolate=True,
    )
    ax_s.set_xlim(0.3, 15)
    ymin = min(-0.12, float(np.nanmin(hvsr)) - 0.08)
    ymax = max(1.08, float(np.nanmax(hvsr)) + 0.08)
    ax_s.set_ylim(ymin, ymax)
    ax_s.set_xticks([0.3, 1, 3, 10] if not clean else [])
    if not clean:
        ax_s.set_xticklabels(["0.3", "1", "3", "10"])
    ax_s.tick_params(axis="both", length=2.2, width=0.45)
    ax_s.spines["bottom"].set_visible(not clean)
    ax_s.spines["left"].set_visible(not clean)
    if clean:
        ax_s.set_yticks([])
    else:
        ax_s.set_xlabel("Frequency (Hz)", labelpad=1.5)
        ax_s.set_ylabel("JsT log10 residual / prediction", labelpad=2)
        ax_s.text(
            0.98,
            0.91,
            f"1-10 Hz mean\n+{band_score:.2f}",
            transform=ax_s.transAxes,
            ha="right",
            va="top",
            fontsize=6.6,
            fontweight="bold",
            color=SEMANTIC["jsT_blue"],
        )
        ax_s.text(
            0.98,
            0.08,
            f"NEHRP {row['nehrp']}",
            transform=ax_s.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.9,
            color="#4A4A4A",
        )
    return fig


def _normalize_model_waveforms(model_sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(model_sample["time_s"], dtype=float)
    observed = np.asarray(model_sample["observed"], dtype=float)
    predicted = np.asarray(model_sample["predicted"], dtype=float)
    residual = np.asarray(model_sample["residual"], dtype=float)
    scale = np.percentile(np.abs(observed), 99.5)
    if not np.isfinite(scale) or scale <= 0:
        scale = max(float(np.max(np.abs(observed))), 1e-12)
    return time, observed / scale, predicted / scale, residual / scale


def _draw_model_element(row: pd.Series, model_sample: dict[str, np.ndarray], band_score: float, clean: bool):
    time, observed, predicted, residual = _normalize_model_waveforms(model_sample)
    hvsr = np.asarray(model_sample["hvsr"], dtype=float)
    freq = np.asarray(model_sample["frequency_hz"], dtype=float)

    fig = plt.figure(figsize=(6.9, 2.65 if not clean else 2.28), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.05, 0.90],
        left=0.055,
        right=0.985,
        bottom=0.20 if not clean else 0.10,
        top=0.76 if not clean else 0.91,
        wspace=0.28,
    )
    ax_obs = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2])

    if not clean:
        fig.text(
            0.055,
            0.955,
            "Observed record and JsT prediction",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=SEMANTIC["black"],
        )
        fig.text(
            0.055,
            0.895,
            f"{STATION_CODE} | {EVENT_LABEL} | profile Vs30={float(row['vs30']):.0f} m s$^{{-1}}$",
            ha="left",
            va="top",
            fontsize=6.1,
            color="#4A4A4A",
        )
        fig.text(
            0.735,
            0.955,
            "JsT site-effect measurement",
            ha="left",
            va="top",
            fontsize=8.0,
            fontweight="bold",
            color=SEMANTIC["black"],
        )

    def draw_three(ax, waves, color, title):
        offsets = np.array([1.35, 0.0, -1.35])
        for idx, (offset, label) in enumerate(zip(offsets, ["N", "E", "Z"])):
            ax.plot(time, waves[idx] + offset, color=color, lw=0.58, solid_capstyle="round")
            ax.text(-22.0, offset, label, ha="right", va="center", fontsize=6.2, color=SEMANTIC["black"])
        ax.axvline(0, color=SEMANTIC["jsT_blue"], lw=0.65)
        ax.text(0.7, 2.00, "P", ha="left", va="center", fontsize=5.9, color=SEMANTIC["jsT_blue"])
        ax.set_xlim(-20, 60)
        ax.set_ylim(-2.25, 2.25)
        ax.set_yticks([])
        ax.set_xticks([-20, 0, 20, 40, 60] if not clean else [])
        ax.tick_params(axis="x", length=2.0, width=0.45)
        ax.spines["bottom"].set_visible(not clean)
        if not clean:
            ax.set_xlabel("Time from P arrival (s)", labelpad=1.2)
            ax.set_title(title, loc="left", fontsize=7.2, pad=3)

    draw_three(ax_obs, observed, "#2A2A2A", "Observed")
    draw_three(ax_pred, predicted, SEMANTIC["jsT_blue"], "JsT predicted")

    if clean:
        ax_obs.text(0.02, 0.96, "observed", transform=ax_obs.transAxes, ha="left", va="top", fontsize=6.2, color="#2A2A2A")
        ax_pred.text(0.02, 0.96, "predicted", transform=ax_pred.transAxes, ha="left", va="top", fontsize=6.2, color=SEMANTIC["jsT_blue"])

    ax_spec.set_xscale("log")
    ax_spec.axhline(0, color="#B9B9B9", lw=0.5, zorder=0)
    ax_spec.axvspan(1, 10, color=SEMANTIC["jsT_blue_light"], alpha=0.12, lw=0, zorder=0)
    ax_spec.plot(freq, hvsr, color=SEMANTIC["jsT_blue"], lw=1.55, solid_capstyle="round")
    ax_spec.fill_between(
        freq,
        0,
        np.maximum(hvsr, 0),
        where=BAND_1_10,
        color=SEMANTIC["jsT_blue_light"],
        alpha=0.26,
        interpolate=True,
    )
    ax_spec.set_xlim(0.3, 15)
    ymin = min(-0.12, float(np.nanmin(hvsr)) - 0.08)
    ymax = max(1.08, float(np.nanmax(hvsr)) + 0.08)
    ax_spec.set_ylim(ymin, ymax)
    ax_spec.set_xticks([0.3, 1, 3, 10] if not clean else [])
    if not clean:
        ax_spec.set_xticklabels(["0.3", "1", "3", "10"])
    ax_spec.tick_params(axis="both", length=2.0, width=0.45)
    ax_spec.spines["bottom"].set_visible(not clean)
    ax_spec.spines["left"].set_visible(not clean)
    if clean:
        ax_spec.set_yticks([])
    else:
        ax_spec.set_xlabel("Frequency (Hz)", labelpad=1.2)
        ax_spec.set_ylabel("log10 residual / prediction", labelpad=1.6)
        ax_spec.text(
            0.98,
            0.91,
            f"JsT score\n+{band_score:.2f}",
            transform=ax_spec.transAxes,
            ha="right",
            va="top",
            fontsize=6.3,
            fontweight="bold",
            color=SEMANTIC["jsT_blue"],
        )
        ax_spec.text(
            0.98,
            0.08,
            f"NEHRP {row['nehrp']}",
            transform=ax_spec.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color="#4A4A4A",
        )
    return fig


def _save_figure(fig, output_base: Path) -> None:
    for ext in [".pdf", ".svg", ".png"]:
        if ext == ".png":
            fig.savefig(output_base.with_suffix(ext), dpi=600, bbox_inches="tight")
        else:
            fig.savefig(output_base.with_suffix(ext), bbox_inches="tight")


def render() -> None:
    apply_nature_style(base_size=6.8)
    plt.rcParams.update(
        {
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )

    row = load_measurement_row()
    times, waves = load_waveform_window(row)
    hvsr = np.asarray([float(row[f"hvsr_bin_{i:02d}"]) for i in range(N_FREQ_BINS)], dtype=float)
    band_score = float(np.mean(hvsr[BAND_1_10]))
    save_source_data(times, waves, row, band_score)

    fig = _draw_element(row, times, waves, hvsr, band_score, clean=False)
    _save_figure(fig, OUT_BASE)
    plt.close(fig)

    fig = _draw_element(row, times, waves, hvsr, band_score, clean=True)
    _save_figure(fig, CLEAN_OUT_BASE)
    plt.close(fig)

    model_sample = load_model_sample()
    if model_sample is not None:
        model_hvsr = np.asarray(model_sample["hvsr"], dtype=float)
        model_band_score = float(np.mean(model_hvsr[BAND_1_10]))
        save_model_source_data(model_sample, row, model_band_score)
        fig = _draw_model_element(row, model_sample, model_band_score, clean=False)
        _save_figure(fig, MODEL_OUT_BASE)
        plt.close(fig)

        fig = _draw_model_element(row, model_sample, model_band_score, clean=True)
        _save_figure(fig, MODEL_CLEAN_OUT_BASE)
        plt.close(fig)


if __name__ == "__main__":
    render()
