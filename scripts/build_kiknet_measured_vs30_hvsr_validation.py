#!/usr/bin/env python3
"""Build a measured/profile Vs30 + earthquake HVSR validation table for KiK-net.

The source HVSR curves are the GFZ open site database earthquake HVSR products.
This script joins those station-response curves to the KiK-net station manifest
whose Vs30 values are derived from borehole/profile VSz fields, not proxy maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATION_MANIFEST = (
    ROOT
    / "data"
    / "kiknet_measured_vs30_pwave_v1"
    / "kiknet_measured_vs30_station_manifest.csv"
)
DEFAULT_HVSR_ZIP = (
    ROOT
    / "data"
    / "measured_vs30_validation"
    / "external"
    / "japan_knet_kiknet"
    / "unzipped"
    / "Earthquake HVSRs at K-NET and KiK-net Strong-Motion Stations.zip"
)
DEFAULT_OUT_DIR = ROOT / "data" / "kiknet_measured_vs30_pwave_v1" / "hvsr_validation"

HVSR_RE = re.compile(r"HVSR_D_KiK_([A-Z0-9]+)\.txt$")


@dataclass
class PeakResult:
    significant_peak_count: int
    first_peak_frequency_hz: float | None
    first_peak_amplitude: float | None
    first_peak_width_log10_hz: float | None
    first_peak_prominence_log10: float | None
    predominant_peak_frequency_hz: float | None
    predominant_peak_amplitude: float | None
    predominant_peak_width_log10_hz: float | None
    predominant_peak_prominence_log10: float | None
    significance_amplitude: float
    geometric_mean_amplitude: float
    peak_pick_frequency_min_hz: float
    peak_pick_frequency_max_hz: float
    peak_pick_frequency_count: int


def _finite_or_none(value: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _read_hvsr_member(zf: zipfile.ZipFile, member: str) -> tuple[np.ndarray, np.ndarray]:
    with zf.open(member) as fh:
        data = np.loadtxt(fh, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected HVSR curve shape for {member}: {data.shape}")
    freq = data[:, 0]
    amp = data[:, 1]
    valid = np.isfinite(freq) & np.isfinite(amp) & (freq > 0) & (amp > 0)
    return freq[valid], amp[valid]


def _pick_hvsr_peaks(freq: np.ndarray, amp: np.ndarray) -> PeakResult:
    """Replicate the GFZ MATLAB peak-screening logic on log10 HVSR curves."""
    # GFZ's published MATLAB script restricts x/y to the first 571 samples,
    # covering the 0.1-30 Hz frequency range in the released curve files.
    freq = freq[:571]
    amp = amp[:571]
    log_freq = np.log10(freq)
    log_amp = np.log10(amp)

    c1 = 2.18
    c2 = 1.47
    c3 = 1.8
    c4 = 0.5

    geometric_mean = float(np.exp(np.mean(np.log(amp))))
    significance = max(c1, c2 * geometric_mean)
    peak_indices, properties = find_peaks(
        log_amp,
        height=math.log10(significance),
        prominence=math.log10(c3),
    )

    if peak_indices.size == 0:
        return PeakResult(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            significance,
            geometric_mean,
            float(freq.min()),
            float(freq.max()),
            int(freq.size),
        )

    widths = peak_widths(log_amp, peak_indices, rel_height=0.5)[0]
    # Convert sample widths to log10-frequency widths by interpolating the peak
    # width's left/right intersection positions onto the log-frequency axis.
    left_ips = properties.get("left_bases", np.zeros_like(peak_indices)).astype(float)
    right_ips = properties.get("right_bases", np.zeros_like(peak_indices)).astype(float)
    width_log10 = np.empty_like(widths, dtype=float)
    x_index = np.arange(len(log_freq), dtype=float)
    for i, peak_index in enumerate(peak_indices):
        width_info = peak_widths(log_amp, [peak_index], rel_height=0.5)
        left = np.interp(width_info[2][0], x_index, log_freq)
        right = np.interp(width_info[3][0], x_index, log_freq)
        width_log10[i] = right - left

    prominence_log10 = properties["prominences"]
    keep = (prominence_log10 / width_log10) > c4
    peak_indices = peak_indices[keep]
    width_log10 = width_log10[keep]
    prominence_log10 = prominence_log10[keep]

    if peak_indices.size == 0:
        return PeakResult(
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            significance,
            geometric_mean,
            float(freq.min()),
            float(freq.max()),
            int(freq.size),
        )

    first = 0
    predominant = int(np.argmax(log_amp[peak_indices]))

    def values(index: int) -> tuple[float, float, float, float]:
        peak_index = peak_indices[index]
        return (
            float(freq[peak_index]),
            float(amp[peak_index]),
            float(width_log10[index]),
            float(prominence_log10[index]),
        )

    f0, a0, w0, p0 = values(first)
    fp, ap, wp, pp = values(predominant)
    return PeakResult(
        int(peak_indices.size),
        f0,
        a0,
        w0,
        p0,
        fp,
        ap,
        wp,
        pp,
        significance,
        geometric_mean,
        float(freq.min()),
        float(freq.max()),
        int(freq.size),
    )


def build(args: argparse.Namespace) -> dict:
    station_manifest_path = Path(args.station_manifest)
    hvsr_zip_path = Path(args.hvsr_zip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_manifest = pd.read_csv(station_manifest_path)
    required_station_cols = {
        "station_id",
        "station_code",
        "station_latitude_deg",
        "station_longitude_deg",
        "vs30_m_s",
        "nehrp_site_class",
        "vs_profile_depth_m",
        "measurement_quality",
        "vs30_source_type",
        "use_as_ground_truth",
        "reference",
        "url",
    }
    missing = sorted(required_station_cols.difference(station_manifest.columns))
    if missing:
        raise ValueError(f"Missing required columns in {station_manifest_path}: {missing}")

    hvsr_rows: list[dict] = []
    spectra_rows: list[dict] = []
    with zipfile.ZipFile(hvsr_zip_path) as zf:
        members = sorted(
            name
            for name in zf.namelist()
            if name.endswith(".txt") and HVSR_RE.search(Path(name).name)
        )
        for member in members:
            match = HVSR_RE.search(Path(member).name)
            if not match:
                continue
            station_code = match.group(1)
            freq, amp = _read_hvsr_member(zf, member)
            peak = _pick_hvsr_peaks(freq, amp)
            hvsr_rows.append(
                {
                    "station_code": station_code,
                    "hvsr_member": member,
                    "hvsr_frequency_min_hz": float(freq.min()),
                    "hvsr_frequency_max_hz": float(freq.max()),
                    "hvsr_frequency_count": int(freq.size),
                    "hvsr_amplitude_min": float(amp.min()),
                    "hvsr_amplitude_max": float(amp.max()),
                    "hvsr_geometric_mean_amplitude": peak.geometric_mean_amplitude,
                    "hvsr_significance_amplitude": peak.significance_amplitude,
                    "hvsr_peak_pick_frequency_min_hz": peak.peak_pick_frequency_min_hz,
                    "hvsr_peak_pick_frequency_max_hz": peak.peak_pick_frequency_max_hz,
                    "hvsr_peak_pick_frequency_count": peak.peak_pick_frequency_count,
                    "hvsr_significant_peak_count": peak.significant_peak_count,
                    "hvsr_first_peak_frequency_hz": _finite_or_none(peak.first_peak_frequency_hz),
                    "hvsr_first_peak_amplitude": _finite_or_none(peak.first_peak_amplitude),
                    "hvsr_first_peak_width_log10_hz": _finite_or_none(peak.first_peak_width_log10_hz),
                    "hvsr_first_peak_prominence_log10": _finite_or_none(peak.first_peak_prominence_log10),
                    "hvsr_predominant_peak_frequency_hz": _finite_or_none(peak.predominant_peak_frequency_hz),
                    "hvsr_predominant_peak_amplitude": _finite_or_none(peak.predominant_peak_amplitude),
                    "hvsr_predominant_peak_width_log10_hz": _finite_or_none(peak.predominant_peak_width_log10_hz),
                    "hvsr_predominant_peak_prominence_log10": _finite_or_none(peak.predominant_peak_prominence_log10),
                    "hvsr_source": "GFZ earthquake HVSR geomean curve",
                    "hvsr_reference": (
                        "Zhu, C., Weatherill, G., Cotton, F., Pilz, M., Kwak, D. Y., "
                        "and Kawase, H. (2020), An Open-Source Site Database of "
                        "Strong-Motion Stations in Japan: K-NET and KiK-net, "
                        "GFZ Data Services, doi:10.5880/GFZ.2.1.2020.006"
                    ),
                    "hvsr_url": "https://doi.org/10.5880/GFZ.2.1.2020.006",
                }
            )
            for f, a in zip(freq, amp):
                spectra_rows.append({"station_code": station_code, "frequency_hz": float(f), "hvsr_amplitude": float(a)})

    hvsr = pd.DataFrame(hvsr_rows).sort_values("station_code")
    spectra = pd.DataFrame(spectra_rows).sort_values(["station_code", "frequency_hz"])

    joined = station_manifest.merge(hvsr, how="left", on="station_code", validate="one_to_one")
    joined["validation_role"] = np.where(
        joined["hvsr_member"].notna(),
        "measured_vs30_with_public_earthquake_hvsr",
        "measured_vs30_without_public_hvsr_curve",
    )
    joined["can_test_jst_waveform_generation"] = False
    joined["waveform_data_status"] = (
        "No raw waveform included. NIED K-NET/KiK-net waveform download requires user registration; "
        "this table uses public GFZ earthquake-HVSR products derived from seismic records."
    )

    hvsr_path = out_dir / "kiknet_public_earthquake_hvsr_station_peaks.csv"
    spectra_path = out_dir / "kiknet_public_earthquake_hvsr_spectra_long.csv"
    joined_path = out_dir / "kiknet_measured_vs30_hvsr_validation_manifest.csv"
    unmatched_path = out_dir / "kiknet_measured_vs30_without_hvsr.csv"
    summary_path = out_dir / "kiknet_measured_vs30_hvsr_validation_summary.json"

    hvsr.to_csv(hvsr_path, index=False)
    spectra.to_csv(spectra_path, index=False, quoting=csv.QUOTE_MINIMAL)
    joined.to_csv(joined_path, index=False)
    joined[joined["hvsr_member"].isna()].to_csv(unmatched_path, index=False)

    compression = dict(method="zip", archive_name=spectra_path.name)
    spectra.to_csv(out_dir / "kiknet_public_earthquake_hvsr_spectra_long.csv.zip", index=False, compression=compression)

    matched = joined[joined["hvsr_member"].notna()].copy()
    summary = {
        "station_manifest": str(station_manifest_path),
        "hvsr_zip": str(hvsr_zip_path),
        "outputs": {
            "validation_manifest": str(joined_path),
            "hvsr_station_peaks": str(hvsr_path),
            "hvsr_spectra_long": str(spectra_path),
            "hvsr_spectra_long_zip": str(out_dir / "kiknet_public_earthquake_hvsr_spectra_long.csv.zip"),
            "unmatched_measured_vs30_stations": str(unmatched_path),
        },
        "station_rows": int(len(station_manifest)),
        "hvsr_curve_rows_total_in_zip": int(len(hvsr)),
        "profile_measured_vs30_stations_with_hvsr": int(matched["hvsr_member"].notna().sum()),
        "profile_measured_vs30_stations_without_hvsr": int(joined["hvsr_member"].isna().sum()),
        "spectra_rows": int(len(spectra)),
        "vs30_range_m_s_for_matched": [
            float(matched["vs30_m_s"].min()) if not matched.empty else None,
            float(matched["vs30_m_s"].max()) if not matched.empty else None,
        ],
        "nehrp_counts_for_matched": matched["nehrp_site_class"].value_counts().sort_index().to_dict(),
        "hvsr_peak_count_distribution_for_matched": matched["hvsr_significant_peak_count"].value_counts(dropna=False).sort_index().to_dict(),
        "hvsr_peak_picking": {
            "source_logic": "Replicates GFZ f0_A0_fp_Ap_Pickup.m thresholds using scipy.signal.find_peaks on log10 amplitude and log10 frequency.",
            "c1_min_significance": 2.18,
            "c2_geomean_multiplier": 1.47,
            "c3_min_prominence": 1.8,
            "c4_min_prominence_width_ratio": 0.5,
            "frequency_range_hz": [0.1, 30.0],
        },
        "limitations": [
            "This is not a raw-waveform JsT test cache.",
            "NIED raw waveform ZIP downloads require user registration/authentication.",
            "GFZ HVSR curves are processed station-response products derived from earthquake recordings.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-manifest", default=str(DEFAULT_STATION_MANIFEST))
    parser.add_argument("--hvsr-zip", default=str(DEFAULT_HVSR_ZIP))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
