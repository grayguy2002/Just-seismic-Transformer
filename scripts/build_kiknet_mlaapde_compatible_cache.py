#!/usr/bin/env python3
"""Build a JsT/MLAAPDE-compatible cache from downloaded KiK-net ASCII ZIPs.

Inputs:
- station manifest with profile-derived Vs30.
- NIED search index with event/source metadata.
- downloaded KiK-net event ZIP files from Download by HTTPS.

Outputs follow the existing JsT dataset contract:
- cache/kiknet_measured_vs30_pwave_v1_X_raw_120s_float32.npy
- cache/kiknet_measured_vs30_pwave_v1_X_model_20p60_streamnorm_float32.npy
- cache/kiknet_measured_vs30_pwave_v1_conditions.csv
- cache/splits/testing_indices.npy
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "kiknet_measured_vs30_pwave_v1"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "kiknet_measured_vs30_station_manifest.csv"
DEFAULT_INDEX = DEFAULT_DATA_DIR / "nied_search" / "nied_kiknet_record_candidates.csv"
DEFAULT_ZIP_DIR = DEFAULT_DATA_DIR / "raw_zips"
DEFAULT_TAG = "kiknet_measured_vs30_pwave_v1"

RAW_SAMPLES = 4800
MODEL_PRE_SEC = 20
MODEL_POST_SEC = 60
SAMPLE_RATE_HZ = 40
MODEL_SAMPLES = (MODEL_PRE_SEC + MODEL_POST_SEC) * SAMPLE_RATE_HZ
SURFACE_COMPONENTS = {"NS2": "N", "EW2": "E", "UD2": "Z"}


MODEL_CONDITION_COLUMNS_V21 = [
    "selected_phase",
    "phase_travel_sec",
    "selected_phase_arrival_sample",
    "selected_phase_status",
    "source_latitude_deg",
    "source_longitude_deg",
    "source_depth_km",
    "source_magnitude",
    "source_magnitude_type",
    "source_origin_uncertainty_sec",
    "source_latitude_uncertainty_km",
    "source_longitude_uncertainty_km",
    "source_depth_uncertainty_km",
    "source_magnitude_uncertainty",
    "source_magnitude_author",
    "path_ep_distance_deg",
    "path_ep_distance_km",
    "path_azimuth_deg",
    "path_back_azimuth_deg",
    "station_network_code",
    "station_code",
    "trace_channel",
    "station_location_code",
    "station_latitude_deg",
    "station_longitude_deg",
    "station_elevation_m",
    "station_local_depth_m",
    "channel_E_azimuth_deg",
    "channel_N_azimuth_deg",
    "channel_Z_azimuth_deg",
    "trace_P_arrival_sample",
    "trace_P_status",
    "trace_Pn_arrival_sample",
    "trace_Pn_status",
    "trace_Pg_arrival_sample",
    "trace_Pg_status",
    "trace_Sg_arrival_sample",
    "trace_Sg_status",
    "trace_Sn_arrival_sample",
    "trace_Sn_status",
]

CONDITION_COLUMNS = [
    "cache_index",
    "phase_id",
    "waves_id",
    "event_id",
    "selected_phase",
    "phase_time",
    "phase_travel_sec",
    "residual_travel_sec",
    "trace_snr_db",
    "month",
    "month_compact",
    "subset_split",
    "source_origin_time",
    "source_latitude_deg",
    "source_longitude_deg",
    "source_depth_km",
    "source_magnitude",
    "source_magnitude_type",
    "source_origin_uncertainty_sec",
    "source_latitude_uncertainty_km",
    "source_longitude_uncertainty_km",
    "source_depth_uncertainty_km",
    "source_magnitude_uncertainty",
    "source_magnitude_author",
    "path_ep_distance_deg",
    "path_ep_distance_km",
    "path_azimuth_deg",
    "path_back_azimuth_deg",
    "station_network_code",
    "station_code",
    "station_id",
    "trace_channel",
    "station_location_code",
    "station_latitude_deg",
    "station_longitude_deg",
    "station_elevation_m",
    "station_local_depth_m",
    "channel_E_azimuth_deg",
    "channel_N_azimuth_deg",
    "channel_Z_azimuth_deg",
    "trace_sampling_rate_hz",
    "trace_start_time",
    "selected_phase_arrival_sample",
    "selected_phase_status",
    "selected_phase_analyst_id",
    "trace_P_arrival_sample",
    "trace_P_status",
    "trace_Pn_arrival_sample",
    "trace_Pn_status",
    "trace_Pg_arrival_sample",
    "trace_Pg_status",
    "trace_Sg_arrival_sample",
    "trace_Sg_status",
    "trace_Sn_arrival_sample",
    "trace_Sn_status",
    "trace_name",
    "hdf5_bucket",
    "hdf5_index",
    "trace_n_channels",
    "trace_n_samples",
    "normalization_scale",
    "model_left_pad_samples",
    "model_right_pad_samples",
    "vs30_m_s",
    "nehrp_site_class",
    "vs_profile_depth_m",
]


@dataclass
class AsciiTrace:
    station_code: str
    component: str
    header: dict[str, str]
    samples: np.ndarray
    sampling_rate_hz: float
    scale_factor: float
    trace_start: pd.Timestamp


def parse_nied_time(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NaT
    return pd.to_datetime(text, utc=True, errors="coerce")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_scale_factor(text: str) -> float:
    match = re.search(r"([-+0-9.eE]+)\s*\([^)]*\)\s*/\s*([-+0-9.eE]+)", text)
    if not match:
        match = re.search(r"([-+0-9.eE]+)\s*/\s*([-+0-9.eE]+)", text)
    if not match:
        return 1.0
    numerator = safe_float(match.group(1), 1.0)
    denominator = safe_float(match.group(2), 1.0)
    return numerator / denominator if denominator else 1.0


def parse_sampling_rate(text: str) -> float:
    match = re.search(r"[-+0-9.]+", text)
    return safe_float(match.group(0), math.nan) if match else math.nan


def parse_ascii_trace(text: str) -> AsciiTrace:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header: dict[str, str] = {}
    data_start = None
    for i, line in enumerate(lines):
        if i >= 17:
            data_start = i
            break
        key = line[:18].strip()
        value = line[18:].strip()
        if key:
            header[key] = value
    if data_start is None:
        raise ValueError("K-NET ASCII file has too few lines")

    values: list[float] = []
    for line in lines[data_start:]:
        for token in line.split():
            try:
                values.append(float(token))
            except ValueError:
                continue
    samples = np.asarray(values, dtype=np.float32)
    station_code = header.get("Station Code", "").strip()
    direction = header.get("Dir.", "").strip()
    component = direction.replace("-", "").replace(" ", "").upper()
    sampling_rate = parse_sampling_rate(header.get("Sampling Freq(Hz)", ""))
    scale = parse_scale_factor(header.get("Scale Factor", ""))
    record_time = parse_nied_time(header.get("Record Time", ""))
    trace_start = record_time - pd.to_timedelta(15, unit="s") if not pd.isna(record_time) else pd.NaT
    return AsciiTrace(
        station_code=station_code,
        component=component,
        header=header,
        samples=samples * np.float32(scale),
        sampling_rate_hz=sampling_rate,
        scale_factor=scale,
        trace_start=trace_start,
    )


def component_from_name(path: str) -> str:
    suffix = Path(path).suffix.upper().replace(".", "")
    if suffix in SURFACE_COMPONENTS:
        return suffix
    match = re.search(r"(NS2|EW2|UD2)$", path.upper())
    return match.group(1) if match else ""


def station_from_name(path: str) -> str:
    name = Path(path).name
    match = re.match(r"([A-Z0-9]{6})", name.upper())
    return match.group(1) if match else ""


def load_event_zip(zip_path: Path, allowed_stations: set[str]) -> dict[str, dict[str, AsciiTrace]]:
    traces: dict[str, dict[str, AsciiTrace]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            comp = component_from_name(member)
            if comp not in SURFACE_COMPONENTS:
                continue
            station = station_from_name(member)
            if station not in allowed_stations:
                continue
            raw = zf.read(member)
            text = raw.decode("shift_jis", errors="replace")
            trace = parse_ascii_trace(text)
            station_code = trace.station_code or station
            traces.setdefault(station_code, {})[comp] = trace
    return traces


def resample_linear(samples: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    if not math.isfinite(src_rate) or src_rate <= 0:
        raise ValueError(f"Invalid sampling rate: {src_rate}")
    if abs(src_rate - dst_rate) < 1e-6:
        return samples.astype(np.float32, copy=False)
    duration = len(samples) / src_rate
    n_out = int(math.floor(duration * dst_rate))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    src_t = np.arange(len(samples), dtype=np.float64) / src_rate
    dst_t = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(dst_t, src_t, samples).astype(np.float32)


def fixed_crop(wave: np.ndarray, start: int, n_samples: int) -> tuple[np.ndarray, int, int]:
    out = np.zeros((wave.shape[0], n_samples), dtype=np.float32)
    src_start = max(start, 0)
    src_end = min(start + n_samples, wave.shape[1])
    dst_start = max(-start, 0)
    dst_end = dst_start + max(src_end - src_start, 0)
    if src_end > src_start:
        out[:, dst_start:dst_end] = wave[:, src_start:src_end]
    return out, max(-start, 0), max(start + n_samples - wave.shape[1], 0)


def haversine_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float, float]:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    distance_km = 6371.0088 * c
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    az = (math.degrees(math.atan2(y, x)) + 360) % 360
    baz = (az + 180) % 360
    return distance_km / 111.195, distance_km, az, baz


_TAUP_MODEL = None


def _get_taup_model():
    global _TAUP_MODEL
    if _TAUP_MODEL is None:
        from obspy.taup import TauPyModel
        _TAUP_MODEL = TauPyModel(model="iasp91")
    return _TAUP_MODEL


def taup_p_arrival_sec(source_depth_km: float, distance_deg: float) -> float:
    """First-arriving P-wave travel time via TauP (iasp91).

    Falls back to a simple velocity model if TauP is unavailable.
    """
    depth = 0.0 if not math.isfinite(source_depth_km) else max(source_depth_km, 0.0)
    dist_deg = max(distance_deg, 0.01)
    try:
        m = _get_taup_model()
        arrivals = m.get_travel_times(
            source_depth_in_km=depth, distance_in_degree=dist_deg,
            phase_list=["P", "Pn", "Pg", "pP"],
        )
        if arrivals:
            return float(arrivals[0].time)
    except Exception:
        pass
    dist_km = dist_deg * 111.195
    return math.sqrt(dist_km**2 + depth**2) / 8.0


def estimate_snr_db(wave: np.ndarray, arrival_sample: int) -> float:
    pre_end = max(arrival_sample - 80, 1)
    sig_start = min(max(arrival_sample, 0), wave.shape[1] - 1)
    sig_end = min(sig_start + 400, wave.shape[1])
    noise = wave[:, :pre_end]
    signal = wave[:, sig_start:sig_end]
    if noise.size == 0 or signal.size == 0:
        return math.nan
    noise_rms = float(np.sqrt(np.mean(noise**2))) + 1e-12
    sig_rms = float(np.sqrt(np.mean(signal**2))) + 1e-12
    return 20.0 * math.log10(sig_rms / noise_rms)


def build_row(
    record: pd.Series,
    station: pd.Series,
    wave: np.ndarray,
    trace_start: pd.Timestamp,
    zip_path: Path,
    cache_index: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    origin = parse_nied_time(record["source_origin_time"])
    if pd.isna(origin) or pd.isna(trace_start):
        raise ValueError("Missing origin or trace start time")
    src_lat = safe_float(record["source_latitude_deg"])
    src_lon = safe_float(record["source_longitude_deg"])
    sta_lat = safe_float(station["station_latitude_deg"])
    sta_lon = safe_float(station["station_longitude_deg"])
    depth = safe_float(record["source_depth_km"], 0.0)
    dist_deg, dist_km, az, baz = haversine_azimuth(src_lat, src_lon, sta_lat, sta_lon)
    if math.isfinite(safe_float(record.get("path_ep_distance_km"))):
        dist_km = safe_float(record.get("path_ep_distance_km"))
        dist_deg = dist_km / 111.195
    p_travel = taup_p_arrival_sec(depth, dist_deg)
    if not math.isfinite(p_travel) or p_travel <= 0:
        # Fallback: simple straight-line at 8 km/s
        dist_km_fb = dist_deg * 111.195 if math.isfinite(dist_deg) else 100.0
        p_travel = math.sqrt(dist_km_fb**2 + max(depth, 0)**2) / 8.0
    if not math.isfinite(p_travel):
        raise ValueError(f"Uncomputable P travel: depth={depth} dist_deg={dist_deg}")
    p_time = origin + pd.to_timedelta(p_travel, unit="s")
    arrival_sample = int(round((p_time - trace_start).total_seconds() * SAMPLE_RATE_HZ))

    raw_wave, raw_left, raw_right = fixed_crop(wave, 0, RAW_SAMPLES)
    if raw_left or raw_right:
        raise ValueError("Raw 120 s crop requires padding; skip short record")
    model_start = arrival_sample - MODEL_PRE_SEC * SAMPLE_RATE_HZ
    model_wave, left_pad, right_pad = fixed_crop(wave, model_start, MODEL_SAMPLES)
    scale = float(np.max(np.abs(model_wave)))
    if scale > 0:
        model_wave = model_wave / scale
    else:
        scale = 1.0
    event_id = str(record["eqid"])
    station_code = str(station["station_code"])
    phase_id = f"{event_id}_KIKNET.{station_code}.HN.--_P"
    row = {
        "cache_index": cache_index,
        "phase_id": phase_id,
        "waves_id": f"{event_id}_KIKNET.{station_code}.HN.--",
        "event_id": event_id,
        "selected_phase": "P",
        "phase_time": p_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "phase_travel_sec": float(p_travel),
        "residual_travel_sec": 0.0,  # TauP theoretical P — no observed residual
        "station_id": f"KIKNET.{station_code}",
        "trace_snr_db": estimate_snr_db(wave, arrival_sample),
        "month": f"{event_id[:4]}-{event_id[4:6]}",
        "month_compact": event_id[:6],
        "subset_split": "testing",
        "source_origin_time": origin.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "source_latitude_deg": src_lat,
        "source_longitude_deg": src_lon,
        "source_depth_km": depth,
        "source_magnitude": safe_float(record["source_magnitude"]),
        "source_magnitude_type": "M",
        "source_origin_uncertainty_sec": np.nan,
        "source_latitude_uncertainty_km": np.nan,
        "source_longitude_uncertainty_km": np.nan,
        "source_depth_uncertainty_km": np.nan,
        "source_magnitude_uncertainty": np.nan,
        "source_magnitude_author": "NIED",
        "path_ep_distance_deg": dist_deg,
        "path_ep_distance_km": dist_km,
        "path_azimuth_deg": az,
        "path_back_azimuth_deg": baz,
        "station_network_code": "KIKNET",
        "station_code": station_code,
        "trace_channel": "HN",
        "station_location_code": "--",
        "station_latitude_deg": sta_lat,
        "station_longitude_deg": sta_lon,
        "station_elevation_m": safe_float(station.get("station_elevation_m"), 0.0),
        "station_local_depth_m": safe_float(station.get("station_local_depth_m"), 0.0),
        "channel_E_azimuth_deg": 90.0,
        "channel_N_azimuth_deg": 0.0,
        "channel_Z_azimuth_deg": -90.0,
        "trace_sampling_rate_hz": SAMPLE_RATE_HZ,
        "trace_start_time": trace_start.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "selected_phase_arrival_sample": arrival_sample,
        "selected_phase_status": "estimated_theoretical",
        "selected_phase_analyst_id": "NIED_TauP_iasp91",
        "trace_P_arrival_sample": arrival_sample,
        "trace_P_status": "estimated_theoretical",
        "trace_Pn_arrival_sample": np.nan,
        "trace_Pn_status": "PHASE_MISSING",
        "trace_Pg_arrival_sample": np.nan,
        "trace_Pg_status": "PHASE_MISSING",
        "trace_Sg_arrival_sample": np.nan,
        "trace_Sg_status": "PHASE_MISSING",
        "trace_Sn_arrival_sample": np.nan,
        "trace_Sn_status": "PHASE_MISSING",
        "trace_name": str(zip_path),
        "hdf5_bucket": "",
        "hdf5_index": cache_index,
        "trace_n_channels": 3,
        "trace_n_samples": wave.shape[1],
        "normalization_scale": scale,
        "model_left_pad_samples": left_pad,
        "model_right_pad_samples": right_pad,
        "vs30_m_s": safe_float(station["vs30_m_s"]),
        "nehrp_site_class": station["nehrp_site_class"],
        "vs_profile_depth_m": safe_float(station["vs_profile_depth_m"]),
    }
    return row, raw_wave, model_wave


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--zip-dir", default=str(DEFAULT_ZIP_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dataset-tag", default=DEFAULT_TAG)
    parser.add_argument("--p-velocity-km-s", type=float, default=6.0, help="deprecated; TauP fallback only")
    parser.add_argument("--min-snr-db", type=float, default=-999.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "cache"
    split_dir = cache_dir / "splits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    raw_path = cache_dir / f"{args.dataset_tag}_X_raw_120s_float32.npy"
    model_path = cache_dir / f"{args.dataset_tag}_X_model_20p60_streamnorm_float32.npy"
    conditions_path = cache_dir / f"{args.dataset_tag}_conditions.csv"
    summary_path = cache_dir / f"{args.dataset_tag}_cache_summary.json"
    existing = [p for p in [raw_path, model_path, conditions_path, summary_path] if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Refusing to overwrite without --overwrite: " + ", ".join(str(p) for p in existing))

    manifest = pd.read_csv(args.manifest)
    manifest_by_code = {str(r.station_code): pd.Series(r._asdict()) for r in manifest.itertuples(index=False)}
    allowed_stations = set(manifest_by_code)
    index = pd.read_csv(args.index)
    index_by_key = {
        (str(r.eqid), str(r.station_code)): pd.Series(r._asdict())
        for r in index.itertuples(index=False)
    }

    rows = []
    raw_waves = []
    model_waves = []
    skipped = []
    for zip_path in sorted(Path(args.zip_dir).glob("*_ascii.zip")):
        if zip_path.name.startswith("._"):
            continue
        eqid = zip_path.name.replace("_ascii.zip", "")
        try:
            event_traces = load_event_zip(zip_path, allowed_stations)
        except zipfile.BadZipFile:
            skipped.append({"zip": str(zip_path), "station_code": "", "reason": "bad_zip_file"})
            continue
        for station_code, comps in event_traces.items():
            if not all(comp in comps for comp in SURFACE_COMPONENTS):
                skipped.append({"zip": str(zip_path), "station_code": station_code, "reason": "missing_surface_components"})
                continue
            record = index_by_key.get((eqid, station_code))
            station = manifest_by_code.get(station_code)
            if record is None or station is None:
                continue
            traces = [comps["NS2"], comps["EW2"], comps["UD2"]]
            try:
                resampled = [resample_linear(t.samples, t.sampling_rate_hz, SAMPLE_RATE_HZ) for t in traces]
                n = min(len(x) for x in resampled)
                wave = np.stack([resampled[0][:n], resampled[1][:n], resampled[2][:n]]).astype(np.float32)
                trace_start = traces[0].trace_start
                row, raw_wave, model_wave = build_row(
                    record,
                    station,
                    wave,
                    trace_start,
                    zip_path,
                    len(rows),
                )
                if math.isfinite(row["trace_snr_db"]) and row["trace_snr_db"] < args.min_snr_db:
                    skipped.append({"zip": str(zip_path), "station_code": station_code, "reason": "low_snr"})
                    continue
            except Exception as exc:
                skipped.append({"zip": str(zip_path), "station_code": station_code, "reason": repr(exc)})
                continue
            rows.append(row)
            raw_waves.append(raw_wave)
            model_waves.append(model_wave)

    conditions = pd.DataFrame(rows)
    if conditions.empty:
        raise ValueError(
            "No cache rows built. Download NIED KiK-net ASCII ZIP files into raw_zips/ first, "
            "or inspect skipped records."
        )
    for col in CONDITION_COLUMNS:
        if col not in conditions.columns:
            conditions[col] = np.nan
    conditions = conditions[CONDITION_COLUMNS]

    np.save(raw_path, np.stack(raw_waves).astype(np.float32))
    np.save(model_path, np.stack(model_waves).astype(np.float32))
    conditions.to_csv(conditions_path, index=False)
    np.save(split_dir / "testing_indices.npy", np.arange(len(conditions), dtype=np.int64))

    skipped_path = cache_dir / f"{args.dataset_tag}_skipped_records.csv"
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    summary = {
        "dataset_tag": args.dataset_tag,
        "n_samples": int(len(conditions)),
        "station_count": int(conditions["station_code"].nunique()),
        "event_count": int(conditions["event_id"].nunique()),
        "condition_schema_version": "v2.1",
        "condition_columns": CONDITION_COLUMNS,
        "model_condition_columns": MODEL_CONDITION_COLUMNS_V21,
        "raw_cache": {
            "path": str(raw_path),
            "shape": list(np.load(raw_path, mmap_mode="r").shape),
            "dtype": "float32",
            "description": "First 120 seconds / 4800 samples from each KiK-net surface trace, acceleration in gal.",
        },
        "model_cache": {
            "path": str(model_path),
            "shape": list(np.load(model_path, mmap_mode="r").shape),
            "dtype": "float32",
            "crop": {
                "pre_sec": MODEL_PRE_SEC,
                "post_sec": MODEL_POST_SEC,
                "sample_rate_hz": SAMPLE_RATE_HZ,
            },
            "normalization": "stream max-abs per sample over all channels and cropped samples",
        },
        "conditions": str(conditions_path),
        "splits": {"testing": str(split_dir / "testing_indices.npy")},
        "source_index": str(args.index),
        "manifest": str(args.manifest),
        "zip_dir": str(args.zip_dir),
        "arrival_note": (
            "P arrival samples are estimated with a simple straight-line velocity model. "
            "Replace with TauP/JMA picks before final analysis."
        ),
        "skipped_records": str(skipped_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
