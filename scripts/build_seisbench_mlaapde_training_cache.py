#!/usr/bin/env python3
"""Build fixed-shape waveform caches from the filtered MLAAPDE P-wave index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RAW_SAMPLES = 4800
MODEL_PRE_SEC = 20
MODEL_POST_SEC = 60
SAMPLE_RATE_HZ = 40
MODEL_SAMPLES = (MODEL_PRE_SEC + MODEL_POST_SEC) * SAMPLE_RATE_HZ


LEGACY_DATASET_TAG = "pwave_v1"

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

AUDIT_ONLY_COLUMNS_V21 = [
    "selected_phase_analyst_id",
]

FORBIDDEN_DEFAULT_CONDITION_COLUMNS_V21 = [
    "normalization_scale",
    "trace_snr_db",
    "event_id",
    "source_id",
    "trace_name",
    "hdf5_bucket",
    "hdf5_index",
    "cache_index",
    "waves_id",
    "phase_id",
    "source_origin_time",
    "trace_start_time",
    "selected_phase_analyst_id",
    "split",
    "subset_split",
    "month",
    "month_compact",
]

CONDITION_COLUMNS = [
    "cache_index",
    "phase_id",
    "waves_id",
    "event_id",
    "selected_phase",
    "phase_time",
    "phase_travel_sec",
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
    "trace_sampling_rate_hz",
    "trace_start_time",
    "selected_phase_arrival_sample",
    "selected_phase_status",
    "selected_phase_analyst_id",
    "trace_name",
    "hdf5_bucket",
    "hdf5_index",
    "trace_n_channels",
    "trace_n_samples",
    "normalization_scale",
]


def condition_columns_for_schema(condition_schema: str) -> list[str]:
    if condition_schema == "v2.1":
        columns = list(CONDITION_COLUMNS)
        for col in MODEL_CONDITION_COLUMNS_V21 + AUDIT_ONLY_COLUMNS_V21:
            if col not in columns:
                columns.append(col)
        return columns
    return list(CONDITION_COLUMNS)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = np.nan


def refuse_overwrite(paths: list[Path], overwrite: bool) -> None:
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --overwrite: "
            + ", ".join(existing)
        )


def fixed_crop(
    wave: np.ndarray,
    start: int,
    n_samples: int,
    *,
    pad_value: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """Return a fixed sample crop and left/right pad counts."""
    channels = wave.shape[0]
    out = np.full((channels, n_samples), pad_value, dtype=np.float32)

    src_start = max(start, 0)
    src_end = min(start + n_samples, wave.shape[1])
    dst_start = max(-start, 0)
    dst_end = dst_start + max(src_end - src_start, 0)

    if src_end > src_start:
        out[:, dst_start:dst_end] = wave[:, src_start:src_end].astype(np.float32, copy=False)

    left_pad = max(-start, 0)
    right_pad = max(start + n_samples - wave.shape[1], 0)
    return out, left_pad, right_pad


def open_hdf5(raw_dir: Path, handles: dict[str, h5py.File], month_compact: str) -> h5py.File:
    if month_compact not in handles:
        handles[month_compact] = h5py.File(raw_dir / f"waveforms_{month_compact}.hdf5", "r")
    return handles[month_compact]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default="data/seisbench_mlaapde_pwave_v1/filtered/mlaapde_pwave_v1_balanced_index.csv",
    )
    parser.add_argument("--raw-dir", default="data/seisbench_mlaapde_pwave_v1/raw")
    parser.add_argument("--out-dir", default="data/seisbench_mlaapde_pwave_v1/cache")
    parser.add_argument("--dataset-tag", default=LEGACY_DATASET_TAG)
    parser.add_argument("--condition-schema", choices=["legacy", "v2.1"], default="legacy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    index_path = Path(args.index)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(index_path)
    index = index.reset_index(drop=True)
    index["cache_index"] = np.arange(len(index), dtype=np.int64)

    raw_path = out_dir / f"{args.dataset_tag}_X_raw_120s_float32.npy"
    model_path = out_dir / f"{args.dataset_tag}_X_model_20p60_streamnorm_float32.npy"
    conditions_path = out_dir / f"{args.dataset_tag}_conditions.csv"
    summary_path = out_dir / f"{args.dataset_tag}_cache_summary.json"
    refuse_overwrite([raw_path, model_path, conditions_path, summary_path], args.overwrite)

    raw_cache = np.lib.format.open_memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(index), 3, RAW_SAMPLES),
    )
    model_cache = np.lib.format.open_memmap(
        model_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(index), 3, MODEL_SAMPLES),
    )

    handles: dict[str, h5py.File] = {}
    left_pads = []
    right_pads = []
    normalization_scales = []

    try:
        for row in index.itertuples(index=False):
            h5 = open_hdf5(raw_dir, handles, str(row.month_compact))
            wave = h5["data"][row.hdf5_bucket][int(row.hdf5_index)]

            raw_wave, raw_left_pad, raw_right_pad = fixed_crop(wave, 0, RAW_SAMPLES)
            if raw_left_pad or raw_right_pad:
                raise ValueError(f"Unexpected raw padding for {row.phase_id}")
            raw_cache[int(row.cache_index)] = raw_wave

            arrival = int(round(float(row.selected_phase_arrival_sample)))
            model_start = arrival - MODEL_PRE_SEC * SAMPLE_RATE_HZ
            model_wave, left_pad, right_pad = fixed_crop(wave, model_start, MODEL_SAMPLES)
            scale = float(np.max(np.abs(model_wave)))
            if scale > 0:
                model_wave = model_wave / scale
            else:
                scale = 1.0
            model_cache[int(row.cache_index)] = model_wave

            left_pads.append(left_pad)
            right_pads.append(right_pad)
            normalization_scales.append(scale)

            if (int(row.cache_index) + 1) % 1000 == 0:
                print(f"cached {int(row.cache_index) + 1:,}/{len(index):,}")
    finally:
        for handle in handles.values():
            handle.close()

    raw_cache.flush()
    model_cache.flush()

    index["normalization_scale"] = normalization_scales
    index["model_left_pad_samples"] = left_pads
    index["model_right_pad_samples"] = right_pads
    condition_columns = condition_columns_for_schema(args.condition_schema)
    ensure_columns(index, condition_columns)
    index[condition_columns].to_csv(conditions_path, index=False)

    split_dir = out_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    split_paths = {}
    for split_name, split_df in index.groupby("subset_split"):
        path = split_dir / f"{split_name}_indices.npy"
        np.save(path, split_df["cache_index"].to_numpy(dtype=np.int64))
        split_paths[str(split_name)] = str(path)

    summary = {
        "n_samples": int(len(index)),
        "dataset_tag": args.dataset_tag,
        "condition_schema_version": args.condition_schema,
        "condition_columns": condition_columns,
        "model_condition_columns": MODEL_CONDITION_COLUMNS_V21 if args.condition_schema == "v2.1" else condition_columns,
        "audit_only_columns": AUDIT_ONLY_COLUMNS_V21 if args.condition_schema == "v2.1" else [],
        "forbidden_default_condition_columns": FORBIDDEN_DEFAULT_CONDITION_COLUMNS_V21 if args.condition_schema == "v2.1" else [],
        "preserved_raw_metadata_columns": sorted(
            set(MODEL_CONDITION_COLUMNS_V21 + AUDIT_ONLY_COLUMNS_V21)
        ) if args.condition_schema == "v2.1" else [],
        "raw_cache": {
            "path": str(raw_path),
            "shape": [int(x) for x in raw_cache.shape],
            "dtype": "float32",
            "description": "First 120 seconds / 4800 samples from each selected SeisBench trace, unnormalized.",
        },
        "model_cache": {
            "path": str(model_path),
            "shape": [int(x) for x in model_cache.shape],
            "dtype": "float32",
            "crop": {
                "pre_sec": MODEL_PRE_SEC,
                "post_sec": MODEL_POST_SEC,
                "sample_rate_hz": SAMPLE_RATE_HZ,
            },
            "normalization": "stream max-abs per sample over all channels and cropped samples",
            "left_pad_nonzero_count": int(np.count_nonzero(left_pads)),
            "right_pad_nonzero_count": int(np.count_nonzero(right_pads)),
        },
        "conditions": str(conditions_path),
        "splits": split_paths,
        "split_counts": {
            str(k): int(v) for k, v in index["subset_split"].value_counts().sort_index().items()
        },
        "source_index": str(index_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote raw cache: {raw_path}")
    print(f"Wrote model cache: {model_path}")
    print(f"Wrote conditions: {conditions_path}")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
