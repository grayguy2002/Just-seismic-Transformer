#!/usr/bin/env python3
"""Filter the downloaded SeisBench MLAAPDE chunks into a P-wave subset index.

The script writes metadata/index files only. It does not copy waveform arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


P_FAMILY = ["P", "Pn", "Pg"]
PHASE_PRIORITY = {phase: i for i, phase in enumerate(P_FAMILY)}
TRACE_RE = re.compile(r"^(?P<bucket>[^$]+)\$(?P<index>\d+),:(?P<n_channels>\d+),:(?P<n_samples>\d+)$")

LEGACY_DATASET_TAG = "mlaapde_pwave_v1"
V21_EXTRA_COLUMNS = [
    "source_origin_uncertainty_sec",
    "source_latitude_uncertainty_km",
    "source_longitude_uncertainty_km",
    "source_depth_uncertainty_km",
    "source_magnitude_uncertainty",
    "source_magnitude_author",
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
    "selected_phase_analyst_id",
]


CORE_NUMERIC_COLUMNS = [
    "source_depth_km",
    "source_magnitude",
    "source_latitude_deg",
    "source_longitude_deg",
    "station_latitude_deg",
    "station_longitude_deg",
    "station_elevation_m",
    "path_ep_distance_deg",
    "path_ep_distance_km",
    "path_back_azimuth_deg",
    "trace_sampling_rate_hz",
    "trace_snr_db",
]

CORE_TEXT_COLUMNS = [
    "source_id",
    "source_origin_time",
    "source_magnitude_type",
    "station_network_code",
    "station_code",
    "trace_channel",
    "trace_start_time",
    "trace_name",
]


def parse_trace_name(value: str) -> dict[str, Any]:
    match = TRACE_RE.match(str(value))
    if not match:
        return {
            "hdf5_bucket": "",
            "hdf5_index": np.nan,
            "trace_n_channels": np.nan,
            "trace_n_samples": np.nan,
        }

    groups = match.groupdict()
    return {
        "hdf5_bucket": groups["bucket"],
        "hdf5_index": int(groups["index"]),
        "trace_n_channels": int(groups["n_channels"]),
        "trace_n_samples": int(groups["n_samples"]),
    }


def load_hdf5_shapes(raw_dir: Path) -> dict[str, dict[str, tuple[int, int, int]]]:
    shapes: dict[str, dict[str, tuple[int, int, int]]] = {}
    for path in sorted(raw_dir.glob("waveforms_*.hdf5")):
        month = path.stem.replace("waveforms_", "")
        with h5py.File(path, "r") as h5:
            shapes[month] = {
                bucket: tuple(h5["data"][bucket].shape)
                for bucket in h5["data"].keys()
            }
    return shapes


def infer_selected_phase(df: pd.DataFrame) -> pd.DataFrame:
    phase_frames = []
    for phase in P_FAMILY:
        arrival_col = f"trace_{phase}_arrival_sample"
        status_col = f"trace_{phase}_status"
        analyst_col = f"trace_{phase}_analyst_id"
        if arrival_col not in df.columns:
            continue

        frame = pd.DataFrame(
            {
                "selected_phase": phase,
                "selected_phase_priority": PHASE_PRIORITY[phase],
                "selected_phase_arrival_sample": pd.to_numeric(df[arrival_col], errors="coerce"),
                "selected_phase_status": df.get(status_col, pd.Series("", index=df.index)),
                "selected_phase_analyst_id": df.get(analyst_col, pd.Series("", index=df.index)),
            },
            index=df.index,
        )
        phase_frames.append(frame)

    if not phase_frames:
        raise ValueError("No P-family arrival columns were found in the metadata.")

    phases = pd.concat(phase_frames)
    phases = phases.dropna(subset=["selected_phase_arrival_sample"])
    phases = phases.sort_values(
        ["selected_phase_arrival_sample", "selected_phase_priority"],
        kind="mergesort",
    )
    earliest = phases.groupby(level=0, sort=False).first()

    out = df.join(earliest, how="left")
    return out


def add_derived_columns(df: pd.DataFrame, month: str) -> pd.DataFrame:
    out = infer_selected_phase(df)
    out["month"] = f"{month[:4]}-{month[4:]}"
    out["month_compact"] = month

    trace_parts = out["trace_name"].apply(parse_trace_name).apply(pd.Series)
    out = pd.concat([out, trace_parts], axis=1)

    location = out.get("station_location_code", pd.Series("", index=out.index))
    out["station_location_code"] = location.fillna("").replace("", "--")
    out["event_id"] = out["source_id"].astype(str)
    out["station_id"] = (
        out["station_network_code"].astype(str)
        + "."
        + out["station_code"].astype(str)
    )
    out["waves_id"] = (
        out["event_id"].astype(str)
        + "_"
        + out["station_network_code"].astype(str)
        + "."
        + out["station_code"].astype(str)
        + "."
        + out["trace_channel"].astype(str)
        + "."
        + out["station_location_code"].astype(str)
    )
    out["phase_id"] = out["waves_id"] + "_" + out["selected_phase"].fillna("")

    out["trace_sampling_rate_hz"] = pd.to_numeric(out["trace_sampling_rate_hz"], errors="coerce")
    out["phase_time"] = pd.to_datetime(out["trace_start_time"], utc=True, errors="coerce") + pd.to_timedelta(
        out["selected_phase_arrival_sample"] / out["trace_sampling_rate_hz"], unit="s"
    )
    out["source_origin_time_dt"] = pd.to_datetime(out["source_origin_time"], utc=True, errors="coerce")
    out["phase_travel_sec"] = (
        out["phase_time"] - out["source_origin_time_dt"]
    ).dt.total_seconds()
    out["phase_time"] = out["phase_time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    for col in CORE_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def mark_hdf5_valid(df: pd.DataFrame, shapes: dict[str, dict[str, tuple[int, int, int]]]) -> pd.Series:
    valid = []
    for row in df.itertuples(index=False):
        month_shapes = shapes.get(row.month_compact, {})
        shape = month_shapes.get(row.hdf5_bucket)
        ok = (
            shape is not None
            and pd.notna(row.hdf5_index)
            and int(row.hdf5_index) < shape[0]
            and shape[1] >= 3
            and int(row.trace_n_channels) == 3
            and int(row.trace_n_samples) <= shape[2]
        )
        valid.append(ok)
    return pd.Series(valid, index=df.index)


def assign_event_splits(df: pd.DataFrame, seed: int) -> pd.Series:
    events = np.array(sorted(df["event_id"].dropna().unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(events)

    n = len(events)
    n_train = int(round(n * 0.8))
    n_valid = int(round(n * 0.1))

    split_by_event = {}
    for event in events[:n_train]:
        split_by_event[event] = "training"
    for event in events[n_train : n_train + n_valid]:
        split_by_event[event] = "validation"
    for event in events[n_train + n_valid :]:
        split_by_event[event] = "testing"

    return df["event_id"].map(split_by_event)


def apply_balancing(df: pd.DataFrame, max_per_event: int, max_per_station_event: int) -> pd.DataFrame:
    ordered = df.sort_values(
        ["event_id", "station_id", "trace_snr_db", "phase_travel_sec"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    station_rank = ordered.groupby(["event_id", "station_id"]).cumcount()
    ordered = ordered[station_rank < max_per_station_event]

    ordered = ordered.sort_values(
        ["event_id", "trace_snr_db", "phase_travel_sec"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    event_rank = ordered.groupby("event_id").cumcount()
    return ordered[event_rank < max_per_event].copy()


def build_summary(raw_rows: int, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    summary: dict[str, Any] = {"raw_rows": raw_rows}
    for name, df in frames.items():
        item: dict[str, Any] = {
            "rows": int(len(df)),
            "events": int(df["event_id"].nunique()) if "event_id" in df else None,
        }
        if "month" in df:
            item["rows_by_month"] = {
                str(k): int(v) for k, v in df["month"].value_counts().sort_index().items()
            }
        if "selected_phase" in df:
            item["rows_by_phase"] = {
                str(k): int(v) for k, v in df["selected_phase"].value_counts().sort_index().items()
            }
        if "subset_split" in df:
            item["rows_by_split"] = {
                str(k): int(v) for k, v in df["subset_split"].value_counts().sort_index().items()
            }
            item["events_by_split"] = {
                str(k): int(v)
                for k, v in df.groupby("subset_split")["event_id"].nunique().sort_index().items()
            }
        summary[name] = item
    return summary


def build_output_columns(condition_schema: str) -> list[str]:
    columns = [
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
        "hdf5_reference_valid",
    ]
    if condition_schema == "v2.1":
        for col in V21_EXTRA_COLUMNS:
            if col not in columns:
                columns.append(col)
    return columns


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/seisbench_mlaapde_pwave_v1/raw")
    parser.add_argument("--out-dir", default="data/seisbench_mlaapde_pwave_v1/filtered")
    parser.add_argument("--min-snr-db", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--max-per-event", type=int, default=5000)
    parser.add_argument("--max-per-station-event", type=int, default=3)
    parser.add_argument("--dataset-tag", default=LEGACY_DATASET_TAG)
    parser.add_argument("--condition-schema", choices=["legacy", "v2.1"], default="legacy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shapes = load_hdf5_shapes(raw_dir)
    parts = []
    raw_rows = 0
    for path in sorted(raw_dir.glob("metadata_*.csv")):
        month = path.stem.replace("metadata_", "")
        df = pd.read_csv(path, low_memory=False)
        raw_rows += len(df)
        parts.append(add_derived_columns(df, month))

    all_rows = pd.concat(parts, ignore_index=True)

    all_rows["hdf5_reference_valid"] = mark_hdf5_valid(all_rows, shapes)

    complete_text = all_rows[CORE_TEXT_COLUMNS].notna().all(axis=1)
    for col in CORE_TEXT_COLUMNS:
        complete_text &= all_rows[col].astype(str).str.len() > 0
    complete_numeric = all_rows[CORE_NUMERIC_COLUMNS].notna().all(axis=1)

    strict = all_rows[
        all_rows["selected_phase"].isin(P_FAMILY)
        & (all_rows["trace_snr_db"] >= args.min_snr_db)
        & (all_rows["trace_sampling_rate_hz"] == 40)
        & (all_rows["trace_n_channels"] == 3)
        & (all_rows["trace_n_samples"] >= 4800)
        & all_rows["hdf5_reference_valid"]
        & complete_text
        & complete_numeric
        & all_rows["phase_travel_sec"].notna()
        & (all_rows["phase_travel_sec"] > 0)
    ].copy()

    strict = strict.sort_values(
        ["waves_id", "phase_time", "selected_phase_priority", "trace_snr_db"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    strict = strict.groupby("waves_id", as_index=False, sort=False).first()

    balanced = apply_balancing(strict, args.max_per_event, args.max_per_station_event)
    balanced["subset_split"] = assign_event_splits(balanced, args.seed)

    output_columns = build_output_columns(args.condition_schema)
    ensure_columns(strict, output_columns)
    ensure_columns(balanced, output_columns)

    strict_for_output = strict.copy()
    strict_for_output["subset_split"] = ""

    strict_path = out_dir / f"{args.dataset_tag}_strict_index.csv"
    balanced_path = out_dir / f"{args.dataset_tag}_balanced_index.csv"
    summary_path = out_dir / f"{args.dataset_tag}_filter_summary.json"
    refuse_overwrite([strict_path, balanced_path, summary_path], args.overwrite)

    strict_for_output[output_columns].to_csv(strict_path, index=False, quoting=csv.QUOTE_MINIMAL)
    balanced[output_columns].to_csv(balanced_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = build_summary(
        raw_rows,
        {
            "all_metadata": all_rows,
            "strict_deduplicated": strict,
            "balanced_split": balanced,
        },
    )
    summary.update(
        {
            "condition_schema": args.condition_schema,
            "dataset_tag": args.dataset_tag,
            "output_columns": output_columns,
            "v21_extra_columns": V21_EXTRA_COLUMNS if args.condition_schema == "v2.1" else [],
            "filter": {
                "phase_family": P_FAMILY,
                "min_snr_db": args.min_snr_db,
                "sample_rate_hz": 40,
                "min_trace_n_samples": 4800,
                "deduplicate_by": "waves_id",
                "deduplicate_order": ["phase_time", "selected_phase_priority", "trace_snr_db desc"],
                "max_per_event": args.max_per_event,
                "max_per_station_event": args.max_per_station_event,
                "split_group": "event_id",
                "split_seed": args.seed,
            },
            "outputs": {
                "strict_index": str(strict_path),
                "balanced_index": str(balanced_path),
                "summary": str(summary_path),
            },
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote strict index: {strict_path} ({len(strict):,} rows)")
    print(f"Wrote balanced split index: {balanced_path} ({len(balanced):,} rows)")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
