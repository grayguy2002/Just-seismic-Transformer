#!/usr/bin/env python3
"""Build dense KiK-net single-earthquake JsT caches from downloaded event ZIPs.

The original Exp F cache is intentionally small because it follows the NIED
search candidate table.  For a direct single-earthquake validation, each
selected ZIP can instead supply all KiK-net surface stations with profile Vs30.
This script keeps the existing JsT cache contract and writes files with the
same cache prefix used by Exp F, so the inference script can be reused by
changing only --data-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from build_kiknet_mlaapde_compatible_cache import (
    CONDITION_COLUMNS,
    DEFAULT_DATA_DIR,
    DEFAULT_INDEX,
    DEFAULT_MANIFEST,
    DEFAULT_TAG,
    DEFAULT_ZIP_DIR,
    MODEL_CONDITION_COLUMNS_V21,
    SAMPLE_RATE_HZ,
    SURFACE_COMPONENTS,
    build_row,
    load_event_zip,
    resample_linear,
)


DEFAULT_OUT_DIR = DEFAULT_DATA_DIR.parent / "kiknet_dense_single_event_v1"


def count_manifest_triplets(zip_path: Path, allowed_stations: set[str]) -> int:
    comps: dict[str, set[str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name.upper()
            if len(name) < 6:
                continue
            station = name[:6]
            if station not in allowed_stations:
                continue
            for comp in SURFACE_COMPONENTS:
                if name.endswith(comp) or name.endswith(f".{comp}"):
                    comps.setdefault(station, set()).add(comp)
    needed = set(SURFACE_COMPONENTS)
    return sum(1 for have in comps.values() if needed.issubset(have))


def select_events(zip_dir: Path, allowed_stations: set[str], requested: str, top_events: int, min_triplets: int) -> pd.DataFrame:
    if requested:
        rows = []
        requested_ids = [x.strip() for x in requested.split(",") if x.strip()]
        for event_id in requested_ids:
            zip_path = zip_dir / f"{event_id}_ascii.zip"
            rows.append({
                "event_id": event_id,
                "zip_path": str(zip_path),
                "manifest_triplets": count_manifest_triplets(zip_path, allowed_stations) if zip_path.exists() else 0,
                "selected": zip_path.exists(),
            })
        return pd.DataFrame(rows)

    rows = []
    for zip_path in sorted(zip_dir.glob("*_ascii.zip")):
        if zip_path.name.startswith("._"):
            continue
        try:
            n_triplets = count_manifest_triplets(zip_path, allowed_stations)
        except zipfile.BadZipFile:
            n_triplets = 0
        rows.append({
            "event_id": zip_path.name.replace("_ascii.zip", ""),
            "zip_path": str(zip_path),
            "manifest_triplets": int(n_triplets),
            "selected": False,
        })
    df = pd.DataFrame(rows).sort_values("manifest_triplets", ascending=False)
    mask = df["manifest_triplets"] >= min_triplets
    selected = df[mask].head(top_events).index
    df.loc[selected, "selected"] = True
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--zip-dir", default=str(DEFAULT_ZIP_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dataset-tag", default=DEFAULT_TAG)
    parser.add_argument("--events", default="", help="Comma-separated event IDs. If omitted, choose dense events automatically.")
    parser.add_argument("--top-events", type=int, default=5)
    parser.add_argument("--min-manifest-triplets", type=int, default=150)
    parser.add_argument("--min-snr-db", type=float, default=-999.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest_by_code = {str(r.station_code): pd.Series(r._asdict()) for r in manifest.itertuples(index=False)}
    allowed_stations = set(manifest_by_code)
    index = pd.read_csv(args.index)
    index["eqid"] = index["eqid"].astype(str)
    event_meta = {event_id: group.iloc[0].copy() for event_id, group in index.groupby("eqid", sort=False)}

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

    selection = select_events(
        Path(args.zip_dir),
        allowed_stations,
        args.events,
        top_events=args.top_events,
        min_triplets=args.min_manifest_triplets,
    )
    selection_path = cache_dir / f"{args.dataset_tag}_event_selection.csv"
    selection.to_csv(selection_path, index=False)

    rows = []
    raw_waves = []
    model_waves = []
    skipped = []
    per_event = []
    for item in selection[selection["selected"]].itertuples(index=False):
        event_id = str(item.event_id)
        zip_path = Path(item.zip_path)
        meta = event_meta.get(event_id)
        if meta is None:
            skipped.append({"event_id": event_id, "station_code": "", "reason": "missing_event_metadata"})
            continue
        try:
            event_traces = load_event_zip(zip_path, allowed_stations)
        except zipfile.BadZipFile:
            skipped.append({"event_id": event_id, "station_code": "", "reason": "bad_zip_file"})
            continue

        n_before = len(rows)
        needed = set(SURFACE_COMPONENTS)
        for station_code in sorted(event_traces):
            comps = event_traces[station_code]
            if not needed.issubset(comps):
                skipped.append({"event_id": event_id, "station_code": station_code, "reason": "missing_surface_components"})
                continue
            station = manifest_by_code.get(station_code)
            if station is None:
                continue
            record = meta.copy()
            record["eqid"] = event_id
            record["station_code"] = station_code
            record["path_ep_distance_km"] = math.nan
            traces = [comps["NS2"], comps["EW2"], comps["UD2"]]
            try:
                resampled = [resample_linear(t.samples, t.sampling_rate_hz, SAMPLE_RATE_HZ) for t in traces]
                n_samples = min(len(x) for x in resampled)
                wave = np.stack([resampled[0][:n_samples], resampled[1][:n_samples], resampled[2][:n_samples]]).astype(np.float32)
                row, raw_wave, model_wave = build_row(record, station, wave, traces[0].trace_start, zip_path, len(rows))
                if math.isfinite(row["trace_snr_db"]) and row["trace_snr_db"] < args.min_snr_db:
                    skipped.append({"event_id": event_id, "station_code": station_code, "reason": "low_snr"})
                    continue
            except Exception as exc:
                skipped.append({"event_id": event_id, "station_code": station_code, "reason": repr(exc)})
                continue
            rows.append(row)
            raw_waves.append(raw_wave)
            model_waves.append(model_wave)
        per_event.append({
            "event_id": event_id,
            "manifest_triplets": int(item.manifest_triplets),
            "cache_rows": int(len(rows) - n_before),
        })
        print(f"{event_id}: built {len(rows) - n_before} rows from {item.manifest_triplets} manifest triplets")

    conditions = pd.DataFrame(rows)
    if conditions.empty:
        raise ValueError("No dense cache rows built; inspect skipped records and event selection.")
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
    per_event_path = cache_dir / f"{args.dataset_tag}_event_build_summary.csv"
    pd.DataFrame(per_event).to_csv(per_event_path, index=False)

    summary = {
        "dataset_tag": args.dataset_tag,
        "purpose": "dense single-earthquake KiK-net measured-Vs30 validation cache",
        "n_samples": int(len(conditions)),
        "station_count": int(conditions["station_code"].nunique()),
        "event_count": int(conditions["event_id"].nunique()),
        "condition_schema_version": "v2.1",
        "condition_columns": CONDITION_COLUMNS,
        "model_condition_columns": MODEL_CONDITION_COLUMNS_V21,
        "events": per_event,
        "raw_cache": {"path": str(raw_path), "shape": list(np.load(raw_path, mmap_mode="r").shape), "dtype": "float32"},
        "model_cache": {"path": str(model_path), "shape": list(np.load(model_path, mmap_mode="r").shape), "dtype": "float32"},
        "conditions": str(conditions_path),
        "splits": {"testing": str(split_dir / "testing_indices.npy")},
        "event_selection": str(selection_path),
        "skipped_records": str(skipped_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
