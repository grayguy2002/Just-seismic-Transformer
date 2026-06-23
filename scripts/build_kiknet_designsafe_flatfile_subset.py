#!/usr/bin/env python3
"""Build a measured/profile Vs30 subset from the DesignSafe KiK-net flatfile.

This script streams the public PRJ-2547 `attributes.csv` flatfile and keeps only
records from the 656 KiK-net stations whose Vs30 values are profile-derived in
our GFZ/NIED station manifest. It deliberately does not download the 14 GB HDF5
ground-motion store.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATION_MANIFEST = (
    ROOT
    / "data"
    / "kiknet_measured_vs30_pwave_v1"
    / "kiknet_measured_vs30_station_manifest.csv"
)
DEFAULT_OUT_DIR = ROOT / "data" / "kiknet_measured_vs30_pwave_v1" / "designsafe_flatfile_validation"

DESIGNSAFE_SYSTEM = "designsafe.storage.published"
DESIGNSAFE_API_BASE = "https://www.designsafe-ci.org/api/datafiles/tapis/public"
DESIGNSAFE_PROJECT_URL = "https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-2547/?version=2"
DESIGNSAFE_DOI = "10.17603/ds2-e0ts-c070"
DESIGNSAFE_ATTRIBUTES_PATH = (
    "/published-data/PRJ-2547/"
    "Project--ground-motion-parameters-for-kik-net-records-an-updated-database--V2/"
    "data/attributes.csv"
)
DESIGNSAFE_README_PATH = (
    "/published-data/PRJ-2547/"
    "Project--ground-motion-parameters-for-kik-net-records-an-updated-database--V2/"
    "data/Read me.txt"
)


KEY_COLUMNS = [
    "Code",
    "Adress",
    "station",
    "Vs30",
    "Vs5",
    "Vs10",
    "Vs20",
    "Vs50",
    "Vs100",
    "Vs800",
    "Z1",
    "stationLat",
    "stationLon",
    "recordDate",
    "originDate",
    "originTime",
    "durationTime",
    "samplingFreq",
    "lat",
    "lon",
    "depth",
    "mag",
    "JMA_Magnitude_",
    "MT_Magnitude_",
    "MT_Depth_",
    "JMA_Depth_",
    "Focal_mechanism_kegan_",
    "Focal_mechanism_Garcia_",
    "Tectonic_Zhoa_",
    "Tectonic_Garcia_",
    "repi_0",
    "repi_1",
    "rhypo_0",
    "rhypo_1",
    "rrup_0",
    "rrup_1",
    "rjb_0",
    "rjb_1",
    "snr_EW1",
    "snr_NS1",
    "snr_EW2",
    "snr_NS2",
    "fLow_EW1",
    "fLow_NS1",
    "fLow_EW2",
    "fLow_NS2",
    "fHigh_EW1",
    "fHigh_NS1",
    "fHigh_EW2",
    "fHigh_NS2",
    "pga_EW1",
    "pga_NS1",
    "pga_EW2",
    "pga_NS2",
    "Ia_EW1",
    "Ia_NS1",
    "Ia_EW2",
    "Ia_NS2",
    "D5_75_EW1",
    "D5_75_NS1",
    "D5_75_EW2",
    "D5_75_NS2",
    "D20_80_EW1",
    "D20_80_NS1",
    "D20_80_EW2",
    "D20_80_NS2",
    "D5_95_EW1",
    "D5_95_NS1",
    "D5_95_EW2",
    "D5_95_NS2",
]


def design_safe_preview_url(path: str) -> str:
    quoted = urllib.parse.quote(path, safe="")
    return f"{DESIGNSAFE_API_BASE}/preview/{DESIGNSAFE_SYSTEM}/{quoted}/"


def request_json(url: str, timeout: int = 120) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "EEW-KiKnet-Vs30-builder/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def resolve_postit_href(path: str) -> str:
    payload = request_json(design_safe_preview_url(path))
    href = payload.get("href")
    if not href:
        raise RuntimeError(f"DesignSafe preview response has no href for {path}: {payload}")
    return href


def download_file(url: str, out_path: Path, *, overwrite: bool = False) -> Path:
    if out_path.exists() and not overwrite:
        return out_path
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "EEW-KiKnet-Vs30-builder/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response, tmp_path.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1024 * 1024)
    tmp_path.replace(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-manifest", default=str(DEFAULT_STATION_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--max-chunks", type=int, default=0, help="Debug limit; 0 means stream all chunks.")
    parser.add_argument("--cache-attributes", action="store_true", help="Download attributes.csv before filtering.")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_manifest = pd.read_csv(args.station_manifest)
    station_codes = set(station_manifest["station_code"].astype(str))
    station_cols = [
        "station_id",
        "station_code",
        "station_name",
        "station_latitude_deg",
        "station_longitude_deg",
        "vs30_m_s",
        "nehrp_site_class",
        "vs_profile_depth_m",
        "measurement_quality",
        "vs30_source_type",
        "use_as_ground_truth",
    ]
    station_lookup = station_manifest[station_cols].copy()

    readme_href = resolve_postit_href(DESIGNSAFE_README_PATH)
    attributes_href = resolve_postit_href(DESIGNSAFE_ATTRIBUTES_PATH)

    readme_path = out_dir / "designsafe_prj2547_readme.txt"
    with urllib.request.urlopen(
        urllib.request.Request(readme_href, headers={"User-Agent": "EEW-KiKnet-Vs30-builder/1.0"}),
        timeout=120,
    ) as response:
        readme_path.write_bytes(response.read())

    input_source: str | Path = attributes_href
    attributes_cache_path = out_dir / "designsafe_prj2547_attributes.csv"
    if args.cache_attributes:
        input_source = download_file(attributes_href, attributes_cache_path, overwrite=args.overwrite_cache)

    selected_chunks: list[pd.DataFrame] = []
    total_rows = 0
    kept_rows = 0
    chunks_seen = 0
    columns_seen: list[str] | None = None

    for chunk in pd.read_csv(input_source, chunksize=args.chunksize, low_memory=False):
        chunks_seen += 1
        if columns_seen is None:
            columns_seen = list(chunk.columns)
        total_rows += len(chunk)
        if "station" not in chunk.columns:
            raise ValueError("DesignSafe attributes.csv does not contain a 'station' column.")
        subset = chunk[chunk["station"].astype(str).isin(station_codes)].copy()
        if not subset.empty:
            available = [col for col in KEY_COLUMNS if col in subset.columns]
            subset = subset[available]
            selected_chunks.append(subset)
            kept_rows += len(subset)
        print(f"streamed chunk {chunks_seen}, rows={total_rows}, kept={kept_rows}", flush=True)
        if args.max_chunks > 0 and chunks_seen >= args.max_chunks:
            break

    if selected_chunks:
        selected = pd.concat(selected_chunks, ignore_index=True)
    else:
        selected = pd.DataFrame(columns=[col for col in KEY_COLUMNS if columns_seen and col in columns_seen])

    selected = selected.merge(station_lookup, how="left", left_on="station", right_on="station_code", validate="many_to_one")
    selected["designsafe_project"] = "PRJ-2547"
    selected["designsafe_doi"] = DESIGNSAFE_DOI
    selected["designsafe_project_url"] = DESIGNSAFE_PROJECT_URL
    selected["designsafe_attributes_path"] = DESIGNSAFE_ATTRIBUTES_PATH
    selected["validation_role"] = "measured_vs30_with_public_designsafe_ground_motion_parameters"
    selected["raw_waveform_included"] = False

    prefix = "debug_" if args.max_chunks > 0 else ""
    out_path = out_dir / f"{prefix}kiknet_measured_vs30_designsafe_ground_motion_subset.csv"
    selected.to_csv(out_path, index=False)

    summary = {
        "station_manifest": str(args.station_manifest),
        "designsafe_project": "PRJ-2547",
        "designsafe_doi": DESIGNSAFE_DOI,
        "designsafe_project_url": DESIGNSAFE_PROJECT_URL,
        "designsafe_attributes_path": DESIGNSAFE_ATTRIBUTES_PATH,
        "designsafe_attributes_href": attributes_href,
        "attributes_cache": str(attributes_cache_path) if args.cache_attributes else None,
        "readme": str(readme_path),
        "output": str(out_path),
        "station_rows_in_manifest": int(len(station_manifest)),
        "source_rows_streamed": int(total_rows),
        "selected_rows": int(len(selected)),
        "selected_unique_stations": int(selected["station"].nunique()) if "station" in selected.columns else 0,
        "selected_vs30_range_m_s_manifest": [
            float(selected["vs30_m_s"].min()) if not selected.empty else None,
            float(selected["vs30_m_s"].max()) if not selected.empty else None,
        ],
        "selected_nehrp_counts_manifest": selected["nehrp_site_class"].value_counts().sort_index().to_dict()
        if "nehrp_site_class" in selected.columns
        else {},
        "chunksize": args.chunksize,
        "chunks_seen": int(chunks_seen),
        "max_chunks": int(args.max_chunks),
        "columns_retained": list(selected.columns),
        "provenance_note": (
            "DesignSafe PRJ-2547 flatfile rows were filtered to stations whose local "
            "manifest Vs30 is GFZ/NIED profile-derived. The DesignSafe Vs30 column is "
            "retained for comparison, but the ground-truth Vs30 field for this project "
            "is vs30_m_s from kiknet_measured_vs30_station_manifest.csv."
        ),
        "limitations": [
            "This is a processed ground-motion parameter flatfile, not raw waveform data.",
            "The 14 GB Database_small.hdf5 file is not downloaded by this script.",
            "Use vs30_m_s from the measured/profile station manifest as the project ground-truth Vs30.",
        ],
    }
    summary_path = out_dir / f"{prefix}kiknet_measured_vs30_designsafe_ground_motion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
