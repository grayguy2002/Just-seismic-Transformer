#!/usr/bin/env python3
"""Build a KiK-net measured/profile Vs30 station manifest.

This manifest is the station-side anchor for an external waveform test set:
each row has a KiK-net station with profile-derived Vs30 from the GFZ/NIED site
database. Proxy Vs30 columns are not used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "measured_vs30_validation" / "japan_knet_kiknet_profile_vs30_source.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "kiknet_measured_vs30_pwave_v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    source_path = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_path)
    required = {
        "source_record_id",
        "source_station_id",
        "source_network",
        "source_station_code",
        "source_station_name",
        "source_latitude",
        "source_longitude",
        "vs30_m_s",
        "nehrp_site_class",
        "profile_depth_m",
        "measurement_quality",
        "reference",
        "url",
        "notes",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Missing required columns in {source_path}: {missing}")

    manifest = source.copy()
    manifest = manifest[manifest["measurement_quality"].eq("profile_direct")].copy()
    manifest = manifest[manifest["source_network"].astype(str).str.upper().eq("KIKNET")].copy()
    manifest = manifest.rename(
        columns={
            "source_record_id": "kiknet_site_code",
            "source_station_id": "station_id",
            "source_network": "station_network_code",
            "source_station_code": "station_code",
            "source_station_name": "station_name",
            "source_latitude": "station_latitude_deg",
            "source_longitude": "station_longitude_deg",
            "profile_depth_m": "vs_profile_depth_m",
        }
    )
    manifest["station_network_code"] = "KIKNET"
    manifest["station_location_code"] = "--"
    manifest["station_elevation_m"] = pd.NA
    manifest["station_local_depth_m"] = 0.0
    manifest["channel_set_expected"] = "NS,EW,UD"
    manifest["trace_channel_for_jst"] = "HN"
    manifest["vs30_source_type"] = "measured_profile"
    manifest["use_as_ground_truth"] = True

    out_cols = [
        "station_id",
        "station_network_code",
        "station_code",
        "station_location_code",
        "kiknet_site_code",
        "station_name",
        "station_latitude_deg",
        "station_longitude_deg",
        "station_elevation_m",
        "station_local_depth_m",
        "vs30_m_s",
        "nehrp_site_class",
        "vs_profile_depth_m",
        "measurement_quality",
        "vs30_source_type",
        "use_as_ground_truth",
        "channel_set_expected",
        "trace_channel_for_jst",
        "reference",
        "url",
        "notes",
    ]
    manifest = manifest[out_cols].sort_values("station_code")

    manifest_path = out_dir / "kiknet_measured_vs30_station_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "source": str(source_path),
        "manifest": str(manifest_path),
        "station_rows": int(len(manifest)),
        "network_counts": manifest["station_network_code"].value_counts().to_dict(),
        "vs30_range_m_s": [
            float(manifest["vs30_m_s"].min()) if not manifest.empty else None,
            float(manifest["vs30_m_s"].max()) if not manifest.empty else None,
        ],
        "nehrp_counts": manifest["nehrp_site_class"].value_counts().sort_index().to_dict(),
        "notes": (
            "The manifest contains only profile-derived KiK-net Vs30 records. "
            "K-NET proxy/regional Vs30 fields are excluded."
        ),
    }
    summary_path = out_dir / "kiknet_measured_vs30_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
