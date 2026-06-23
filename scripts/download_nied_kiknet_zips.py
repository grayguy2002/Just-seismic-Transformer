#!/usr/bin/env python3
"""Download KiK-net ASCII ZIP archives listed by the NIED search index.

NIED protects waveform downloads with HTTP Basic authentication. Provide either
--username/--password or environment variables NIED_USERNAME/NIED_PASSWORD. If
credentials are absent, the script writes an auditable download plan only.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "kiknet_measured_vs30_pwave_v1"
DEFAULT_INDEX = DEFAULT_DATA_DIR / "nied_search" / "nied_kiknet_record_candidates.csv"
NIED_DOWNLOAD_BASE = "https://www.kyoshin.bosai.go.jp/kyoshin/download"


def eqid_to_zip_url(eqid: str, network: str = "kik") -> str:
    eqid = str(eqid)
    year = eqid[:4]
    month = eqid[4:6]
    return f"{NIED_DOWNLOAD_BASE}/{network}/zip/{year}/{month}/{eqid}/{eqid}_ascii.zip"


def auth_header(username: str, password: str) -> str:
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def download_url(url: str, out_path: Path, username: str, password: str, timeout: int = 300) -> tuple[str, str]:
    headers = {"User-Agent": "EEW-KiKnet-Vs30-builder/1.0"}
    if username and password:
        headers["Authorization"] = auth_header(username, password)
    request = urllib.request.Request(url, headers=headers)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                return "error", f"HTTP {status}"
            tmp_path.write_bytes(response.read())
        tmp_path.replace(out_path)
        return "downloaded", ""
    except urllib.error.HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return "auth_required" if exc.code == 401 else "error", f"HTTP {exc.code}: {exc.reason}"
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        return "error", repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--username", default=os.environ.get("NIED_USERNAME", ""))
    parser.add_argument("--password", default=os.environ.get("NIED_PASSWORD", ""))
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    zip_dir = out_dir / "raw_zips"
    zip_dir.mkdir(parents=True, exist_ok=True)
    plan_dir = out_dir / "nied_download"
    plan_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.index)
    if index.empty:
        raise ValueError(f"No records in index: {args.index}")
    events = (
        index[["eqid", "source_origin_time", "source_magnitude"]]
        .drop_duplicates("eqid")
        .sort_values(["source_magnitude", "eqid"], ascending=[False, True])
        .copy()
    )
    if args.max_events > 0:
        events = events.head(args.max_events).copy()

    events["zip_url"] = events["eqid"].map(eqid_to_zip_url)
    events["zip_path"] = events["eqid"].map(lambda x: str(zip_dir / f"{x}_ascii.zip"))

    statuses = []
    has_credentials = bool(args.username and args.password)
    for row in events.itertuples(index=False):
        out_path = Path(row.zip_path)
        if out_path.exists() and not args.overwrite:
            statuses.append({"eqid": row.eqid, "status": "exists", "error": ""})
            continue
        if not has_credentials:
            statuses.append(
                {
                    "eqid": row.eqid,
                    "status": "credentials_missing",
                    "error": "Set NIED_USERNAME and NIED_PASSWORD or pass --username/--password.",
                }
            )
            continue
        status, error = download_url(row.zip_url, out_path, args.username, args.password)
        statuses.append({"eqid": row.eqid, "status": status, "error": error})
        time.sleep(args.sleep_sec)

    status_df = pd.DataFrame(statuses)
    plan = events.merge(status_df, on="eqid", how="left")
    plan_path = plan_dir / "nied_kiknet_zip_download_plan.csv"
    plan.to_csv(plan_path, index=False)

    summary = {
        "source_index": str(args.index),
        "download_plan": str(plan_path),
        "zip_dir": str(zip_dir),
        "events": int(len(events)),
        "has_credentials": has_credentials,
        "status_counts": status_df["status"].value_counts().to_dict() if not status_df.empty else {},
        "auth_note": (
            "NIED waveform ZIP downloads require HTTP Basic authentication. "
            "Use NIED_USERNAME/NIED_PASSWORD environment variables or --username/--password."
        ),
    }
    summary_path = plan_dir / "nied_kiknet_zip_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
