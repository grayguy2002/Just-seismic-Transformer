#!/usr/bin/env python3
"""Prepare a MLAAPDE download plan without downloading waveform archives.

This script only queries public ScienceBase metadata and writes small local
planning files. It never downloads ZIP, CSV, or HDF5 data archives.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_ITEM_ID = "6127b30fd34e40dd9c05094c"
SCIENCEBASE_ITEM_URL = "https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
SCIENCEBASE_CHILDREN_URL = (
    "https://www.sciencebase.gov/catalog/items?parentId={item_id}&format=json&max=1000"
)
START_MONTH = "2013-07"
END_MONTH = "2020-12"
DEFAULT_SELECTED_MONTHS = "2014-04,2017-09,2018-08,2019-07"
MONTH_RE = re.compile(r"(20\d{2})[-_]?([01]\d)")


@dataclass(frozen=True)
class RemoteFile:
    item_id: str
    item_title: str
    name: str
    size_bytes: int
    content_type: str
    download_uri: str
    md5: str
    month: str
    role: str


def fetch_json(url: str, timeout_sec: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "mlaapde-planner/0.1 (+metadata only)",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(2000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc}") from exc


def month_range(start: str, end: str) -> list[str]:
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    months: list[str] = []
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return months


def infer_month(name: str) -> str:
    match = MONTH_RE.search(name)
    if not match:
        return ""
    year, month = match.groups()
    return f"{year}-{month}"


def infer_role(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".csv") or "_mlaapde.csv" in lower:
        return "csv"
    if lower.endswith(".h5") or lower.endswith(".hdf5") or "_mlaapde.h5" in lower:
        return "hdf5"
    return "other"


def extract_files(item: dict[str, Any]) -> list[RemoteFile]:
    item_id = str(item.get("id", ""))
    item_title = str(item.get("title", ""))
    files = []
    for file_info in item.get("files", []) or []:
        name = str(file_info.get("name") or "")
        checksum = file_info.get("checksum") or {}
        files.append(
            RemoteFile(
                item_id=item_id,
                item_title=item_title,
                name=name,
                size_bytes=int(file_info.get("size") or 0),
                content_type=str(file_info.get("contentType") or ""),
                download_uri=str(
                    file_info.get("downloadUri") or file_info.get("url") or ""
                ),
                md5=str(checksum.get("value") or ""),
                month=infer_month(name),
                role=infer_role(name),
            )
        )
    return files


def collect_metadata(item_id: str, timeout_sec: int) -> tuple[dict[str, Any], list[RemoteFile], list[str]]:
    errors: list[str] = []
    parent = fetch_json(SCIENCEBASE_ITEM_URL.format(item_id=item_id), timeout_sec)
    all_files = extract_files(parent)

    try:
        children = fetch_json(SCIENCEBASE_CHILDREN_URL.format(item_id=item_id), timeout_sec)
    except RuntimeError as exc:
        errors.append(str(exc))
        children = {}

    for child in children.get("items", []) or []:
        all_files.extend(extract_files(child))

    return parent, all_files, errors


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size_bytes} B"


def write_manifest(path: Path, files: list[RemoteFile]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "month",
                "role",
                "name",
                "size_bytes",
                "size_human",
                "item_id",
                "item_title",
                "content_type",
                "md5",
                "download_uri",
            ],
        )
        writer.writeheader()
        for remote_file in sorted(files, key=lambda f: (f.month, f.role, f.name)):
            writer.writerow(
                {
                    "month": remote_file.month,
                    "role": remote_file.role,
                    "name": remote_file.name,
                    "size_bytes": remote_file.size_bytes,
                    "size_human": human_size(remote_file.size_bytes),
                    "item_id": remote_file.item_id,
                    "item_title": remote_file.item_title,
                    "content_type": remote_file.content_type,
                    "md5": remote_file.md5,
                    "download_uri": remote_file.download_uri,
                }
            )


def write_month_plan(path: Path, files: list[RemoteFile], months: list[str]) -> None:
    by_month: dict[str, list[RemoteFile]] = {month: [] for month in months}
    for remote_file in files:
        if remote_file.month in by_month:
            by_month[remote_file.month].append(remote_file)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "month",
                "file_count",
                "zip_count",
                "csv_count",
                "hdf5_count",
                "other_count",
                "size_bytes",
                "size_human",
                "status",
            ],
        )
        writer.writeheader()
        for month in months:
            month_files = by_month[month]
            roles = [file.role for file in month_files]
            status = "found" if month_files else "missing_from_metadata"
            writer.writerow(
                {
                    "month": month,
                    "file_count": len(month_files),
                    "zip_count": roles.count("zip"),
                    "csv_count": roles.count("csv"),
                    "hdf5_count": roles.count("hdf5"),
                    "other_count": roles.count("other"),
                    "size_bytes": sum(file.size_bytes for file in month_files),
                    "size_human": human_size(sum(file.size_bytes for file in month_files)),
                    "status": status,
                }
            )


def write_download_script(path: Path, files: list[RemoteFile], months: set[str]) -> None:
    selected = [
        file
        for file in files
        if file.month in months and file.role in {"zip", "csv", "hdf5"} and file.download_uri
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by prepare_mlaapde_plan.py.",
        "# Review the month list and target directory before running.",
        '# Usage: DATA_DIR="/path/to/mlaapde-data" bash plans/download_selected_months.sh',
        "",
        'DATA_DIR="${DATA_DIR:-./mlaapde-data}"',
        'mkdir -p "$DATA_DIR"',
        "",
    ]
    for file in sorted(selected, key=lambda f: (f.month, f.name)):
        safe_name = file.name.replace('"', '\\"')
        safe_url = file.download_uri.replace('"', '\\"')
        lines.append(f'curl -L --fail --continue-at - --output "$DATA_DIR/{safe_name}" "{safe_url}"')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_report(
    path: Path,
    parent: dict[str, Any],
    files: list[RemoteFile],
    months: list[str],
    selected_months: list[str],
    errors: list[str],
) -> None:
    data_files = [file for file in files if file.role in {"zip", "csv", "hdf5"}]
    selected_files = [file for file in data_files if file.month in set(selected_months)]
    lines = [
        "# MLAAPDE Download Plan",
        "",
        f"Generated: {date.today().isoformat()}",
        f"ScienceBase item: {parent.get('title', '')}",
        f"Item id: {parent.get('id', '')}",
        "",
        "This report is metadata-only. No waveform archive was downloaded.",
        "",
        "## Metadata Summary",
        "",
        f"- Expected month range: {months[0]} to {months[-1]} ({len(months)} months)",
        f"- Remote files discovered: {len(files)}",
        f"- Data-like files discovered: {len(data_files)}",
        f"- Discovered data size: {human_size(sum(file.size_bytes for file in data_files))}",
        f"- Selected months: {', '.join(selected_months) if selected_months else '(none)'}",
        f"- Selected data size: {human_size(sum(file.size_bytes for file in selected_files))}",
        "",
        "## Recommended First Subset",
        "",
        "- phase_hint in P, Pn, Pg",
        "- one earliest P-family phase per waves_id",
        "- snr_db >= 15 for v1 strict quality; loosen to >= 10 only if yield is too small",
        "- keep complete 3-component traces",
        "- keep the native 120 s window at 40 Hz",
        "- require complete core condition columns: source_magnitude, source_depth_km, source_distance_deg, source_back_azimuth_deg, phase_travel_sec, phase_time, network, station, channel, event_id, phase_id",
        "- split train/validation/test by event_id, not by phase_id",
        "",
    ]
    if errors:
        lines.extend(
            [
                "## Metadata Warnings",
                "",
                "ScienceBase returned an error for at least one metadata endpoint. Re-run later before finalizing download sizes.",
                "",
            ]
        )
        lines.extend(f"- {error}" for error in errors)
        lines.append("")
    lines.extend(
        [
            "## Files Written",
            "",
            "- plans/mlaapde_manifest.csv: raw remote file manifest",
            "- plans/mlaapde_month_plan.csv: month-level size/status summary",
            "- plans/download_selected_months.sh: optional download script for selected months",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a metadata-only MLAAPDE file manifest and month download plan."
    )
    parser.add_argument("--item-id", default=DEFAULT_ITEM_ID)
    parser.add_argument("--start", default=START_MONTH, help="First month, YYYY-MM")
    parser.add_argument("--end", default=END_MONTH, help="Last month, YYYY-MM")
    parser.add_argument(
        "--months",
        default=DEFAULT_SELECTED_MONTHS,
        help="Comma-separated months to include in the optional download script.",
    )
    parser.add_argument("--out-dir", default="plans")
    parser.add_argument("--timeout-sec", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    months = month_range(args.start, args.end)
    selected_months = [month.strip() for month in args.months.split(",") if month.strip()]
    invalid_months = sorted(set(selected_months) - set(months))
    if invalid_months:
        print(f"Invalid selected months outside range: {', '.join(invalid_months)}", file=sys.stderr)
        return 2

    parent, files, errors = collect_metadata(args.item_id, args.timeout_sec)

    write_manifest(out_dir / "mlaapde_manifest.csv", files)
    write_month_plan(out_dir / "mlaapde_month_plan.csv", files, months)
    write_download_script(out_dir / "download_selected_months.sh", files, set(selected_months))
    write_report(out_dir / "mlaapde_download_plan.md", parent, files, months, selected_months, errors)

    print(f"Wrote metadata-only MLAAPDE plan to {out_dir}")
    if errors:
        print("Warnings were recorded in plans/mlaapde_download_plan.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
