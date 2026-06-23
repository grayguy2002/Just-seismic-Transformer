#!/usr/bin/env python3
"""Search NIED K-NET/KiK-net records for measured-Vs30 KiK-net stations.

The script uses the public search endpoint behind the NIED "Download by Data
Condition" page. Downloading waveform ZIP files still requires NIED
authentication; this script only builds an auditable candidate record index.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "kiknet_measured_vs30_pwave_v1"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "kiknet_measured_vs30_station_manifest.csv"
NIED_BASE = "https://www.kyoshin.bosai.go.jp"
SEARCH_URL = f"{NIED_BASE}/en/search/dt/"
FORM_URL = f"{NIED_BASE}/en/dtdownload/"


class NiedClient:
    def __init__(self) -> None:
        self.cookies = CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookies))

    def open(self, request: urllib.request.Request, timeout: int = 120) -> bytes:
        with self.opener.open(request, timeout=timeout) as response:
            return response.read()

    def get_text(self, url: str, timeout: int = 60) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "EEW-KiKnet-Vs30-builder/1.0"})
        return self.open(request, timeout=timeout).decode("utf-8", errors="replace")

    def post_json(self, url: str, data: dict[str, Any], headers: dict[str, str], timeout: int = 120) -> dict[str, Any]:
        body = urllib.parse.urlencode(data).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "User-Agent": "EEW-KiKnet-Vs30-builder/1.0",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                **headers,
            },
            method="POST",
        )
        payload = self.open(request, timeout=timeout)
        return json.loads(payload.decode("utf-8", errors="replace"))

    def cookie_value(self, name: str) -> str:
        for cookie in self.cookies:
            if cookie.name == name:
                return cookie.value
        return ""


def get_csrf(client: NiedClient) -> str:
    text = client.get_text(FORM_URL)
    cookie_token = client.cookie_value("csrftoken")
    if cookie_token:
        return cookie_token
    match = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', text)
    return match.group(1) if match else ""


def nied_search(
    client: NiedClient,
    csrf_token: str,
    *,
    site_code: str,
    date_from: str,
    date_to: str,
    min_magnitude: float | None,
    max_records_per_station: int,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "csrfmiddlewaretoken": csrf_token,
        "datakind": "2",  # KiK-net
        "date_from": date_from,
        "date_to": date_to,
        "site_method": "0",
        "sitecode": site_code,
        "sort_id": "5",
        "sort_kind": "desc",
    }
    if min_magnitude is not None:
        data["mag1"] = str(min_magnitude)
    headers = {
        "Referer": FORM_URL,
        "X-CSRFToken": csrf_token,
        "X-Requested-With": "XMLHttpRequest",
    }
    payload = client.post_json(SEARCH_URL, data=data, headers=headers, timeout=120)
    results = payload.get("results", [])
    if max_records_per_station > 0:
        results = results[:max_records_per_station]
    payload["results"] = results
    return payload


def normalize_record(record: dict[str, Any], station: pd.Series) -> dict[str, Any]:
    station_code = str(station["station_code"])
    eqid = str(record.get("eqid") or "")
    site_type_name = str(record.get("site_type_name") or "KiK-net")
    return {
        "accdat_id": record.get("accdat_id"),
        "eqid": eqid,
        "download_select_site": f"{eqid}_{site_type_name}_{station_code}",
        "site_type_name": site_type_name,
        "station_id": station["station_id"],
        "station_network_code": station["station_network_code"],
        "station_code": station_code,
        "station_location_code": station.get("station_location_code", "--"),
        "station_name": station.get("station_name", ""),
        "station_latitude_deg": station["station_latitude_deg"],
        "station_longitude_deg": station["station_longitude_deg"],
        "station_elevation_m": station.get("station_elevation_m", pd.NA),
        "station_local_depth_m": station.get("station_local_depth_m", 0.0),
        "vs30_m_s": station["vs30_m_s"],
        "nehrp_site_class": station["nehrp_site_class"],
        "vs_profile_depth_m": station["vs_profile_depth_m"],
        "record_start_time": record.get("record_start_time"),
        "source_origin_time": record.get("origintime"),
        "source_latitude_deg": record.get("originlat"),
        "source_longitude_deg": record.get("originlon"),
        "source_depth_km": record.get("depth"),
        "source_magnitude": record.get("mag"),
        "source_magnitude_type": "M",
        "source_magnitude_author": "NIED",
        "path_ep_distance_km": record.get("distance"),
        "maxacc_gal": record.get("maxacc"),
        "maxvel_cm_s": record.get("maxvel"),
        "inst_seismic_intensity": record.get("inst_seismic_intensity"),
        "nied_datname": record.get("datname"),
        "nied_acmap": record.get("acmap"),
        "raw_search_record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--date-from", default="1996/01/01")
    parser.add_argument("--date-to", default="2026/06/17")
    parser.add_argument("--min-magnitude", type=float, default=4.5)
    parser.add_argument("--max-stations", type=int, default=0)
    parser.add_argument("--max-records-per-station", type=int, default=20)
    parser.add_argument("--sleep-sec", type=float, default=0.4)
    parser.add_argument("--station-code", action="append", default=[])
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.station_code:
        allowed = {x.upper() for x in args.station_code}
        manifest = manifest[manifest["station_code"].astype(str).str.upper().isin(allowed)].copy()
    if args.max_stations > 0:
        manifest = manifest.head(args.max_stations).copy()

    out_dir = Path(args.out_dir)
    index_dir = out_dir / "nied_search"
    index_dir.mkdir(parents=True, exist_ok=True)
    client = NiedClient()
    csrf = get_csrf(client)

    rows = []
    station_summaries = []
    for i, station in enumerate(manifest.itertuples(index=False), start=1):
        station_series = pd.Series(station._asdict())
        site_code = str(station_series["station_code"])
        try:
            payload = nied_search(
                client,
                csrf,
                site_code=site_code,
                date_from=args.date_from,
                date_to=args.date_to,
                min_magnitude=args.min_magnitude,
                max_records_per_station=args.max_records_per_station,
            )
        except Exception as exc:
            station_summaries.append(
                {
                    "station_code": site_code,
                    "status": "error",
                    "error": repr(exc),
                    "records_kept": 0,
                    "records_total_reported": None,
                }
            )
            continue

        results = payload.get("results", [])
        for record in results:
            rows.append(normalize_record(record, station_series))
        station_summaries.append(
            {
                "station_code": site_code,
                "status": "ok",
                "records_kept": len(results),
                "records_total_reported": payload.get("total"),
            }
        )
        if i % 25 == 0:
            print(f"searched {i}/{len(manifest)} stations, records={len(rows)}")
        time.sleep(args.sleep_sec)

    records = pd.DataFrame(rows)
    summary = pd.DataFrame(station_summaries)
    records_path = index_dir / "nied_kiknet_record_candidates.csv"
    summary_path = index_dir / "nied_kiknet_search_summary.csv"
    records.to_csv(records_path, index=False)
    summary.to_csv(summary_path, index=False)

    diagnostics = {
        "manifest": str(args.manifest),
        "records": str(records_path),
        "summary": str(summary_path),
        "stations_requested": int(len(manifest)),
        "stations_with_records": int((summary["records_kept"] > 0).sum()) if not summary.empty else 0,
        "record_rows": int(len(records)),
        "date_from": args.date_from,
        "date_to": args.date_to,
        "min_magnitude": args.min_magnitude,
        "max_records_per_station": args.max_records_per_station,
        "nied_search_url": SEARCH_URL,
    }
    diagnostics_path = index_dir / "nied_kiknet_search_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
