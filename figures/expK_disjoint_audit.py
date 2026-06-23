"""Exp K: station- and region-disjoint audit for JsT site-effect evidence.

This audit separates two questions that reviewers can conflate:

1. The MLAAPDE train/validation/test split used for Fig. 1 is event-disjoint,
   but may contain the same station in different splits.  We quantify that
   overlap and test whether the cached Fig. 1 station-signature subset contains
   any station-disjoint support.
2. The KiK-net waveform experiment is external to training.  We quantify that
   station/network disjointness and add leave-region/leave-block stability for
   the measured-profile Vs30 association.

The script is deterministic and uses existing caches; it does not rerun JsT
inference.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


F_MIN, F_MAX, N_FREQ_BINS = 0.3, 15.0, 40
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
BINS = [f"hvsr_bin_{i:02d}" for i in range(N_FREQ_BINS)]
BANDS = {
    "0.5-5Hz": [i for i, f in enumerate(FREQ_CENTERS) if 0.5 <= f < 5.0],
    "1-3Hz": [i for i, f in enumerate(FREQ_CENTERS) if 1.0 <= f < 3.0],
    "1-10Hz": [i for i, f in enumerate(FREQ_CENTERS) if 1.0 <= f < 10.0],
    "3-10Hz": [i for i, f in enumerate(FREQ_CENTERS) if 3.0 <= f < 10.0],
}


def _safe_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None, None, int(mask.sum())
    stat = spearmanr(x[mask], y[mask])
    return _safe_float(stat.statistic), _safe_float(stat.pvalue), int(mask.sum())


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return None, None, int(mask.sum())
    stat = pearsonr(x[mask], y[mask])
    return _safe_float(stat.statistic), _safe_float(stat.pvalue), int(mask.sum())


def _rank(values) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def _standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    mu = np.nanmean(matrix, axis=0)
    sigma = np.nanstd(matrix, axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return (matrix - mu) / sigma


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    controls = _standardize(controls)
    design = np.column_stack([np.ones(len(controls)), controls])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def add_station_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["station_id"] = (
        out["station_network_code"].fillna("UNKNOWN").astype(str)
        + "."
        + out["station_code"].fillna("UNKNOWN").astype(str)
    )
    return out


def quantile_blocks(df: pd.DataFrame, q: int = 4, prefix: str = "block") -> pd.Series:
    lat_bin = pd.qcut(df["station_latitude_deg"], q=q, labels=False, duplicates="drop")
    lon_bin = pd.qcut(df["station_longitude_deg"], q=q, labels=False, duplicates="drop")
    return prefix + "_" + lat_bin.astype(str) + "_" + lon_bin.astype(str)


def coarse_japan_region(lat: float, lon: float) -> str:
    """Coarse geographic regions for KiK-net stress tests."""
    lat = float(lat)
    lon = float(lon)
    if lat >= 41.0:
        return "Hokkaido"
    if lat >= 38.0:
        return "Tohoku"
    if lat >= 35.2 and lon >= 138.0:
        return "Kanto_Chubu_east"
    if lat >= 35.2:
        return "Chubu_Kinki_west"
    if lon >= 134.0:
        return "Chugoku_Shikoku_Kyushu_east"
    return "Kyushu_Ryukyu_west"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, idx in BANDS.items():
        out[name] = out[[BINS[i] for i in idx]].mean(axis=1)
    out["arrival_qc"] = (
        (out["arrival_sample"] >= 0)
        & (out["arrival_sample"] < out["trace_n_samples"])
        & (out["left_pad"] <= 800)
        & (out["right_pad"] <= 0)
    )
    out["full_pre_qc"] = (
        (out["arrival_sample"] >= 800)
        & (out["arrival_sample"] < out["trace_n_samples"])
        & (out["left_pad"] == 0)
        & (out["right_pad"] <= 0)
    )
    return out


def fig1_pairwise_subset(station_hvsr: dict, station_ids: list[str]) -> dict:
    curves = []
    labels = []
    for label, station_id in enumerate(station_ids):
        item = station_hvsr.get(station_id, {})
        for curve in item.get("curves", []):
            curves.append(np.asarray(curve, dtype=float))
            labels.append(label)
    if len(curves) < 4 or len(set(labels)) < 2:
        return {
            "n_stations": int(len(station_ids)),
            "n_curves": int(len(curves)),
            "intra": None,
            "inter": None,
            "delta": None,
            "ratio": None,
        }
    x = np.vstack(curves).astype(float)
    x = x - x.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(x, axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    x = x / scale
    corr = x @ x.T
    labels_arr = np.asarray(labels)
    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    same = labels_arr[:, None] == labels_arr[None, :]
    intra_vals = corr[upper & same]
    inter_vals = corr[upper & ~same]
    intra_m = float(np.mean(intra_vals)) if len(intra_vals) else np.nan
    inter_m = float(np.mean(inter_vals)) if len(inter_vals) else np.nan
    return {
        "n_stations": int(len(station_ids)),
        "n_curves": int(len(curves)),
        "intra": _safe_float(intra_m),
        "inter": _safe_float(inter_m),
        "delta": _safe_float(intra_m - inter_m),
        "ratio": _safe_float(intra_m / inter_m) if np.isfinite(inter_m) and inter_m != 0 else None,
    }


def mlaapde_split_audit(data_dir: Path, fig1_cache: Path, min_events: int) -> tuple[pd.DataFrame, dict]:
    conditions = add_station_id(pd.read_csv(data_dir / "cache" / "pwave_v21_conditions.csv"))
    split_dir = data_dir / "cache" / "splits"
    split_idx = {
        split: np.load(split_dir / f"{split}_indices.npy").astype(int)
        for split in ["training", "validation", "testing"]
    }
    split_rows = {split: conditions.iloc[idx].copy() for split, idx in split_idx.items()}
    split_stations = {split: set(rows["station_id"]) for split, rows in split_rows.items()}
    split_networks = {split: set(rows["station_network_code"].astype(str)) for split, rows in split_rows.items()}

    rows = []
    for split, data in split_rows.items():
        counts = data["station_id"].value_counts()
        rows.append(
            {
                "analysis": "mlaapde_split",
                "split": split,
                "records": int(len(data)),
                "stations": int(data["station_id"].nunique()),
                "networks": int(data["station_network_code"].nunique()),
                "stations_ge_min_events": int((counts >= min_events).sum()),
            }
        )
    for a, b in combinations(["training", "validation", "testing"], 2):
        rows.append(
            {
                "analysis": "mlaapde_overlap",
                "split": f"{a}_vs_{b}",
                "records": None,
                "stations": int(len(split_stations[a] & split_stations[b])),
                "networks": int(len(split_networks[a] & split_networks[b])),
                "stations_ge_min_events": None,
            }
        )

    test_counts = split_rows["testing"]["station_id"].value_counts()
    station_disjoint_test = sorted(
        s for s in split_stations["testing"] - split_stations["training"] if test_counts.get(s, 0) >= min_events
    )
    all_station_disjoint_test = sorted(split_stations["testing"] - split_stations["training"])

    fig_summary = {}
    fig_station_ids = []
    if fig1_cache.exists():
        station_hvsr = json.loads(fig1_cache.read_text())
        fig_station_ids = sorted(station_hvsr)
        fig_disjoint = sorted(set(fig_station_ids) - split_stations["training"])
        fig_overlap = sorted(set(fig_station_ids) & split_stations["training"])
        fig_summary = {
            "fig1_cached_stations": int(len(fig_station_ids)),
            "fig1_train_overlap_stations": int(len(fig_overlap)),
            "fig1_station_disjoint_stations": int(len(fig_disjoint)),
            "fig1_all_cached_pairwise": fig1_pairwise_subset(station_hvsr, fig_station_ids),
            "fig1_station_disjoint_pairwise": fig1_pairwise_subset(station_hvsr, fig_disjoint),
        }

    summary = {
        "event_split_note": "MLAAPDE split is event-record based; it is not station-disjoint.",
        "train_stations": int(len(split_stations["training"])),
        "test_stations": int(len(split_stations["testing"])),
        "test_stations_seen_in_training": int(len(split_stations["testing"] & split_stations["training"])),
        "test_stations_seen_in_training_fraction": _safe_float(
            len(split_stations["testing"] & split_stations["training"]) / max(len(split_stations["testing"]), 1)
        ),
        "station_disjoint_test_stations": int(len(all_station_disjoint_test)),
        "station_disjoint_test_records": int(
            split_rows["testing"]["station_id"].isin(all_station_disjoint_test).sum()
        ),
        "station_disjoint_test_stations_ge_min_events": int(len(station_disjoint_test)),
        "station_disjoint_test_records_ge_min_events": int(
            split_rows["testing"]["station_id"].isin(station_disjoint_test).sum()
        ),
        "station_disjoint_test_station_ids": all_station_disjoint_test,
        **fig_summary,
    }
    return pd.DataFrame(rows), summary


def station_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["arrival_qc"]].copy()
    return (
        sub.groupby("station_id", as_index=False)
        .agg(
            score=(metric, "mean"),
            vs30=("vs30", "first"),
            n_records=(metric, "size"),
            n_events=("event_id", "nunique"),
            station_latitude_deg=("station_latitude_deg", "first"),
            station_longitude_deg=("station_longitude_deg", "first"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score", "vs30", "station_latitude_deg", "station_longitude_deg"])
    )


def partial_latlon(df: pd.DataFrame, score_col: str, vs30_col: str = "vs30") -> tuple[float | None, float | None]:
    sub = df[[score_col, vs30_col, "station_latitude_deg", "station_longitude_deg"]].dropna()
    if len(sub) < 8:
        return None, None
    x = _rank(sub[score_col])
    y = _rank(sub[vs30_col])
    cov = sub[["station_latitude_deg", "station_longitude_deg"]].to_numpy(float)
    xr = _residualize(x, cov)
    yr = _residualize(y, cov)
    r, p, _ = _pearson(xr, yr)
    return r, p


def kiknet_external_audit(
    records_path: Path,
    mlaapde_data_dir: Path,
    metric: str,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    records = add_features(pd.read_csv(records_path))
    if metric not in records.columns:
        raise ValueError(f"Metric {metric!r} not available")

    train = add_station_id(pd.read_csv(mlaapde_data_dir / "cache" / "pwave_v21_conditions.csv"))
    train_idx = np.load(mlaapde_data_dir / "cache" / "splits" / "training_indices.npy").astype(int)
    train = train.iloc[train_idx].copy()
    train_stations = set(train["station_id"])
    train_networks = set(train["station_network_code"].astype(str))

    kik_stations = set(records["station_id"].astype(str))
    kik_networks = set(records["station_id"].astype(str).str.split(".").str[0])
    arrival = records[records["arrival_qc"]].copy()
    st_min1 = station_means(records, metric)
    st_min2 = st_min1[st_min1["n_events"] >= 2].copy()

    rows = []
    for label, sub in [
        ("records_arrival_qc", arrival.rename(columns={metric: "score"})),
        ("station_mean_min1", st_min1),
        ("station_mean_min2", st_min2),
    ]:
        rho, p, n = _spearman(sub["score"], sub["vs30"])
        pr, pp = partial_latlon(sub, "score")
        rows.append(
            {
                "analysis": "kiknet_external",
                "check": label,
                "block": "all",
                "N": n,
                "rho": rho,
                "p": p,
                "partial_latlon_r": pr,
                "partial_latlon_p": pp,
            }
        )

    # Leave-one coarse region and leave-one quantile block on station means.
    st = st_min2.copy()
    st["coarse_region"] = [
        coarse_japan_region(lat, lon) for lat, lon in zip(st["station_latitude_deg"], st["station_longitude_deg"])
    ]
    st["spatial_block"] = quantile_blocks(st, q=4, prefix="q")
    region_rhos = []
    block_rhos = []
    for block_col, out_name, store in [
        ("coarse_region", "leave_one_region_out", region_rhos),
        ("spatial_block", "leave_one_spatial_block_out", block_rhos),
    ]:
        for block, group in st.groupby(block_col, sort=True):
            keep = st[st[block_col] != block]
            rho, p, n = _spearman(keep["score"], keep["vs30"])
            pr, pp = partial_latlon(keep, "score")
            if rho is not None:
                store.append(rho)
            rows.append(
                {
                    "analysis": "kiknet_external",
                    "check": out_name,
                    "block": str(block),
                    "N": n,
                    "rho": rho,
                    "p": p,
                    "partial_latlon_r": pr,
                    "partial_latlon_p": pp,
                }
            )

    rng = np.random.default_rng(seed)
    blocks = sorted(st["coarse_region"].unique())
    boot_rhos = []
    for _ in range(n_boot):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        boot = pd.concat([st[st["coarse_region"] == block] for block in chosen], ignore_index=True)
        rho, _, _ = _spearman(boot["score"], boot["vs30"])
        if rho is not None:
            boot_rhos.append(rho)
    boot_rhos = np.asarray(boot_rhos, dtype=float)

    summary = {
        "training_station_disjoint": {
            "kiknet_stations": int(len(kik_stations)),
            "kiknet_training_station_overlap": int(len(kik_stations & train_stations)),
            "kiknet_networks": sorted(kik_networks),
            "kiknet_training_network_overlap": sorted(kik_networks & train_networks),
        },
        "records_arrival_qc": {
            "N": int(len(arrival)),
            "stations": int(arrival["station_id"].nunique()),
            "rho": _spearman(arrival[metric], arrival["vs30"])[0],
            "p": _spearman(arrival[metric], arrival["vs30"])[1],
        },
        "station_mean_min2": {
            "N": int(len(st_min2)),
            "rho": _spearman(st_min2["score"], st_min2["vs30"])[0],
            "p": _spearman(st_min2["score"], st_min2["vs30"])[1],
            "partial_latlon_r": partial_latlon(st_min2, "score")[0],
            "partial_latlon_p": partial_latlon(st_min2, "score")[1],
            "leave_one_region_min": _safe_float(np.min(region_rhos)) if region_rhos else None,
            "leave_one_region_max": _safe_float(np.max(region_rhos)) if region_rhos else None,
            "leave_one_region_median": _safe_float(np.median(region_rhos)) if region_rhos else None,
            "leave_one_block_min": _safe_float(np.min(block_rhos)) if block_rhos else None,
            "leave_one_block_max": _safe_float(np.max(block_rhos)) if block_rhos else None,
            "leave_one_block_median": _safe_float(np.median(block_rhos)) if block_rhos else None,
            "coarse_region_bootstrap_median": _safe_float(np.median(boot_rhos)) if len(boot_rhos) else None,
            "coarse_region_bootstrap_ci95": [
                _safe_float(np.percentile(boot_rhos, 2.5)) if len(boot_rhos) else None,
                _safe_float(np.percentile(boot_rhos, 97.5)) if len(boot_rhos) else None,
            ],
            "n_coarse_regions": int(st["coarse_region"].nunique()),
            "n_spatial_blocks": int(st["spatial_block"].nunique()),
        },
    }
    return pd.DataFrame(rows), summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlaapde-data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--fig1-cache", default="outputs/fig1_cache/station_hvsr.json")
    parser.add_argument("--kiknet-records", default="outputs/expH_kiknet_dense_arrival_qc_events/per_station_event_records.csv")
    parser.add_argument("--output-dir", default="outputs/expK_disjoint_audit")
    parser.add_argument("--metric", default="1-10Hz")
    parser.add_argument("--min-events", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_table, split_summary = mlaapde_split_audit(
        Path(args.mlaapde_data_dir),
        Path(args.fig1_cache),
        args.min_events,
    )
    kik_table, kik_summary = kiknet_external_audit(
        Path(args.kiknet_records),
        Path(args.mlaapde_data_dir),
        args.metric,
        args.n_bootstrap,
        args.seed,
    )

    split_table.to_csv(out_dir / "mlaapde_split_overlap.csv", index=False)
    kik_table.to_csv(out_dir / "kiknet_disjoint_region_controls.csv", index=False)
    summary = {
        "metric": args.metric,
        "min_events": int(args.min_events),
        "mlaapde": split_summary,
        "kiknet": kik_summary,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
