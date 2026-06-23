#!/usr/bin/env python3
"""Empirical monitoring baselines for JsT anomaly score tables."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-8
FEATURE_COLUMNS = [
    "log_peak_counts",
    "log_rms_counts",
    "peak_norm",
    "mean_abs_norm",
    "log_post_rms_counts",
    "log_pre_rms_counts",
    "post_pre_log_rms_ratio",
    "spectral_centroid_hz",
    "spectral_low_frac",
    "spectral_mid_frac",
    "spectral_high_frac",
]
BIN_COLUMNS = ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel"]
BIN_FALLBACKS = [
    ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel"],
    ["selected_phase", "mag_bin", "distance_bin"],
    ["selected_phase", "distance_bin"],
    ["selected_phase"],
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare empirical monitoring baselines with JsT anomaly scores."
    )
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--cache-prefix", default="pwave_v21")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jst-score", action="append", required=True, help="name=path/to/scores.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--nn-k", type=int, default=128)
    parser.add_argument("--nn-chunk-size", type=int, default=128)
    parser.add_argument("--min-bin-size", type=int, default=50)
    return parser.parse_args()


def parse_named_paths(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected name=path, got {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing name in {spec}")
        out[name] = resolve_project_path(path)
    return out


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mag_bin"] = pd.cut(
        out["source_magnitude"],
        [-np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf],
        labels=["<=3", "3-4", "4-5", "5-6", "6-7", ">7"],
    ).astype(str)
    out["depth_bin"] = pd.cut(
        out["source_depth_km"],
        [-np.inf, 10.0, 35.0, 70.0, 150.0, 300.0, np.inf],
        labels=["<=10", "10-35", "35-70", "70-150", "150-300", ">300"],
    ).astype(str)
    out["distance_bin"] = pd.cut(
        out["path_ep_distance_deg"],
        [-np.inf, 1.0, 3.0, 10.0, 30.0, 60.0, np.inf],
        labels=["<=1", "1-3", "3-10", "10-30", "30-60", ">60"],
    ).astype(str)
    for col in ["selected_phase", "trace_channel", "station_network_code", "station_code"]:
        if col in out.columns:
            out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


def load_cache(data_dir: Path, cache_prefix: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    cache_dir = data_dir / "cache"
    waveforms = np.load(
        str(cache_dir / f"{cache_prefix}_X_model_20p60_streamnorm_float32.npy"),
        mmap_mode="r",
    )
    conditions = pd.read_csv(cache_dir / f"{cache_prefix}_conditions.csv")
    train_idx = np.load(str(cache_dir / "splits" / "training_indices.npy")).astype(np.int64)
    test_idx = np.load(str(cache_dir / "splits" / "testing_indices.npy")).astype(np.int64)
    conditions = add_bins(conditions)
    return waveforms, conditions, train_idx, test_idx


def extract_features(
    waveforms: np.ndarray,
    conditions: pd.DataFrame,
    indices: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    freqs: np.ndarray | None = None
    low_mask: np.ndarray | None = None
    mid_mask: np.ndarray | None = None
    high_mask: np.ndarray | None = None
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        x = np.asarray(waveforms[batch_idx], dtype=np.float32)
        abs_x = np.abs(x)
        peak_norm = abs_x.reshape(len(batch_idx), -1).max(axis=1)
        mean_abs_norm = abs_x.reshape(len(batch_idx), -1).mean(axis=1)
        rms_norm = np.sqrt(np.mean(x * x, axis=(1, 2)))
        pre = x[:, :, :800]
        post = x[:, :, 800:]
        pre_rms_norm = np.sqrt(np.mean(pre * pre, axis=(1, 2)))
        post_rms_norm = np.sqrt(np.mean(post * post, axis=(1, 2)))
        spec = np.abs(np.fft.rfft(x, axis=-1)).sum(axis=1)
        if freqs is None:
            freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / 40.0)
            low_mask = freqs <= 2.0
            mid_mask = (freqs > 2.0) & (freqs <= 8.0)
            high_mask = freqs > 8.0
        spec_sum = spec.sum(axis=1).clip(EPS)
        spectral_centroid = (spec * freqs[None, :]).sum(axis=1) / spec_sum
        spectral_low_frac = spec[:, low_mask].sum(axis=1) / spec_sum
        spectral_mid_frac = spec[:, mid_mask].sum(axis=1) / spec_sum
        spectral_high_frac = spec[:, high_mask].sum(axis=1) / spec_sum
        scales = pd.to_numeric(conditions.iloc[batch_idx]["normalization_scale"], errors="coerce").to_numpy(float)
        peak_counts = peak_norm * scales
        rms_counts = rms_norm * scales
        pre_rms_counts = pre_rms_norm * scales
        post_rms_counts = post_rms_norm * scales
        for i, cache_index in enumerate(batch_idx):
            rows.append(
                {
                    "cache_index": int(cache_index),
                    "peak_norm": float(peak_norm[i]),
                    "mean_abs_norm": float(mean_abs_norm[i]),
                    "log_peak_counts": float(np.log(peak_counts[i] + EPS)),
                    "log_rms_counts": float(np.log(rms_counts[i] + EPS)),
                    "log_pre_rms_counts": float(np.log(pre_rms_counts[i] + EPS)),
                    "log_post_rms_counts": float(np.log(post_rms_counts[i] + EPS)),
                    "post_pre_log_rms_ratio": float(np.log((post_rms_counts[i] + EPS) / (pre_rms_counts[i] + EPS))),
                    "spectral_centroid_hz": float(spectral_centroid[i]),
                    "spectral_low_frac": float(spectral_low_frac[i]),
                    "spectral_mid_frac": float(spectral_mid_frac[i]),
                    "spectral_high_frac": float(spectral_high_frac[i]),
                }
            )
        print(f"features {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
    return pd.DataFrame(rows)


def feature_stats(values: np.ndarray, fallback_std: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    if fallback_std is not None:
        std = np.where(std < EPS, fallback_std, std)
    std = np.where(std < EPS, 1.0, std)
    return mean, std


def mahalanobis_diag(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (values - mean[None, :]) / std[None, :]
    return np.sqrt(np.nanmean(z * z, axis=1))


def build_group_stats(
    train_meta: pd.DataFrame,
    train_features: pd.DataFrame,
    min_bin_size: int,
) -> tuple[list[dict[tuple[str, ...], dict[str, Any]]], np.ndarray, np.ndarray]:
    train = train_meta[["cache_index", *BIN_COLUMNS]].merge(train_features, on="cache_index", how="inner")
    global_mean, global_std = feature_stats(train[FEATURE_COLUMNS].to_numpy(float))
    all_stats: list[dict[tuple[str, ...], dict[str, Any]]] = []
    for cols in BIN_FALLBACKS:
        stats: dict[tuple[str, ...], dict[str, Any]] = {}
        for key, group in train.groupby(cols, observed=True, dropna=False):
            if len(group) < min_bin_size:
                continue
            if not isinstance(key, tuple):
                key = (str(key),)
            key = tuple(str(x) for x in key)
            values = group[FEATURE_COLUMNS].to_numpy(float)
            mean, std = feature_stats(values, global_std)
            stats[key] = {
                "n": int(len(group)),
                "mean": mean,
                "std": std,
                "log_peak_p95": float(np.quantile(group["log_peak_counts"], 0.95)),
                "log_peak_p99": float(np.quantile(group["log_peak_counts"], 0.99)),
            }
        all_stats.append(stats)
    return all_stats, global_mean, global_std


def condition_bin_scores(
    test_meta: pd.DataFrame,
    test_features: pd.DataFrame,
    group_stats: list[dict[tuple[str, ...], dict[str, Any]]],
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> pd.DataFrame:
    merged = test_meta[["cache_index", *BIN_COLUMNS]].merge(test_features, on="cache_index", how="inner")
    X = merged[FEATURE_COLUMNS].to_numpy(float)
    scores = np.empty(len(merged), dtype=float)
    bin_ns = np.empty(len(merged), dtype=int)
    fallback_levels = np.empty(len(merged), dtype=int)
    p95 = np.empty(len(merged), dtype=float)
    p99 = np.empty(len(merged), dtype=float)
    for i, row in merged.iterrows():
        found: dict[str, Any] | None = None
        found_level = len(BIN_FALLBACKS)
        for level, cols in enumerate(BIN_FALLBACKS):
            key = tuple(str(row[col]) for col in cols)
            stat = group_stats[level].get(key)
            if stat is not None:
                found = stat
                found_level = level
                break
        if found is None:
            mean, std = global_mean, global_std
            bin_ns[i] = 0
            p95[i] = np.nan
            p99[i] = np.nan
        else:
            mean, std = found["mean"], found["std"]
            bin_ns[i] = int(found["n"])
            p95[i] = float(found["log_peak_p95"])
            p99[i] = float(found["log_peak_p99"])
        scores[i] = mahalanobis_diag(X[i:i + 1], mean, std)[0]
        fallback_levels[i] = found_level
    return pd.DataFrame(
        {
            "cache_index": merged["cache_index"].to_numpy(int),
            "condition_bin_empirical_score": scores,
            "condition_bin_n": bin_ns,
            "condition_bin_fallback_level": fallback_levels,
            "condition_bin_log_peak_p95": p95,
            "condition_bin_log_peak_p99": p99,
        }
    )


def categorical_one_hot(train: pd.Series, test: pd.Series, min_count: int = 30) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_s = train.fillna("UNKNOWN").astype(str)
    test_s = test.fillna("UNKNOWN").astype(str)
    counts = train_s.value_counts()
    cats = counts[counts >= min_count].index.tolist()
    cat_to_idx = {cat: i for i, cat in enumerate(cats)}
    train_out = np.zeros((len(train_s), len(cats)), dtype=np.float32)
    test_out = np.zeros((len(test_s), len(cats)), dtype=np.float32)
    for i, value in enumerate(train_s):
        idx = cat_to_idx.get(value)
        if idx is not None:
            train_out[i, idx] = 1.0
    for i, value in enumerate(test_s):
        idx = cat_to_idx.get(value)
        if idx is not None:
            test_out[i, idx] = 1.0
    return train_out, test_out, cats


def condition_matrix(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    def num_frame(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["source_magnitude"] = pd.to_numeric(df["source_magnitude"], errors="coerce")
        out["log1p_depth"] = np.log1p(pd.to_numeric(df["source_depth_km"], errors="coerce").clip(lower=0.0))
        out["log1p_distance_km"] = np.log1p(pd.to_numeric(df["path_ep_distance_km"], errors="coerce").clip(lower=0.0))
        out["phase_travel_sec"] = pd.to_numeric(df["phase_travel_sec"], errors="coerce")
        for col in ["path_azimuth_deg", "path_back_azimuth_deg"]:
            radians = np.deg2rad(pd.to_numeric(df[col], errors="coerce"))
            out[f"{col}_sin"] = np.sin(radians)
            out[f"{col}_cos"] = np.cos(radians)
        out["station_elevation_km"] = pd.to_numeric(df["station_elevation_m"], errors="coerce") / 1000.0
        return out

    train_num = num_frame(train_meta)
    test_num = num_frame(test_meta)
    med = train_num.median(numeric_only=True)
    train_num = train_num.fillna(med)
    test_num = test_num.fillna(med)
    mean = train_num.mean(axis=0)
    std = train_num.std(axis=0).replace(0.0, 1.0)
    train_parts = [((train_num - mean) / std).to_numpy(np.float32)]
    test_parts = [((test_num - mean) / std).to_numpy(np.float32)]
    for col in ["selected_phase", "trace_channel", "station_network_code"]:
        tr, te, _ = categorical_one_hot(train_meta[col], test_meta[col])
        if tr.shape[1] > 0:
            train_parts.append(tr)
            test_parts.append(te)
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def nn_scores(
    train_condition: np.ndarray,
    test_condition: np.ndarray,
    train_features: np.ndarray,
    test_features: np.ndarray,
    global_std: np.ndarray,
    k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, train_condition.shape[0])
    train_norm = np.sum(train_condition * train_condition, axis=1)
    scores = np.empty(test_condition.shape[0], dtype=float)
    mean_dists = np.empty(test_condition.shape[0], dtype=float)
    for start in range(0, test_condition.shape[0], chunk_size):
        end = min(start + chunk_size, test_condition.shape[0])
        chunk = test_condition[start:end]
        dist = np.sum(chunk * chunk, axis=1)[:, None] + train_norm[None, :] - 2.0 * chunk @ train_condition.T
        dist = np.maximum(dist, 0.0)
        nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        for local_i, neighbors in enumerate(nn_idx):
            vals = train_features[neighbors]
            mean, std = feature_stats(vals, global_std)
            scores[start + local_i] = mahalanobis_diag(test_features[start + local_i:start + local_i + 1], mean, std)[0]
            mean_dists[start + local_i] = float(np.sqrt(dist[local_i, neighbors]).mean())
        print(f"condition nn {end}/{test_condition.shape[0]}", flush=True)
    return scores, mean_dists


def auc_score(scores: pd.Series, labels: pd.Series) -> float | None:
    s = pd.to_numeric(scores, errors="coerce")
    y = labels.astype(bool)
    mask = np.isfinite(s) & y.notna()
    s = s[mask]
    y = y[mask]
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = s.rank(method="average")
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def top_overlap(a: pd.Series, b: pd.Series, k: int) -> float | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < k:
        return None
    aa = set(a[mask].nlargest(k).index.tolist())
    bb = set(b[mask].nlargest(k).index.tolist())
    return len(aa & bb) / k


def merge_jst_scores(test_meta: pd.DataFrame, named_paths: dict[str, Path]) -> pd.DataFrame:
    out = test_meta.copy()
    metadata_cols = [
        "cache_index",
        "split_index",
        "residual_travel_sec",
        "combined_raw_weighted",
        "combined_z",
        "peak_log_ratio_abs",
        "time_rel_l2_min",
        "envelope_rel_l2_min",
        "spectral_rel_l2_min",
    ]
    for name, path in named_paths.items():
        df = pd.read_csv(path)
        keep = [c for c in metadata_cols if c in df.columns]
        rename = {c: f"jst_{name}_{c}" for c in keep if c != "cache_index"}
        out = out.merge(df[keep].rename(columns=rename), on="cache_index", how="left")
    return out


def summarize_methods(scores: pd.DataFrame, methods: list[str], labels: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for method in methods:
        values = pd.to_numeric(scores[method], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        summary_rows.append(
            {
                "method": method,
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=0)),
                "p50": float(finite.quantile(0.50)),
                "p90": float(finite.quantile(0.90)),
                "p95": float(finite.quantile(0.95)),
                "p99": float(finite.quantile(0.99)),
            }
        )
        for label_name, label_values in labels.items():
            auc_rows.append({"method": method, "label": label_name, "auc": auc_score(values, label_values)})
    for i, left in enumerate(methods):
        for right in methods[i + 1:]:
            for k in [50, 100, 250]:
                overlap_rows.append({"left": left, "right": right, "k": k, "top_overlap": top_overlap(scores[left], scores[right], k)})
    return pd.DataFrame(summary_rows), pd.DataFrame(auc_rows), pd.DataFrame(overlap_rows)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        v = value.item()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return value


def main() -> None:
    args = parse_args()
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jst_paths = parse_named_paths(args.jst_score)

    waveforms, conditions, train_idx, _ = load_cache(data_dir, args.cache_prefix)
    first_jst = pd.read_csv(next(iter(jst_paths.values())))
    test_idx = first_jst["cache_index"].to_numpy(np.int64)
    train_meta = conditions.iloc[train_idx].copy()
    test_meta = conditions.iloc[test_idx].copy()
    train_meta["cache_index"] = train_idx
    test_meta["cache_index"] = test_idx

    print(f"training samples={len(train_idx)} testing scored samples={len(test_idx)}")
    train_features = extract_features(waveforms, conditions, train_idx, args.batch_size)
    test_features = extract_features(waveforms, conditions, test_idx, args.batch_size)

    train_X = train_features[FEATURE_COLUMNS].to_numpy(float)
    test_X = test_features[FEATURE_COLUMNS].to_numpy(float)
    global_mean, global_std = feature_stats(train_X)
    baseline = test_features[["cache_index", *FEATURE_COLUMNS]].copy()
    baseline["global_empirical_score"] = mahalanobis_diag(test_X, global_mean, global_std)

    group_stats, _, _ = build_group_stats(train_meta, train_features, args.min_bin_size)
    baseline = baseline.merge(
        condition_bin_scores(test_meta, test_features, group_stats, global_mean, global_std),
        on="cache_index",
        how="left",
    )

    train_cond, test_cond = condition_matrix(train_meta, test_meta)
    nn_score, nn_mean_dist = nn_scores(
        train_cond,
        test_cond,
        train_X,
        test_X,
        global_std,
        args.nn_k,
        args.nn_chunk_size,
    )
    baseline["condition_nn_empirical_score"] = nn_score
    baseline["condition_nn_mean_distance"] = nn_mean_dist

    scores = merge_jst_scores(test_meta, jst_paths)
    scores = scores.merge(baseline, on="cache_index", how="left")

    train_log_peak = train_features["log_peak_counts"]
    labels = {
        "global_peak_p95": scores["log_peak_counts"] >= float(train_log_peak.quantile(0.95)),
        "global_peak_p99": scores["log_peak_counts"] >= float(train_log_peak.quantile(0.99)),
        "condition_bin_peak_p95": scores["log_peak_counts"] >= scores["condition_bin_log_peak_p95"],
        "condition_bin_peak_p99": scores["log_peak_counts"] >= scores["condition_bin_log_peak_p99"],
    }
    residual_cols = [c for c in scores.columns if c.endswith("_residual_travel_sec")]
    if residual_cols:
        residual = pd.to_numeric(scores[residual_cols[0]], errors="coerce").abs()
        labels["abs_residual_top05"] = residual >= residual.quantile(0.95)
        labels["abs_residual_top01"] = residual >= residual.quantile(0.99)

    method_cols = ["global_empirical_score", "condition_bin_empirical_score", "condition_nn_empirical_score"]
    for name in jst_paths:
        for metric in ["combined_raw_weighted", "combined_z", "peak_log_ratio_abs"]:
            col = f"jst_{name}_{metric}"
            if col in scores.columns:
                method_cols.append(col)

    method_summary, pseudo_label_auc, top_overlap_df = summarize_methods(scores, method_cols, labels)
    corr = scores[method_cols].corr(method="spearman")

    train_features.to_csv(output_dir / "train_features.csv", index=False)
    test_features.to_csv(output_dir / "test_features.csv", index=False)
    scores.to_csv(output_dir / "baseline_scores_testing.csv", index=False)
    method_summary.to_csv(output_dir / "method_summary.csv", index=False)
    pseudo_label_auc.to_csv(output_dir / "pseudo_label_auc.csv", index=False)
    top_overlap_df.to_csv(output_dir / "top_overlap.csv", index=False)
    corr.to_csv(output_dir / "score_spearman_corr.csv")

    report = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "feature_columns": FEATURE_COLUMNS,
        "method_columns": method_cols,
        "outputs": {
            "baseline_scores_testing": str(output_dir / "baseline_scores_testing.csv"),
            "method_summary": str(output_dir / "method_summary.csv"),
            "pseudo_label_auc": str(output_dir / "pseudo_label_auc.csv"),
            "top_overlap": str(output_dir / "top_overlap.csv"),
            "score_spearman_corr": str(output_dir / "score_spearman_corr.csv"),
        },
    }
    (output_dir / "monitoring_baseline_report.json").write_text(json.dumps(sanitize_json(report), indent=2))

    print("Method summary:")
    print(method_summary.to_string(index=False))
    print("\nPseudo-label AUC:")
    print(pseudo_label_auc.to_string(index=False))
    print(f"\nWrote monitoring baselines to {output_dir}")


if __name__ == "__main__":
    main()
