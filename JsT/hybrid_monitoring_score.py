#!/usr/bin/env python3
"""Build morphology-aware hybrid monitoring scores from existing JsT and empirical outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EMPIRICAL_SCORE_COLUMNS = [
    "global_empirical_score",
    "condition_bin_empirical_score",
    "condition_nn_empirical_score",
]
META_COLUMNS = [
    "cache_index",
    "event_id",
    "phase_id",
    "waves_id",
    "trace_name",
    "station_network_code",
    "station_code",
    "selected_phase",
    "trace_channel",
    "source_magnitude",
    "source_depth_km",
    "path_ep_distance_deg",
    "path_ep_distance_km",
    "phase_travel_sec",
    "mag_bin",
    "depth_bin",
    "distance_bin",
]
FEATURE_COLUMNS = [
    "log_peak_counts",
    "log_rms_counts",
    "log_post_rms_counts",
    "log_pre_rms_counts",
    "post_pre_log_rms_ratio",
    "spectral_centroid_hz",
    "spectral_low_frac",
    "spectral_mid_frac",
    "spectral_high_frac",
]
JST_MORPHOLOGY_METRICS = [
    "time_rel_l2_min",
    "envelope_rel_l2_min",
    "spectral_rel_l2_min",
]
JST_ALL_METRICS = [
    *JST_MORPHOLOGY_METRICS,
    "peak_log_ratio_abs",
    "combined_raw_weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create morphology-aware hybrid monitoring score tables.")
    parser.add_argument("--scores", required=True, help="baseline_scores_testing.csv from monitoring_baselines.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jst-run", action="append", default=None)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--top-pct", type=float, default=0.965)
    parser.add_argument("--mid-pct", type=float, default=0.90)
    parser.add_argument("--image-audit-manifest", default="")
    return parser.parse_args()


def zrank(series: pd.Series, *, ascending: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(method="average", pct=True, ascending=ascending)
    return ranks.fillna(0.0)


def mean_existing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.zeros(len(df)), index=df.index)
    return df[existing].mean(axis=1)


def safe_mean(values: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def safe_quantile(values: pd.Series | np.ndarray, q: float) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def add_score_axes(df: pd.DataFrame, jst_runs: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in EMPIRICAL_SCORE_COLUMNS:
        if col in out.columns:
            out[f"{col}_rank"] = zrank(out[col])
    out["empirical_tail_rank"] = mean_existing(out, [f"{c}_rank" for c in EMPIRICAL_SCORE_COLUMNS])

    amplitude_parts = []
    for col in ["log_peak_counts", "log_rms_counts", "log_post_rms_counts", "post_pre_log_rms_ratio"]:
        if col in out.columns:
            rank_col = f"{col}_rank"
            out[rank_col] = zrank(out[col])
            amplitude_parts.append(rank_col)
    if "condition_bin_empirical_score_rank" in out.columns:
        amplitude_parts.append("condition_bin_empirical_score_rank")
    out["amplitude_tail_rank"] = mean_existing(out, amplitude_parts)

    high_freq_parts = []
    for col in ["spectral_centroid_hz", "spectral_high_frac"]:
        if col in out.columns:
            rank_col = f"{col}_rank"
            out[rank_col] = zrank(out[col])
            high_freq_parts.append(rank_col)
    if "spectral_low_frac" in out.columns:
        out["spectral_low_frac_inverse_rank"] = zrank(out["spectral_low_frac"], ascending=False)
        high_freq_parts.append("spectral_low_frac_inverse_rank")
    out["high_frequency_rank"] = mean_existing(out, high_freq_parts)

    low_freq_parts = []
    if "spectral_low_frac" in out.columns:
        out["spectral_low_frac_rank"] = zrank(out["spectral_low_frac"])
        low_freq_parts.append("spectral_low_frac_rank")
    if "spectral_centroid_hz" in out.columns:
        out["spectral_centroid_low_rank"] = zrank(out["spectral_centroid_hz"], ascending=False)
        low_freq_parts.append("spectral_centroid_low_rank")
    out["low_frequency_rank"] = mean_existing(out, low_freq_parts)

    morphology_consensus_cols = []
    combined_cols = []
    peak_cols = []
    for run in jst_runs:
        metric_rank_cols = []
        for metric in JST_ALL_METRICS:
            col = f"jst_{run}_{metric}"
            if col not in out.columns:
                continue
            rank_col = f"jst_{run}_{metric}_rank"
            out[rank_col] = zrank(out[col])
            if metric in JST_MORPHOLOGY_METRICS:
                metric_rank_cols.append(rank_col)
            elif metric == "combined_raw_weighted":
                combined_cols.append(rank_col)
            elif metric == "peak_log_ratio_abs":
                peak_cols.append(rank_col)
        if metric_rank_cols:
            run_morph_col = f"jst_{run}_morphology_rank"
            out[run_morph_col] = mean_existing(out, metric_rank_cols)
            morphology_consensus_cols.append(run_morph_col)
    out["jst_morphology_rank"] = mean_existing(out, morphology_consensus_cols)
    out["jst_combined_rank"] = mean_existing(out, combined_cols)
    out["jst_peak_mismatch_rank"] = mean_existing(out, peak_cols)

    envelope_cols = [f"jst_{run}_envelope_rel_l2_min_rank" for run in jst_runs]
    time_cols = [f"jst_{run}_time_rel_l2_min_rank" for run in jst_runs]
    spectral_cols = [f"jst_{run}_spectral_rel_l2_min_rank" for run in jst_runs]
    out["jst_envelope_mismatch_rank"] = mean_existing(out, envelope_cols)
    out["jst_time_mismatch_rank"] = mean_existing(out, time_cols)
    out["jst_spectral_mismatch_rank"] = mean_existing(out, spectral_cols)

    out["hybrid_monitoring_score"] = (
        0.40 * out["jst_morphology_rank"]
        + 0.35 * out["empirical_tail_rank"]
        + 0.15 * out["amplitude_tail_rank"]
        + 0.10 * out["high_frequency_rank"]
    )
    out["hybrid_monitoring_rank"] = zrank(out["hybrid_monitoring_score"])
    out["jst_empirical_disagreement"] = out["jst_morphology_rank"] - out["empirical_tail_rank"]
    out["absolute_disagreement"] = out["jst_empirical_disagreement"].abs()
    return out


def assign_class(row: pd.Series, top_pct: float, mid_pct: float) -> str:
    jst = float(row.get("jst_morphology_rank", 0.0))
    emp = float(row.get("empirical_tail_rank", 0.0))
    amp = float(row.get("amplitude_tail_rank", 0.0))
    high = float(row.get("high_frequency_rank", 0.0))
    low = float(row.get("low_frequency_rank", 0.0))
    env = float(row.get("jst_envelope_mismatch_rank", 0.0))
    time = float(row.get("jst_time_mismatch_rank", 0.0))
    spec = float(row.get("jst_spectral_mismatch_rank", 0.0))
    post = float(row.get("post_pre_log_rms_ratio_rank", 0.0))
    hybrid = float(row.get("hybrid_monitoring_rank", 0.0))

    if jst >= top_pct and emp >= top_pct:
        return "consensus_severe"
    if jst >= top_pct and emp < mid_pct:
        if env >= top_pct and (post >= 0.75 or low >= 0.80):
            return "long_duration_morphology"
        if spec >= top_pct and high >= 0.80:
            return "conditional_spectral_morphology"
        if time >= top_pct or env >= mid_pct:
            return "component_path_mismatch"
        return "jst_morphology_only"
    if emp >= top_pct and jst < mid_pct:
        if high >= top_pct:
            return "high_frequency_or_instrument"
        if amp >= top_pct or post >= top_pct:
            return "amplitude_tail"
        if low >= top_pct and env >= 0.80:
            return "low_frequency_tail"
        return "empirical_tail_only"
    if jst >= top_pct:
        return "jst_dominant_morphology"
    if emp >= top_pct:
        return "empirical_dominant_tail"
    if high >= top_pct:
        return "high_frequency_watch"
    if amp >= top_pct:
        return "amplitude_tail_watch"
    if hybrid >= top_pct and (jst >= 0.80 or emp >= 0.90):
        return "mixed_hybrid_anomaly"
    return "background"


def summarize_class(df: pd.DataFrame, label: str, score_cols: list[str]) -> dict[str, Any]:
    sub = df[df["anomaly_class"] == label]
    row: dict[str, Any] = {"anomaly_class": label, "n": int(len(sub))}
    for col in score_cols:
        if col in sub.columns:
            row[f"{col}_mean"] = safe_mean(sub[col])
            row[f"{col}_p90"] = safe_quantile(sub[col], 0.90)
    for col in ["source_magnitude", "source_depth_km", "path_ep_distance_deg", "phase_travel_sec"]:
        if col in sub.columns:
            row[f"{col}_mean"] = safe_mean(sub[col])
    for col in ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel", "station_network_code"]:
        if col in sub.columns and len(sub):
            vc = sub[col].fillna("UNKNOWN").astype(str).value_counts(normalize=True).head(8)
            row[f"{col}_top"] = "; ".join(f"{k}:{v:.2f}" for k, v in vc.items())
    return row


def write_top(df: pd.DataFrame, path: Path, sort_col: str, top_n: int, *, query: str | None = None) -> None:
    sub = df.query(query).copy() if query else df.copy()
    if sort_col in sub.columns:
        sub = sub.sort_values(sort_col, ascending=False)
    cols = [
        *[c for c in META_COLUMNS if c in sub.columns],
        *[c for c in FEATURE_COLUMNS if c in sub.columns],
        "anomaly_class",
        "hybrid_monitoring_rank",
        "hybrid_monitoring_score",
        "jst_morphology_rank",
        "empirical_tail_rank",
        "amplitude_tail_rank",
        "high_frequency_rank",
        "low_frequency_rank",
        "jst_envelope_mismatch_rank",
        "jst_time_mismatch_rank",
        "jst_spectral_mismatch_rank",
        "jst_empirical_disagreement",
        "absolute_disagreement",
    ]
    cols = [c for c in cols if c in sub.columns]
    sub[cols].head(top_n).to_csv(path, index=False)


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
    jst_runs = args.jst_run or ["run020_last", "run023_last"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    df = add_score_axes(df, jst_runs)
    df["anomaly_class"] = df.apply(assign_class, axis=1, top_pct=args.top_pct, mid_pct=args.mid_pct)

    if args.image_audit_manifest:
        manifest = pd.read_csv(args.image_audit_manifest)
        audited = manifest[["cache_index", "audit_group", "figure"]].copy()
        df = df.merge(audited, on="cache_index", how="left")

    score_cols = [
        "hybrid_monitoring_rank",
        "jst_morphology_rank",
        "empirical_tail_rank",
        "amplitude_tail_rank",
        "high_frequency_rank",
        "low_frequency_rank",
        "jst_envelope_mismatch_rank",
        "jst_time_mismatch_rank",
        "jst_spectral_mismatch_rank",
        "jst_empirical_disagreement",
    ]
    class_summary = pd.DataFrame([
        summarize_class(df, label, score_cols)
        for label in df["anomaly_class"].value_counts().index.tolist()
    ])

    df.to_csv(output_dir / "hybrid_scores.csv", index=False)
    class_summary.to_csv(output_dir / "class_summary.csv", index=False)
    write_top(df, output_dir / "top_hybrid_monitoring.csv", "hybrid_monitoring_rank", args.top_n)
    write_top(df, output_dir / "top_jst_morphology.csv", "jst_morphology_rank", args.top_n)
    write_top(df, output_dir / "top_empirical_tail.csv", "empirical_tail_rank", args.top_n)
    write_top(df, output_dir / "top_amplitude_tail.csv", "amplitude_tail_rank", args.top_n)
    write_top(df, output_dir / "top_high_frequency.csv", "high_frequency_rank", args.top_n)
    write_top(df, output_dir / "top_disagreement_jst_gt_empirical.csv", "jst_empirical_disagreement", args.top_n)
    write_top(df, output_dir / "top_disagreement_empirical_gt_jst.csv", "jst_empirical_disagreement", args.top_n, query="jst_empirical_disagreement < 0")
    for label in df["anomaly_class"].value_counts().index.tolist():
        safe = label.replace("/", "_")
        write_top(df, output_dir / f"top_class_{safe}.csv", "hybrid_monitoring_rank", args.top_n, query=f"anomaly_class == '{label}'")

    report = {
        "scores": args.scores,
        "jst_runs": jst_runs,
        "top_pct": args.top_pct,
        "mid_pct": args.mid_pct,
        "n": int(len(df)),
        "class_counts": df["anomaly_class"].value_counts().to_dict(),
        "outputs": {
            "hybrid_scores": str(output_dir / "hybrid_scores.csv"),
            "class_summary": str(output_dir / "class_summary.csv"),
        },
    }
    (output_dir / "hybrid_monitoring_report.json").write_text(json.dumps(sanitize_json(report), indent=2))

    pd.set_option("display.max_columns", 120)
    show_cols = [
        "anomaly_class",
        "n",
        "hybrid_monitoring_rank_mean",
        "jst_morphology_rank_mean",
        "empirical_tail_rank_mean",
        "amplitude_tail_rank_mean",
        "high_frequency_rank_mean",
        "source_magnitude_mean",
        "path_ep_distance_deg_mean",
        "selected_phase_top",
        "distance_bin_top",
    ]
    show_cols = [c for c in show_cols if c in class_summary.columns]
    print("Class summary:")
    print(class_summary[show_cols].to_string(index=False))
    print(f"\nWrote hybrid monitoring outputs to {output_dir}")


if __name__ == "__main__":
    main()
