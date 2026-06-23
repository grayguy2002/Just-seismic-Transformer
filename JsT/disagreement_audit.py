#!/usr/bin/env python3
"""Audit disagreements between JsT and empirical monitoring anomaly scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    "jst_run020_last_residual_travel_sec",
    "mag_bin",
    "depth_bin",
    "distance_bin",
]
FEATURE_COLUMNS = [
    "log_peak_counts",
    "log_rms_counts",
    "post_pre_log_rms_ratio",
    "spectral_centroid_hz",
    "spectral_low_frac",
    "spectral_mid_frac",
    "spectral_high_frac",
]
EMPIRICAL_SCORE_COLUMNS = [
    "global_empirical_score",
    "condition_bin_empirical_score",
    "condition_nn_empirical_score",
]
JST_COMPONENTS = ["time_rel_l2_min", "envelope_rel_l2_min", "spectral_rel_l2_min", "peak_log_ratio_abs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit JsT/empirical anomaly disagreements.")
    parser.add_argument("--scores", required=True, help="baseline_scores_testing.csv from monitoring_baselines.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--jst-run", action="append", default=["run020_last", "run023_last"])
    return parser.parse_args()


def safe_mean(values: pd.Series) -> float | None:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def safe_quantile(values: pd.Series, q: float) -> float | None:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.quantile(arr, q))


def zrank(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(method="average", pct=True)
    return ranks.fillna(0.0)


def add_composite_scores(df: pd.DataFrame, jst_runs: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = df.copy()
    out["empirical_composite_rank"] = sum(zrank(out[col]) for col in EMPIRICAL_SCORE_COLUMNS) / len(EMPIRICAL_SCORE_COLUMNS)
    jst_score_cols: list[str] = []
    morphology_cols: list[str] = []
    for run in jst_runs:
        combined = f"jst_{run}_combined_raw_weighted"
        if combined in out.columns:
            out[f"jst_{run}_combined_rank"] = zrank(out[combined])
            jst_score_cols.append(f"jst_{run}_combined_rank")
        morph_parts = []
        for metric in ["time_rel_l2_min", "envelope_rel_l2_min", "spectral_rel_l2_min"]:
            col = f"jst_{run}_{metric}"
            if col in out.columns:
                rank_col = f"jst_{run}_{metric}_rank"
                out[rank_col] = zrank(out[col])
                morph_parts.append(out[rank_col])
        if morph_parts:
            morph_col = f"jst_{run}_morphology_rank"
            out[morph_col] = sum(morph_parts) / len(morph_parts)
            morphology_cols.append(morph_col)
    if jst_score_cols:
        out["jst_combined_consensus_rank"] = sum(out[col] for col in jst_score_cols) / len(jst_score_cols)
    if morphology_cols:
        out["jst_morphology_consensus_rank"] = sum(out[col] for col in morphology_cols) / len(morphology_cols)
    return out, jst_score_cols, morphology_cols


def top_set(df: pd.DataFrame, score_col: str, top_n: int) -> set[int]:
    return set(df.nlargest(top_n, score_col).index.tolist())


def label_sets(df: pd.DataFrame, jst_col: str, empirical_col: str, top_n: int) -> dict[str, set[int]]:
    jst_top = top_set(df, jst_col, top_n)
    emp_top = top_set(df, empirical_col, top_n)
    return {
        "jst_only": jst_top - emp_top,
        "empirical_only": emp_top - jst_top,
        "consensus": jst_top & emp_top,
        "jst_top": jst_top,
        "empirical_top": emp_top,
    }


def summarize_subset(df: pd.DataFrame, indices: set[int], label: str, score_cols: list[str]) -> dict[str, Any]:
    sub = df.loc[list(indices)] if indices else df.iloc[[]]
    row: dict[str, Any] = {"subset": label, "n": int(len(sub))}
    for col in [*FEATURE_COLUMNS, *score_cols, *EMPIRICAL_SCORE_COLUMNS]:
        if col in sub.columns:
            row[f"{col}_mean"] = safe_mean(sub[col])
            row[f"{col}_p90"] = safe_quantile(sub[col], 0.90)
    for col in ["source_magnitude", "source_depth_km", "path_ep_distance_deg", "phase_travel_sec"]:
        if col in sub.columns:
            row[f"{col}_mean"] = safe_mean(sub[col])
    residual_col = "jst_run020_last_residual_travel_sec"
    if residual_col in sub.columns:
        row["abs_residual_travel_sec_mean"] = safe_mean(sub[residual_col].abs())
        row["abs_residual_travel_sec_p90"] = safe_quantile(sub[residual_col].abs(), 0.90)
    for col in ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel", "station_network_code"]:
        if col in sub.columns and len(sub):
            vc = sub[col].fillna("UNKNOWN").astype(str).value_counts(normalize=True).head(8)
            row[f"{col}_top"] = "; ".join(f"{k}:{v:.2f}" for k, v in vc.items())
    return row


def write_top_table(df: pd.DataFrame, indices: set[int], score_cols: list[str], path: Path, top_n: int) -> None:
    columns = [c for c in [*META_COLUMNS, *FEATURE_COLUMNS, *EMPIRICAL_SCORE_COLUMNS, *score_cols] if c in df.columns]
    if indices:
        sub = df.loc[list(indices)].copy()
        sort_cols = [score_cols[0]] if score_cols else []
        if sort_cols:
            sub = sub.sort_values(sort_cols[0], ascending=False)
        sub[columns].head(top_n).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=columns).to_csv(path, index=False)


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
    scores_path = Path(args.scores)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_path)
    df, jst_score_cols, morphology_cols = add_composite_scores(df, args.jst_run)
    score_cols = [
        "empirical_composite_rank",
        "jst_combined_consensus_rank",
        "jst_morphology_consensus_rank",
        *jst_score_cols,
        *morphology_cols,
    ]
    score_cols = [c for c in score_cols if c in df.columns]

    audit_specs = []
    if "jst_combined_consensus_rank" in df.columns:
        audit_specs.append(("jst_combined_vs_empirical", "jst_combined_consensus_rank", "empirical_composite_rank"))
    if "jst_morphology_consensus_rank" in df.columns:
        audit_specs.append(("jst_morphology_vs_empirical", "jst_morphology_consensus_rank", "empirical_composite_rank"))
    for run in args.jst_run:
        col = f"jst_{run}_combined_rank"
        if col in df.columns:
            audit_specs.append((f"{run}_combined_vs_empirical", col, "empirical_composite_rank"))
        morph_col = f"jst_{run}_morphology_rank"
        if morph_col in df.columns:
            audit_specs.append((f"{run}_morphology_vs_empirical", morph_col, "empirical_composite_rank"))

    summary_rows: list[dict[str, Any]] = []
    for audit_name, jst_col, emp_col in audit_specs:
        sets = label_sets(df, jst_col, emp_col, args.top_n)
        for label, indices in sets.items():
            row = summarize_subset(df, indices, f"{audit_name}:{label}", [jst_col, emp_col, *score_cols])
            row["audit"] = audit_name
            row["comparison_subset"] = label
            summary_rows.append(row)
            if label in {"jst_only", "empirical_only", "consensus"}:
                write_top_table(
                    df,
                    indices,
                    [jst_col, emp_col, *score_cols],
                    output_dir / f"top_{audit_name}_{label}.csv",
                    args.top_n,
                )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "disagreement_summary.csv", index=False)
    df.to_csv(output_dir / "scores_with_disagreement_ranks.csv", index=False)

    report = {
        "top_n": args.top_n,
        "jst_runs": args.jst_run,
        "audit_specs": [list(x) for x in audit_specs],
        "score_columns": score_cols,
        "outputs": {
            "summary": str(output_dir / "disagreement_summary.csv"),
            "scores_with_disagreement_ranks": str(output_dir / "scores_with_disagreement_ranks.csv"),
        },
    }
    (output_dir / "disagreement_report.json").write_text(json.dumps(sanitize_json(report), indent=2))

    pd.set_option("display.max_columns", 120)
    print("Disagreement summary:")
    show_cols = [
        "audit",
        "comparison_subset",
        "n",
        "log_peak_counts_mean",
        "post_pre_log_rms_ratio_mean",
        "spectral_centroid_hz_mean",
        "source_magnitude_mean",
        "path_ep_distance_deg_mean",
        "abs_residual_travel_sec_mean",
        "selected_phase_top",
        "distance_bin_top",
        "mag_bin_top",
    ]
    show_cols = [c for c in show_cols if c in summary.columns]
    print(summary[show_cols].to_string(index=False))
    print(f"\nWrote disagreement audit to {output_dir}")


if __name__ == "__main__":
    main()
