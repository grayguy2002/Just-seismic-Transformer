#!/usr/bin/env python3
"""Build run025 hard validation sets and non-leaking train-bin weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-12

MORPHOLOGY_CLASSES = [
    "component_path_mismatch",
    "long_duration_morphology",
    "jst_dominant_morphology",
    "jst_morphology_only",
]
MIXED_CLASSES = ["mixed_hybrid_anomaly"]
EMPIRICAL_CONTROL_CLASSES = [
    "empirical_tail_only",
    "empirical_dominant_tail",
    "amplitude_tail",
    "high_frequency_or_instrument",
]
CONSENSUS_CLASSES = ["consensus_severe"]
WEIGHT_TARGET_CLASSES = MORPHOLOGY_CLASSES + MIXED_CLASSES + CONSENSUS_CLASSES

SAFE_BIN_COLUMNS = [
    "selected_phase",
    "mag_bin",
    "depth_bin",
    "distance_bin",
    "trace_channel",
    "station_network_code",
]
GROUP_LEVELS = [
    ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel", "station_network_code"],
    ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel"],
    ["selected_phase", "mag_bin", "distance_bin", "trace_channel"],
    ["selected_phase", "distance_bin", "trace_channel"],
    ["selected_phase", "distance_bin"],
    ["selected_phase"],
]
VALIDATION_COLUMNS = [
    "cache_index",
    "anomaly_class",
    "hardset_role",
    "selected_phase",
    "mag_bin",
    "depth_bin",
    "distance_bin",
    "trace_channel",
    "station_network_code",
    "source_magnitude",
    "source_depth_km",
    "path_ep_distance_deg",
    "phase_travel_sec",
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
    "audit_group",
    "figure",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build run025 hard validation and train-bin sampling recommendation artifacts."
    )
    parser.add_argument(
        "--hybrid-scores",
        default="outputs/hybrid_monitoring_run020_run023/hybrid_scores.csv",
    )
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--cache-prefix", default="pwave_v21")
    parser.add_argument("--train-features", default="outputs/monitoring_baselines_run020_run023/train_features.csv")
    parser.add_argument("--test-features", default="outputs/monitoring_baselines_run020_run023/test_features.csv")
    parser.add_argument("--output-dir", default="outputs/run025_hardsets")
    parser.add_argument("--min-train-bin-size", type=int, default=20)
    parser.add_argument("--min-test-bin-size", type=int, default=5)
    parser.add_argument("--weight-cap", type=float, default=5.0)
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--max-bins-per-level", type=int, default=80)
    return parser.parse_args()


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source_magnitude" in out.columns:
        out["mag_bin"] = pd.cut(
            pd.to_numeric(out["source_magnitude"], errors="coerce"),
            [-np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf],
            labels=["<=3", "3-4", "4-5", "5-6", "6-7", ">7"],
        ).astype(str)
    if "source_depth_km" in out.columns:
        out["depth_bin"] = pd.cut(
            pd.to_numeric(out["source_depth_km"], errors="coerce"),
            [-np.inf, 10.0, 35.0, 70.0, 150.0, 300.0, np.inf],
            labels=["<=10", "10-35", "35-70", "70-150", "150-300", ">300"],
        ).astype(str)
    if "path_ep_distance_deg" in out.columns:
        out["distance_bin"] = pd.cut(
            pd.to_numeric(out["path_ep_distance_deg"], errors="coerce"),
            [-np.inf, 1.0, 3.0, 10.0, 30.0, 60.0, np.inf],
            labels=["<=1", "1-3", "3-10", "10-30", "30-60", ">60"],
        ).astype(str)
    for col in SAFE_BIN_COLUMNS:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


def load_split_meta(data_dir: Path, cache_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir = data_dir / "cache"
    conditions = pd.read_csv(cache_dir / f"{cache_prefix}_conditions.csv")
    if "cache_index" not in conditions.columns:
        conditions.insert(0, "cache_index", np.arange(len(conditions), dtype=np.int64))
    train_idx = np.load(str(cache_dir / "splits" / "training_indices.npy")).astype(np.int64)
    test_idx = np.load(str(cache_dir / "splits" / "testing_indices.npy")).astype(np.int64)
    conditions = add_bins(conditions)
    return conditions, conditions.iloc[train_idx].copy(), conditions.iloc[test_idx].copy()


def read_optional_features(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    df = pd.read_csv(path, usecols=["cache_index"])
    return {
        "path": str(path),
        "exists": True,
        "n": int(len(df)),
        "unique_cache_index": int(df["cache_index"].nunique()),
    }


def class_subset(df: pd.DataFrame, classes: list[str], role: str) -> pd.DataFrame:
    sub = df[df["anomaly_class"].isin(classes)].copy()
    sub["hardset_role"] = role
    sort_cols = [
        c
        for c in [
            "hybrid_monitoring_rank",
            "jst_morphology_rank",
            "empirical_tail_rank",
            "amplitude_tail_rank",
            "high_frequency_rank",
        ]
        if c in sub.columns
    ]
    if sort_cols:
        sub = sub.sort_values(sort_cols, ascending=False)
    cols = [c for c in VALIDATION_COLUMNS if c in sub.columns]
    extra = [c for c in sub.columns if c.startswith("jst_run") and c.endswith("_rank")]
    return sub[cols + extra]


def top_counts(series: pd.Series, n: int = 6) -> str:
    if series.empty:
        return ""
    counts = series.fillna("UNKNOWN").astype(str).value_counts(normalize=True).head(n)
    return "; ".join(f"{idx}:{value:.2f}" for idx, value in counts.items())


def summarize_subset(name: str, df: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {"subset": name, "n": int(len(df))}
    if len(df) == 0:
        return row
    row["classes"] = top_counts(df["anomaly_class"], 12)
    for col in [
        "hybrid_monitoring_rank",
        "jst_morphology_rank",
        "empirical_tail_rank",
        "amplitude_tail_rank",
        "high_frequency_rank",
        "low_frequency_rank",
    ]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_p90"] = float(values.quantile(0.90))
    for col in SAFE_BIN_COLUMNS:
        if col in df.columns:
            row[f"{col}_top"] = top_counts(df[col])
    return row


def key_to_tuple(key: Any) -> tuple[str, ...]:
    if isinstance(key, tuple):
        return tuple(str(x) for x in key)
    return (str(key),)


def grouped_counts(df: pd.DataFrame, cols: list[str]) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    if len(df) == 0:
        return counts
    grouped = df.groupby(cols, dropna=False, observed=True).size()
    for key, value in grouped.items():
        counts[key_to_tuple(key)] = int(value)
    return counts


def build_weight_table(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    target: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_rate = len(target) / max(len(test_meta), 1)
    for level, cols in enumerate(GROUP_LEVELS):
        train_counts = grouped_counts(train_meta, cols)
        test_counts = grouped_counts(test_meta, cols)
        target_counts = grouped_counts(target, cols)
        level_rows: list[dict[str, Any]] = []
        for key, target_n in target_counts.items():
            test_n = test_counts.get(key, 0)
            train_n = train_counts.get(key, 0)
            target_rate = (target_n + args.smooth) / max(test_n + 2.0 * args.smooth, EPS)
            enrichment = target_rate / max(global_rate, EPS)
            multiplier = min(args.weight_cap, max(1.0, math.sqrt(max(enrichment, 0.0))))
            row: dict[str, Any] = {
                "target_role": "morphology_focus",
                "target_classes": ",".join(WEIGHT_TARGET_CLASSES),
                "group_level": level,
                "bin_columns": ",".join(cols),
                "train_n": int(train_n),
                "test_n": int(test_n),
                "target_n": int(target_n),
                "global_target_rate": float(global_rate),
                "target_rate": float(target_n / test_n) if test_n else 0.0,
                "smoothed_target_rate": float(target_rate),
                "enrichment": float(enrichment),
                "recommended_sampling_multiplier": float(multiplier),
                "eligible": bool(train_n >= args.min_train_bin_size and test_n >= args.min_test_bin_size),
            }
            for col, value in zip(cols, key):
                row[col] = value
            level_rows.append(row)
        level_rows.sort(
            key=lambda r: (
                bool(r["eligible"]),
                float(r["recommended_sampling_multiplier"]),
                int(r["target_n"]),
                int(r["train_n"]),
            ),
            reverse=True,
        )
        rows.extend(level_rows[: args.max_bins_per_level])
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in SAFE_BIN_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    order = [
        "target_role",
        "group_level",
        "bin_columns",
        *SAFE_BIN_COLUMNS,
        "train_n",
        "test_n",
        "target_n",
        "global_target_rate",
        "target_rate",
        "smoothed_target_rate",
        "enrichment",
        "recommended_sampling_multiplier",
        "eligible",
        "target_classes",
    ]
    return out[order]


def validate_no_training_leakage(validation: pd.DataFrame, train_meta: pd.DataFrame) -> dict[str, Any]:
    validation_ids = set(pd.to_numeric(validation["cache_index"], errors="coerce").dropna().astype(int).tolist())
    train_ids = set(pd.to_numeric(train_meta["cache_index"], errors="coerce").dropna().astype(int).tolist())
    overlap = sorted(validation_ids & train_ids)
    if overlap:
        raise RuntimeError(f"Hard validation contains training cache_index values: {overlap[:20]}")
    return {
        "hard_validation_n": len(validation_ids),
        "training_n": len(train_ids),
        "train_validation_cache_index_overlap": 0,
    }


def main() -> None:
    args = parse_args()
    hybrid_path = resolve_project_path(args.hybrid_scores)
    data_dir = resolve_project_path(args.data_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, train_meta, test_meta = load_split_meta(data_dir, args.cache_prefix)
    hybrid = add_bins(pd.read_csv(hybrid_path))

    morphology = class_subset(hybrid, MORPHOLOGY_CLASSES, "morphology_hard_validation")
    empirical = class_subset(hybrid, EMPIRICAL_CONTROL_CLASSES, "empirical_tail_control")
    mixed = class_subset(hybrid, MIXED_CLASSES, "mixed_hybrid_validation")
    consensus = class_subset(hybrid, CONSENSUS_CLASSES, "consensus_severe_validation")
    hard_all = pd.concat([morphology, mixed, empirical, consensus], ignore_index=True)
    hard_all = hard_all.drop_duplicates(subset=["cache_index", "hardset_role"])

    subsets = {
        "hard_validation_morphology.csv": morphology,
        "hard_validation_empirical_tail.csv": empirical,
        "hard_validation_mixed_hybrid.csv": mixed,
        "hard_validation_consensus.csv": consensus,
        "hard_validation_all.csv": hard_all,
    }
    for filename, df in subsets.items():
        df.to_csv(output_dir / filename, index=False)

    target = hybrid[hybrid["anomaly_class"].isin(WEIGHT_TARGET_CLASSES)].copy()
    train_weights = build_weight_table(train_meta, add_bins(hybrid), target, args)
    train_weights.to_csv(output_dir / "train_condition_bin_weights.csv", index=False)

    leakage = validate_no_training_leakage(hard_all, train_meta)
    summary_rows = [summarize_subset(filename, df) for filename, df in subsets.items()]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "hardset_summary.csv", index=False)

    report = {
        "hybrid_scores": str(hybrid_path),
        "data_dir": str(data_dir),
        "cache_prefix": args.cache_prefix,
        "n_hybrid_testing_rows": int(len(hybrid)),
        "n_train_rows": int(len(train_meta)),
        "n_test_split_rows": int(len(test_meta)),
        "validation_subsets": {name: int(len(df)) for name, df in subsets.items()},
        "weight_target_classes": WEIGHT_TARGET_CLASSES,
        "empirical_control_classes_not_training_target": EMPIRICAL_CONTROL_CLASSES,
        "safe_training_weight_columns": SAFE_BIN_COLUMNS,
        "group_levels": GROUP_LEVELS,
        "weight_formula": "sqrt(((target_n+smooth)/(test_n+2*smooth))/(target_total/test_total)), clipped to [1, weight_cap]",
        "min_train_bin_size": args.min_train_bin_size,
        "min_test_bin_size": args.min_test_bin_size,
        "weight_cap": args.weight_cap,
        "smooth": args.smooth,
        "no_training_leakage_check": leakage,
        "feature_inputs": {
            "train_features": read_optional_features(resolve_project_path(args.train_features)),
            "test_features": read_optional_features(resolve_project_path(args.test_features)),
        },
        "outputs": {name.replace(".csv", ""): str(output_dir / name) for name in subsets},
    }
    report["outputs"]["train_condition_bin_weights"] = str(output_dir / "train_condition_bin_weights.csv")
    report["outputs"]["hardset_summary"] = str(output_dir / "hardset_summary.csv")

    with (output_dir / "run025_hardset_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(f"wrote {output_dir}")
    print(summary[["subset", "n"]].to_string(index=False))
    if not train_weights.empty:
        cols = [
            "group_level",
            "bin_columns",
            "train_n",
            "test_n",
            "target_n",
            "recommended_sampling_multiplier",
            "eligible",
        ]
        print(train_weights[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
