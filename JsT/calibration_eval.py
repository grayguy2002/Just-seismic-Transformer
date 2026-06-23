#!/usr/bin/env python3
"""Physical and tail calibration audit for JsT anomaly/generation score tables."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-8
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
KEY_METRICS = [
    "combined_raw_weighted",
    "time_rel_l2_min",
    "envelope_rel_l2_min",
    "spectral_rel_l2_min",
    "peak_log_ratio_abs",
]
GROUP_COLUMNS = ["selected_phase", "mag_bin", "depth_bin", "distance_bin", "trace_channel", "station_network_code"]
CONTROL_QUANTILES = [0.90, 0.95, 0.99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit JsT generated distribution calibration from anomaly_exp1 score CSVs."
    )
    parser.add_argument(
        "--score",
        action="append",
        required=True,
        help="Run score table as name=path/to/scores.csv. Can be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_score_specs(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--score must be name=path, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Missing run name in --score {spec}")
        parsed[name] = Path(path)
    return parsed


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    values = pd.to_numeric(df[col], errors="coerce")
    return values[np.isfinite(values)]


def safe_mean(values: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def safe_std(values: pd.Series | np.ndarray) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.std(ddof=0))


def quantile_dict(values: pd.Series | np.ndarray, prefix: str) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_q{int(q * 100):02d}": None for q in QUANTILES}
    return {f"{prefix}_q{int(q * 100):02d}": float(np.quantile(arr, q)) for q in QUANTILES}


def ks_distance(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = np.sort(x[np.isfinite(x)])
    y = np.sort(y[np.isfinite(y)])
    if x.size == 0 or y.size == 0:
        return None
    grid = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def wasserstein_sorted(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None
    qs = np.linspace(0.0, 1.0, max(x.size, y.size))
    return float(np.mean(np.abs(np.quantile(x, qs) - np.quantile(y, qs))))


def safe_corr(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return None
    xa = xa[mask]
    ya = ya[mask]
    if xa.std() < EPS or ya.std() < EPS:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def linear_slope(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float | None:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < 3:
        return None
    xa = xa[mask]
    ya = ya[mask]
    if xa.std() < EPS:
        return None
    X = np.stack([np.ones_like(xa), xa], axis=1)
    coef, *_ = np.linalg.lstsq(X, ya, rcond=None)
    return float(coef[1])


def prepare_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mag_bin"] = pd.cut(
        out["source_magnitude"],
        [-np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf],
        labels=["<=3", "3-4", "4-5", "5-6", "6-7", ">7"],
    )
    out["depth_bin"] = pd.cut(
        out["source_depth_km"],
        [-np.inf, 10.0, 35.0, 70.0, 150.0, 300.0, np.inf],
        labels=["<=10", "10-35", "35-70", "70-150", "150-300", ">300"],
    )
    out["distance_bin"] = pd.cut(
        out["path_ep_distance_deg"],
        [-np.inf, 1.0, 3.0, 10.0, 30.0, 60.0, np.inf],
        labels=["<=1", "1-3", "3-10", "10-30", "30-60", ">60"],
    )
    if "normalization_scale" in out.columns:
        for src, dst in [
            ("gen_peak_min_abs_norm", "gen_peak_min_counts"),
            ("gen_peak_max_abs_norm", "gen_peak_max_counts"),
            ("gen_peak_median_abs_norm", "gen_peak_median_counts"),
            ("gen_peak_std_abs_norm", "gen_peak_std_counts"),
        ]:
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src] * out["normalization_scale"]
    return out


def peak_distribution_summary(
    row: dict[str, Any],
    df: pd.DataFrame,
    real_col: str,
    gen_col: str,
    prefix: str,
) -> None:
    real_peak = finite_series(df, real_col)
    gen_peak = finite_series(df, gen_col)
    row[f"{prefix}_real_mean"] = safe_mean(real_peak)
    row[f"{prefix}_gen_mean"] = safe_mean(gen_peak)
    row[f"{prefix}_real_std"] = safe_std(real_peak)
    row[f"{prefix}_gen_std_across_conditions"] = safe_std(gen_peak)
    row[f"{prefix}_gen_to_real_mean_ratio"] = safe_mean(gen_peak) / max(safe_mean(real_peak) or EPS, EPS)
    row[f"{prefix}_distribution_ks"] = ks_distance(real_peak, gen_peak)
    row[f"{prefix}_distribution_w1"] = wasserstein_sorted(real_peak, gen_peak)
    row.update(quantile_dict(real_peak, f"{prefix}_real"))
    row.update(quantile_dict(gen_peak, f"{prefix}_gen"))
    for q in [0.90, 0.95, 0.99]:
        rq = float(np.quantile(real_peak, q))
        gq = float(np.quantile(gen_peak, q))
        row[f"{prefix}_q{int(q * 100)}_gen_to_real_ratio"] = gq / max(rq, EPS)


def run_summary(name: str, df: pd.DataFrame) -> dict[str, Any]:
    gen_peak_cv = finite_series(df, "gen_peak_std_abs_norm") / finite_series(df, "gen_peak_mean_abs_norm").clip(lower=EPS)
    row: dict[str, Any] = {
        "run": name,
        "n": int(len(df)),
        "same_condition_gen_peak_cv_mean": safe_mean(gen_peak_cv),
        "same_condition_gen_peak_cv_p95": float(np.quantile(gen_peak_cv.dropna(), 0.95)) if len(gen_peak_cv.dropna()) else None,
    }
    peak_distribution_summary(row, df, "real_peak_abs_norm", "gen_peak_mean_abs_norm", "norm_peak")
    peak_distribution_summary(row, df, "real_peak_counts", "gen_peak_mean_counts", "count_peak")
    for metric in KEY_METRICS:
        row[f"{metric}_mean"] = safe_mean(finite_series(df, metric))
        row[f"{metric}_median"] = float(np.quantile(finite_series(df, metric), 0.5))
        row[f"{metric}_p95"] = float(np.quantile(finite_series(df, metric), 0.95))
    return row


def tail_rows(name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for peak_kind, real_col, gen_mean_col, gen_max_col in [
        ("norm", "real_peak_abs_norm", "gen_peak_mean_abs_norm", "gen_peak_max_abs_norm"),
        ("counts", "real_peak_counts", "gen_peak_mean_counts", "gen_peak_max_counts"),
    ]:
        if gen_max_col not in df.columns:
            continue
        real_peak = finite_series(df, real_col)
        for q in [0.90, 0.95, 0.99]:
            threshold = float(np.quantile(real_peak, q))
            real_tail = df[real_col] >= threshold
            gen_mean_tail = df[gen_mean_col] >= threshold
            gen_max_tail = df[gen_max_col] >= threshold
            tail_df = df[real_tail]
            row: dict[str, Any] = {
                "run": name,
                "peak_kind": peak_kind,
                "tail_quantile": q,
                "real_threshold": threshold,
                "real_tail_rate": float(real_tail.mean()),
                "gen_mean_exceed_rate": float(gen_mean_tail.mean()),
                "gen_max_exceed_rate": float(gen_max_tail.mean()),
                "real_tail_n": int(real_tail.sum()),
                "real_tail_gen_peak_mean": safe_mean(tail_df[gen_mean_col]),
                "real_tail_real_peak_mean": safe_mean(tail_df[real_col]),
                "real_tail_gen_to_real_peak_ratio": safe_mean(tail_df[gen_mean_col]) / max(safe_mean(tail_df[real_col]) or EPS, EPS),
            }
            for metric in KEY_METRICS:
                row[f"real_tail_{metric}_mean"] = safe_mean(tail_df[metric])
            rows.append(row)
    return rows


def group_rows(name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_col in GROUP_COLUMNS:
        if group_col not in df.columns:
            continue
        for group, g in df.groupby(group_col, observed=True, dropna=False):
            if len(g) < 10:
                continue
            row: dict[str, Any] = {
                "run": name,
                "group_type": group_col,
                "group": str(group),
                "n": int(len(g)),
                "norm_peak_real_mean": safe_mean(g["real_peak_abs_norm"]),
                "norm_peak_gen_mean": safe_mean(g["gen_peak_mean_abs_norm"]),
                "norm_peak_gen_to_real_ratio": safe_mean(g["gen_peak_mean_abs_norm"]) / max(safe_mean(g["real_peak_abs_norm"]) or EPS, EPS),
                "count_peak_real_mean": safe_mean(g["real_peak_counts"]),
                "count_peak_gen_mean": safe_mean(g["gen_peak_mean_counts"]),
                "count_peak_gen_to_real_ratio": safe_mean(g["gen_peak_mean_counts"]) / max(safe_mean(g["real_peak_counts"]) or EPS, EPS),
                "count_peak_distribution_ks": ks_distance(g["real_peak_counts"], g["gen_peak_mean_counts"]),
                "same_condition_gen_peak_cv_mean": safe_mean(g["gen_peak_std_abs_norm"] / g["gen_peak_mean_abs_norm"].clip(lower=EPS)),
            }
            for q in CONTROL_QUANTILES:
                real_q = float(np.quantile(finite_series(g, "real_peak_counts"), q))
                gen_q = float(np.quantile(finite_series(g, "gen_peak_mean_counts"), q))
                row[f"count_peak_q{int(q * 100)}_real"] = real_q
                row[f"count_peak_q{int(q * 100)}_gen"] = gen_q
                row[f"count_peak_q{int(q * 100)}_gen_to_real_ratio"] = gen_q / max(real_q, EPS)
            for metric in KEY_METRICS:
                row[f"{metric}_mean"] = safe_mean(g[metric])
            rows.append(row)
    return rows


def weighted_abs_log_ratio(values: pd.Series, weights: pd.Series) -> float | None:
    ratio = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = np.isfinite(ratio) & np.isfinite(w) & (ratio > 0) & (w > 0)
    if mask.sum() == 0:
        return None
    return float(np.average(np.abs(np.log(ratio[mask])), weights=w[mask]))


def dynamic_range_ratio(real_values: pd.Series, gen_values: pd.Series) -> float | None:
    real = pd.to_numeric(real_values, errors="coerce")
    gen = pd.to_numeric(gen_values, errors="coerce")
    real = real[np.isfinite(real) & (real > 0)]
    gen = gen[np.isfinite(gen) & (gen > 0)]
    if len(real) < 2 or len(gen) < 2:
        return None
    real_range = float(np.log(real.max()) - np.log(real.min()))
    gen_range = float(np.log(gen.max()) - np.log(gen.min()))
    if real_range <= EPS:
        return None
    return gen_range / real_range


def axis_control_rows(group_calibration: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if group_calibration.empty:
        return rows
    for (run, group_type), g in group_calibration.groupby(["run", "group_type"], observed=True):
        if len(g) < 2:
            continue
        row: dict[str, Any] = {
            "run": run,
            "condition_axis": group_type,
            "n_groups": int(len(g)),
            "n_samples": int(g["n"].sum()),
            "mean_abs_log_count_peak_ratio": weighted_abs_log_ratio(g["count_peak_gen_to_real_ratio"], g["n"]),
            "mean_abs_log_norm_peak_ratio": weighted_abs_log_ratio(g["norm_peak_gen_to_real_ratio"], g["n"]),
            "mean_count_peak_ks": float(np.average(g["count_peak_distribution_ks"], weights=g["n"])),
            "group_combined_raw_weighted_mean": float(np.average(g["combined_raw_weighted_mean"], weights=g["n"])),
            "real_group_dynamic_range_log": None,
            "gen_group_dynamic_range_log": None,
            "group_dynamic_range_ratio": dynamic_range_ratio(g["count_peak_real_mean"], g["count_peak_gen_mean"]),
            "group_mean_corr": safe_corr(g["count_peak_real_mean"], g["count_peak_gen_mean"]),
        }
        real_means = pd.to_numeric(g["count_peak_real_mean"], errors="coerce")
        gen_means = pd.to_numeric(g["count_peak_gen_mean"], errors="coerce")
        real_pos = real_means[np.isfinite(real_means) & (real_means > 0)]
        gen_pos = gen_means[np.isfinite(gen_means) & (gen_means > 0)]
        if len(real_pos) >= 2:
            row["real_group_dynamic_range_log"] = float(np.log(real_pos.max()) - np.log(real_pos.min()))
        if len(gen_pos) >= 2:
            row["gen_group_dynamic_range_log"] = float(np.log(gen_pos.max()) - np.log(gen_pos.min()))
        tail_errors: list[float] = []
        for q in CONTROL_QUANTILES:
            ratio_col = f"count_peak_q{int(q * 100)}_gen_to_real_ratio"
            if ratio_col in g.columns:
                tail_error = weighted_abs_log_ratio(g[ratio_col], g["n"])
                row[f"mean_abs_log_q{int(q * 100)}_ratio"] = tail_error
                if tail_error is not None and q >= 0.95:
                    tail_errors.append(tail_error)
        range_ratio = row["group_dynamic_range_ratio"]
        range_penalty = abs(math.log(range_ratio)) if range_ratio is not None and range_ratio > 0 else None
        corr = row["group_mean_corr"]
        corr_penalty = max(0.0, 1.0 - corr) if corr is not None else None
        base_error = row["mean_abs_log_count_peak_ratio"]
        tail_error_mean = float(np.mean(tail_errors)) if tail_errors else None
        components = [base_error, tail_error_mean, range_penalty, corr_penalty]
        finite_components = [v for v in components if v is not None and np.isfinite(v)]
        row["dynamic_range_penalty"] = range_penalty
        row["group_corr_penalty"] = corr_penalty
        row["condition_control_score"] = float(np.mean(finite_components)) if finite_components else None
        rows.append(row)
    return rows


def trend_rows(name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    real_log_peak = np.log10(df["real_peak_counts"].clip(lower=EPS))
    gen_log_peak = np.log10(df["gen_peak_mean_counts"].clip(lower=EPS))
    features = {
        "source_magnitude": df["source_magnitude"],
        "log10_distance_km": np.log10(df["path_ep_distance_km"].clip(lower=EPS)),
        "source_depth_km": df["source_depth_km"],
        "phase_travel_sec": df["phase_travel_sec"],
        "residual_travel_sec": df["residual_travel_sec"],
    }
    rows: list[dict[str, Any]] = []
    for feature, values in features.items():
        real_slope = linear_slope(values, real_log_peak)
        gen_slope = linear_slope(values, gen_log_peak)
        rows.append(
            {
                "run": name,
                "feature": feature,
                "real_log_peak_corr": safe_corr(values, real_log_peak),
                "gen_log_peak_corr": safe_corr(values, gen_log_peak),
                "real_log_peak_slope": real_slope,
                "gen_log_peak_slope": gen_slope,
                "slope_abs_error": abs((gen_slope or 0.0) - (real_slope or 0.0)) if real_slope is not None and gen_slope is not None else None,
            }
        )
    return rows


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
    score_paths = parse_score_specs(args.score)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metric_rows: list[dict[str, Any]] = []
    all_tail_rows: list[dict[str, Any]] = []
    all_group_rows: list[dict[str, Any]] = []
    all_trend_rows: list[dict[str, Any]] = []

    for name, path in score_paths.items():
        df = pd.read_csv(path)
        df = prepare_scores(df)
        run_metric_rows.append(run_summary(name, df))
        all_tail_rows.extend(tail_rows(name, df))
        all_group_rows.extend(group_rows(name, df))
        all_trend_rows.extend(trend_rows(name, df))

    run_metrics = pd.DataFrame(run_metric_rows)
    tail_calibration = pd.DataFrame(all_tail_rows)
    group_calibration = pd.DataFrame(all_group_rows)
    physical_trends = pd.DataFrame(all_trend_rows)
    condition_control = pd.DataFrame(axis_control_rows(group_calibration))

    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    tail_calibration.to_csv(output_dir / "tail_calibration.csv", index=False)
    group_calibration.to_csv(output_dir / "group_calibration.csv", index=False)
    physical_trends.to_csv(output_dir / "physical_trends.csv", index=False)
    condition_control.to_csv(output_dir / "condition_control.csv", index=False)

    report = {
        "runs": list(score_paths.keys()),
        "run_metrics": run_metric_rows,
        "tail_calibration": all_tail_rows,
        "physical_trends": all_trend_rows,
        "condition_control": condition_control.to_dict(orient="records"),
        "outputs": {
            "run_metrics": str(output_dir / "run_metrics.csv"),
            "tail_calibration": str(output_dir / "tail_calibration.csv"),
            "group_calibration": str(output_dir / "group_calibration.csv"),
            "physical_trends": str(output_dir / "physical_trends.csv"),
            "condition_control": str(output_dir / "condition_control.csv"),
        },
    }
    (output_dir / "calibration_report.json").write_text(json.dumps(sanitize_json(report), indent=2))

    pd.set_option("display.max_columns", 200)
    print("Run metrics:")
    cols = [
        "run",
        "n",
        "combined_raw_weighted_mean",
        "time_rel_l2_min_mean",
        "envelope_rel_l2_min_mean",
        "spectral_rel_l2_min_mean",
        "peak_log_ratio_abs_mean",
        "norm_peak_gen_to_real_mean_ratio",
        "norm_peak_distribution_ks",
        "norm_peak_q95_gen_to_real_ratio",
        "count_peak_gen_to_real_mean_ratio",
        "count_peak_distribution_ks",
        "count_peak_q95_gen_to_real_ratio",
        "same_condition_gen_peak_cv_mean",
    ]
    print(run_metrics[cols].to_string(index=False))
    print(f"\nWrote calibration audit to {output_dir}")


if __name__ == "__main__":
    main()
