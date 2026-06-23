"""Validation controls for dense KiK-net JsT site-effect tests.

This post-processing experiment addresses three review-facing questions:

1. How much of the JsT--Vs30 association is reproduced by metadata alone?
2. Does the station-level ordering survive spatial block resampling?
3. Does a modest rank correlation still produce useful soft-site screening?

The script reads the Exp H station-event records and writes deterministic CSV
and JSON outputs. It does not rerun waveform inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

NUMERIC_METADATA = [
    "station_latitude_deg",
    "station_longitude_deg",
    "source_latitude_deg",
    "source_longitude_deg",
    "source_depth_km",
    "source_magnitude",
    "path_ep_distance_km",
    "arrival_sample",
    "left_pad",
]
CATEGORICAL_METADATA = ["event_id"]


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


def _rank(values):
    return pd.Series(values).rank(method="average").to_numpy(float)


def _standardize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    mu = np.nanmean(matrix, axis=0)
    sigma = np.nanstd(matrix, axis=0)
    sigma = np.where(sigma > 0, sigma, 1.0)
    return (matrix - mu) / sigma


def _residualize(y, controls):
    y = np.asarray(y, dtype=float)
    controls = _standardize(controls)
    design = np.column_stack([np.ones(len(controls)), controls])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def _metadata_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric = df[NUMERIC_METADATA].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).to_numpy(float)
    events = pd.get_dummies(df["event_id"].astype(str), drop_first=True).to_numpy(float)
    return np.column_stack([numeric, events])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, idx in BANDS.items():
        df[name] = df[[BINS[i] for i in idx]].mean(axis=1)
    df["arrival_qc"] = (
        (df["arrival_sample"] >= 0)
        & (df["arrival_sample"] < df["trace_n_samples"])
        & (df["left_pad"] <= 800)
        & (df["right_pad"] <= 0)
    )
    df["full_pre_qc"] = (
        (df["arrival_sample"] >= 800)
        & (df["arrival_sample"] < df["trace_n_samples"])
        & (df["left_pad"] == 0)
        & (df["right_pad"] <= 0)
    )
    return df


def build_metadata_model() -> Pipeline:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        [
            ("numeric", numeric_pipe, NUMERIC_METADATA),
            ("event", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_METADATA),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("pre", pre),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]
    )


def manual_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(labels)
    labels = labels[mask]
    scores = scores[mask]
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return _safe_float(auc)


def metadata_controls(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    sub = df[df["arrival_qc"]].copy()
    cols = ["station_id", "vs30", metric, *NUMERIC_METADATA, *CATEGORICAL_METADATA]
    sub = sub[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["station_id", "vs30", metric]).reset_index(drop=True)

    raw_rho, raw_p, n = _spearman(sub[metric], sub["vs30"])
    controls = _metadata_matrix(sub)
    x_res = _residualize(_rank(sub[metric]), controls)
    y_res = _residualize(_rank(sub["vs30"]), controls)
    residual_r, residual_p, _ = _pearson(x_res, y_res)

    model = build_metadata_model()
    cv = GroupKFold(n_splits=5)
    groups = sub["station_id"].astype(str)
    xcols = [*NUMERIC_METADATA, *CATEGORICAL_METADATA]
    jst_rank = _rank(sub[metric])
    predicted_jst_rank = cross_val_predict(model, sub[xcols], jst_rank, groups=groups, cv=cv)
    cv_jst_r, cv_jst_p, _ = _pearson(predicted_jst_rank, jst_rank)
    cv_jst_vs30_rho, cv_jst_vs30_p, _ = _spearman(predicted_jst_rank, sub["vs30"])

    vs30_rank = _rank(np.log10(sub["vs30"]))
    predicted_vs30_rank = cross_val_predict(model, sub[xcols], vs30_rank, groups=groups, cv=cv)
    cv_vs30_r, cv_vs30_p, _ = _pearson(predicted_vs30_rank, vs30_rank)
    cv_vs30_vs30_rho, cv_vs30_vs30_p, _ = _spearman(predicted_vs30_rank, sub["vs30"])

    soft = (sub["vs30"].to_numpy(float) < 360.0).astype(int)
    raw_auc = manual_auc(soft, sub[metric])
    metadata_score_auc = manual_auc(soft, predicted_jst_rank)
    metadata_vs30_auc = manual_auc(soft, -predicted_vs30_rank)

    rows = [
        {
            "check": "raw_jst_single_records",
            "N": n,
            "statistic": "spearman_rho",
            "value": raw_rho,
            "p": raw_p,
            "detail": f"{metric}; arrival/window QC records",
        },
        {
            "check": "metadata_residualized_jst",
            "N": n,
            "statistic": "partial_rank_r",
            "value": residual_r,
            "p": residual_p,
            "detail": "Ranks of JsT score and Vs30 residualized against station/source/path/window metadata and event fixed effects",
        },
        {
            "check": "metadata_only_predicts_jst",
            "N": n,
            "statistic": "cross_validated_r",
            "value": cv_jst_r,
            "p": cv_jst_p,
            "detail": "Station-grouped 5-fold ridge model; target is JsT score rank",
        },
        {
            "check": "metadata_only_score_vs_vs30",
            "N": n,
            "statistic": "spearman_rho",
            "value": cv_jst_vs30_rho,
            "p": cv_jst_vs30_p,
            "detail": "Cross-validated metadata-only JsT-score prediction versus profile Vs30",
        },
        {
            "check": "metadata_only_predicts_vs30",
            "N": n,
            "statistic": "cross_validated_r",
            "value": cv_vs30_r,
            "p": cv_vs30_p,
            "detail": "Station-grouped 5-fold ridge model; target is log Vs30 rank",
        },
        {
            "check": "metadata_only_vs30_prediction",
            "N": n,
            "statistic": "spearman_rho",
            "value": cv_vs30_vs30_rho,
            "p": cv_vs30_vs30_p,
            "detail": "Cross-validated metadata-only predicted log Vs30 rank versus profile Vs30",
        },
        {
            "check": "raw_jst_soft_site_auc",
            "N": n,
            "statistic": "auc_vs30_lt_360",
            "value": raw_auc,
            "p": None,
            "detail": "Positive class is Vs30 < 360 m s-1",
        },
        {
            "check": "metadata_only_score_soft_site_auc",
            "N": n,
            "statistic": "auc_vs30_lt_360",
            "value": metadata_score_auc,
            "p": None,
            "detail": "Soft-site AUC from cross-validated metadata-only JsT-score prediction",
        },
        {
            "check": "metadata_only_vs30_soft_site_auc",
            "N": n,
            "statistic": "auc_vs30_lt_360",
            "value": metadata_vs30_auc,
            "p": None,
            "detail": "Soft-site AUC from cross-validated metadata-only Vs30-rank prediction",
        },
    ]
    table = pd.DataFrame(rows)
    summary = {row["check"]: row for row in rows}
    return table, summary


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


def add_spatial_blocks(stations: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    out = stations.copy()
    lat_bin = pd.qcut(out["station_latitude_deg"], q=q, labels=False, duplicates="drop")
    lon_bin = pd.qcut(out["station_longitude_deg"], q=q, labels=False, duplicates="drop")
    out["spatial_block"] = lat_bin.astype(str) + "_" + lon_bin.astype(str)
    return out


def spatial_block_controls(df: pd.DataFrame, metric: str, n_boot: int = 2000, seed: int = 123) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary = {}
    rng = np.random.default_rng(seed)
    for label, sub in [
        ("station_mean_min1", station_means(df, metric)),
        ("station_mean_min2", station_means(df, metric).query("n_events >= 2").copy()),
    ]:
        sub = add_spatial_blocks(sub)
        rho, pval, n = _spearman(sub["score"], sub["vs30"])
        rows.append(
            {
                "analysis": label,
                "check": "station_correlation",
                "block": "all",
                "N": n,
                "n_blocks": int(sub["spatial_block"].nunique()),
                "rho": rho,
                "p": pval,
                "ci95_low": None,
                "ci95_high": None,
            }
        )

        jackknife = []
        for block, group in sub.groupby("spatial_block", sort=True):
            keep = sub[sub["spatial_block"] != block]
            krho, kp, kn = _spearman(keep["score"], keep["vs30"])
            if krho is not None:
                jackknife.append(krho)
                rows.append(
                    {
                        "analysis": label,
                        "check": "leave_one_spatial_block_out",
                        "block": block,
                        "N": kn,
                        "n_blocks": int(keep["spatial_block"].nunique()),
                        "rho": krho,
                        "p": kp,
                        "ci95_low": None,
                        "ci95_high": None,
                    }
                )

        blocks = sorted(sub["spatial_block"].unique())
        boot_rhos = []
        for _ in range(n_boot):
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            boot = pd.concat([sub[sub["spatial_block"] == block] for block in chosen], ignore_index=True)
            brho, _, _ = _spearman(boot["score"], boot["vs30"])
            if brho is not None:
                boot_rhos.append(brho)
        boot_rhos = np.asarray(boot_rhos, dtype=float)
        ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
        boot_median = float(np.median(boot_rhos))
        rows.append(
            {
                "analysis": label,
                "check": "spatial_block_bootstrap",
                "block": "cluster_resample",
                "N": n,
                "n_blocks": int(len(blocks)),
                "rho": boot_median,
                "p": None,
                "ci95_low": _safe_float(ci_low),
                "ci95_high": _safe_float(ci_high),
            }
        )
        summary[label] = {
            "N": n,
            "n_blocks": int(len(blocks)),
            "rho": rho,
            "p": pval,
            "jackknife_min": _safe_float(np.nanmin(jackknife)),
            "jackknife_max": _safe_float(np.nanmax(jackknife)),
            "jackknife_median": _safe_float(np.nanmedian(jackknife)),
            "block_bootstrap_median": _safe_float(boot_median),
            "block_bootstrap_ci95": [_safe_float(ci_low), _safe_float(ci_high)],
        }
    return pd.DataFrame(rows), summary


def screening_rows(label: str, scores, vs30) -> dict:
    scores = np.asarray(scores, dtype=float)
    vs30 = np.asarray(vs30, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(vs30)
    scores = scores[mask]
    vs30 = vs30[mask]
    soft = vs30 < 360.0
    low_quartile_cut = np.quantile(vs30, 0.25)
    low_quartile = vs30 <= low_quartile_cut
    high_score = scores >= np.quantile(scores, 0.75)
    low_score = scores <= np.quantile(scores, 0.25)

    soft_base = float(soft.mean())
    soft_precision = float(soft[high_score].mean())
    soft_recall = float(soft[high_score].sum() / max(soft.sum(), 1))
    low_base = float(low_quartile.mean())
    low_precision = float(low_quartile[high_score].mean())
    low_recall = float(low_quartile[high_score].sum() / max(low_quartile.sum(), 1))

    return {
        "analysis": label,
        "N": int(mask.sum()),
        "auc_vs30_lt_360": manual_auc(soft.astype(int), scores),
        "soft_site_prevalence": _safe_float(soft_base),
        "top_score_quartile_soft_precision": _safe_float(soft_precision),
        "top_score_quartile_soft_recall": _safe_float(soft_recall),
        "top_score_quartile_soft_enrichment": _safe_float(soft_precision / soft_base) if soft_base else None,
        "low_vs30_quartile_cut": _safe_float(low_quartile_cut),
        "auc_lowest_vs30_quartile": manual_auc(low_quartile.astype(int), scores),
        "lowest_quartile_prevalence": _safe_float(low_base),
        "top_score_quartile_lowest_quartile_precision": _safe_float(low_precision),
        "top_score_quartile_lowest_quartile_recall": _safe_float(low_recall),
        "top_score_quartile_lowest_quartile_enrichment": _safe_float(low_precision / low_base) if low_base else None,
        "median_vs30_top_score_quartile": _safe_float(np.median(vs30[high_score])),
        "median_vs30_bottom_score_quartile": _safe_float(np.median(vs30[low_score])),
    }


def screening_utility(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    record = df[df["arrival_qc"]].copy()
    st_min1 = station_means(df, metric)
    st_min2 = st_min1[st_min1["n_events"] >= 2].copy()
    rows = [
        screening_rows("single_records", record[metric], record["vs30"]),
        screening_rows("station_means_min1", st_min1["score"], st_min1["vs30"]),
        screening_rows("station_means_min2", st_min2["score"], st_min2["vs30"]),
    ]
    table = pd.DataFrame(rows)
    return table, {row["analysis"]: row for row in rows}


def event_fixed_rank_controls(
    df: pd.DataFrame,
    metric: str,
    n_perm: int = 2000,
    seed: int = 123,
) -> tuple[pd.DataFrame, dict]:
    sub = (
        df[df["arrival_qc"]][
            [
                "event_id",
                "station_id",
                metric,
                "vs30",
                "station_latitude_deg",
                "station_longitude_deg",
            ]
        ]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    groups = []
    for event_id, group in sub.groupby("event_id", sort=True):
        if len(group) < 3:
            continue
        g = group.copy()
        score_rank = _rank(g[metric])
        vs30_rank = _rank(g["vs30"])
        g["score_rank"] = (score_rank - score_rank.mean()) / max(score_rank.std(), 1e-12)
        g["vs30_rank"] = (vs30_rank - vs30_rank.mean()) / max(vs30_rank.std(), 1e-12)
        coords = g[["station_latitude_deg", "station_longitude_deg"]].to_numpy(float)
        g["score_rank_spatial_resid"] = _residualize(g["score_rank"].to_numpy(float), coords)
        g["vs30_rank_spatial_resid"] = _residualize(g["vs30_rank"].to_numpy(float), coords)
        groups.append(g)
    ranked = pd.concat(groups, ignore_index=True)

    raw_r, raw_p, n = _pearson(ranked["score_rank"], ranked["vs30_rank"])
    partial_r, partial_p, _ = _pearson(ranked["score_rank_spatial_resid"], ranked["vs30_rank_spatial_resid"])

    rng = np.random.default_rng(seed)
    raw_perm = []
    partial_perm = []
    event_indices = [g.index.to_numpy() for _, g in ranked.groupby("event_id", sort=True)]
    score = ranked["score_rank"].to_numpy(float)
    score_resid = ranked["score_rank_spatial_resid"].to_numpy(float)
    vs30_rank = ranked["vs30_rank"].to_numpy(float)
    vs30_resid = ranked["vs30_rank_spatial_resid"].to_numpy(float)
    for _ in range(n_perm):
        shuffled_rank = vs30_rank.copy()
        shuffled_resid = vs30_resid.copy()
        for idx in event_indices:
            shuffled_rank[idx] = rng.permutation(shuffled_rank[idx])
            shuffled_resid[idx] = rng.permutation(shuffled_resid[idx])
        raw_perm.append(abs(pearsonr(score, shuffled_rank).statistic))
        partial_perm.append(abs(pearsonr(score_resid, shuffled_resid).statistic))
    raw_perm = np.asarray(raw_perm, dtype=float)
    partial_perm = np.asarray(partial_perm, dtype=float)
    raw_perm_p = (float((raw_perm >= abs(raw_r)).sum()) + 1.0) / (len(raw_perm) + 1.0)
    partial_perm_p = (float((partial_perm >= abs(partial_r)).sum()) + 1.0) / (len(partial_perm) + 1.0)

    rows = [
        {
            "check": "event_fixed_rank",
            "N": n,
            "n_events": int(ranked["event_id"].nunique()),
            "statistic": "pearson_r_within_event_ranks",
            "value": raw_r,
            "p": raw_p,
            "permutation_p": raw_perm_p,
        },
        {
            "check": "event_fixed_rank_spatial_residual",
            "N": n,
            "n_events": int(ranked["event_id"].nunique()),
            "statistic": "pearson_r_within_event_spatial_residual_ranks",
            "value": partial_r,
            "p": partial_p,
            "permutation_p": partial_perm_p,
        },
    ]
    table = pd.DataFrame(rows)
    return table, {row["check"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="outputs/expH_kiknet_dense_arrival_qc_events/per_station_event_records.csv")
    parser.add_argument("--output-dir", default="outputs/expI_kiknet_validation_controls")
    parser.add_argument("--metric", default="1-10Hz")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    records = Path(args.records)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(pd.read_csv(records))
    metric = args.metric
    if metric not in df.columns:
        raise ValueError(f"Metric {metric!r} not available")

    metadata_table, metadata_summary = metadata_controls(df, metric)
    spatial_table, spatial_summary = spatial_block_controls(df, metric, n_boot=args.n_bootstrap)
    screening_table, screening_summary = screening_utility(df, metric)
    event_table, event_summary = event_fixed_rank_controls(df, metric, n_perm=args.n_bootstrap)

    metadata_table.to_csv(out_dir / "metadata_only_controls.csv", index=False)
    spatial_table.to_csv(out_dir / "spatial_block_controls.csv", index=False)
    screening_table.to_csv(out_dir / "screening_utility.csv", index=False)
    event_table.to_csv(out_dir / "event_fixed_rank_controls.csv", index=False)

    summary = {
        "metric": metric,
        "record_counts": {
            "all_records": int(len(df)),
            "arrival_qc_records": int(df["arrival_qc"].sum()),
            "arrival_qc_stations": int(df.loc[df["arrival_qc"], "station_id"].nunique()),
            "full_pre_qc_records": int(df["full_pre_qc"].sum()),
            "full_pre_qc_stations": int(df.loc[df["full_pre_qc"], "station_id"].nunique()),
        },
        "metadata_controls": metadata_summary,
        "spatial_block_controls": spatial_summary,
        "screening_utility": screening_summary,
        "event_fixed_controls": event_summary,
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
