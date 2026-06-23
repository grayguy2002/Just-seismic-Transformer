"""Post-process dense KiK-net single-earthquake validation outputs.

Inputs are the Exp H JsT event-record CSV and the dense cache model waveforms.
The script computes frequency-band correlations, one-record-per-station
bootstrap statistics, and a standard single-event HVSR baseline on the same
records.  It does not rerun JsT inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


F_MIN, F_MAX, N_FREQ_BINS = 0.3, 15.0, 40
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
BINS = [f"hvsr_bin_{i:02d}" for i in range(N_FREQ_BINS)]

BANDS = {
    "0.3-1Hz": [i for i, f in enumerate(FREQ_CENTERS) if 0.3 <= f < 1.0],
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
        return None, None
    stat = spearmanr(x[mask], y[mask])
    return _safe_float(stat.statistic), _safe_float(stat.pvalue)


def _partial_rank_latlon(df: pd.DataFrame, metric: str):
    cols = [metric, "vs30", "station_latitude_deg", "station_longitude_deg"]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8:
        return None, None
    x = pd.Series(sub[metric]).rank(method="average").to_numpy(float)
    y = pd.Series(sub["vs30"]).rank(method="average").to_numpy(float)
    cov = sub[["station_latitude_deg", "station_longitude_deg"]].to_numpy(float)
    cov = (cov - cov.mean(axis=0)) / np.where(cov.std(axis=0) > 0, cov.std(axis=0), 1.0)
    design = np.column_stack([np.ones(len(cov)), cov])
    bx, *_ = np.linalg.lstsq(design, x, rcond=None)
    by, *_ = np.linalg.lstsq(design, y, rcond=None)
    xr = x - design @ bx
    yr = y - design @ by
    if np.std(xr) == 0 or np.std(yr) == 0:
        return None, None
    stat = pearsonr(xr, yr)
    return _safe_float(stat.statistic), _safe_float(stat.pvalue)


def add_jst_bands(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for name, idx in BANDS.items():
        df[name] = df[[BINS[i] for i in idx]].mean(axis=1)
    return add_qc_flags(df)


def add_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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


def summarize_jst_frequency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["mean_amp", *BANDS.keys()]
    scopes = [("all", df), ("arrival_qc", df[df["arrival_qc"]]), ("full_pre_qc", df[df["full_pre_qc"]])]
    for scope, sub in scopes:
        for metric in metrics:
            rho, pval = _spearman(sub[metric], sub["vs30"])
            prho, ppval = _partial_rank_latlon(sub, metric)
            rows.append({
                "scope": scope,
                "event_id": "pooled",
                "metric": metric,
                "N": int(len(sub)),
                "rho": rho,
                "p": pval,
                "partial_latlon_r": prho,
                "partial_latlon_p": ppval,
            })
        for event_id, group in sub.groupby("event_id", sort=True):
            for metric in metrics:
                rho, pval = _spearman(group[metric], group["vs30"])
                prho, ppval = _partial_rank_latlon(group, metric)
                rows.append({
                    "scope": scope,
                    "event_id": str(event_id),
                    "metric": metric,
                    "N": int(len(group)),
                    "rho": rho,
                    "p": pval,
                    "partial_latlon_r": prho,
                    "partial_latlon_p": ppval,
                })
    return pd.DataFrame(rows)


def bootstrap_one_record(df: pd.DataFrame, n_boot: int = 2000, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metrics = ["mean_amp", *BANDS.keys()]
    sub = df[["station_id", "vs30", *metrics]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    groups = [g.index.to_numpy() for _, g in sub.groupby("station_id", sort=True)]
    vs30 = sub["vs30"].to_numpy(float)
    metric_values = {metric: sub[metric].to_numpy(float) for metric in metrics}
    for _ in range(n_boot):
        idx = np.array([rng.choice(group) for group in groups], dtype=int)
        for metric in metrics:
            rho, _ = _spearman(metric_values[metric][idx], vs30[idx])
            rows.append({"metric": metric, "rho": rho})
    return pd.DataFrame(rows)


def compute_standard_hvsr(cache_dir: Path, conditions: pd.DataFrame) -> pd.DataFrame:
    wave_path = cache_dir / "kiknet_measured_vs30_pwave_v1_X_model_20p60_streamnorm_float32.npy"
    waves = np.load(wave_path, mmap_mode="r")
    freqs = np.fft.rfftfreq(waves.shape[2], d=1.0 / 40.0)
    rows = []
    for i, row in conditions.iterrows():
        amps = np.abs(np.fft.rfft(np.asarray(waves[i], dtype=np.float32), axis=1))
        horizontal = np.sqrt((amps[0] ** 2 + amps[1] ** 2) / 2.0)
        vertical = amps[2]
        spectrum = []
        for j in range(N_FREQ_BINS):
            mask = (freqs >= FREQ_EDGES[j]) & (freqs < FREQ_EDGES[j + 1])
            if mask.any():
                h = max(float(horizontal[mask].mean()), 1e-12)
                z = max(float(vertical[mask].mean()), 1e-12)
                spectrum.append(np.log10(h / z))
            else:
                spectrum.append(0.0)
        spectrum = np.asarray(spectrum, dtype=float)
        rec = {
            "event_id": str(row["event_id"]),
            "station_id": row["station_id"],
            "cache_index": int(row["cache_index"]),
            "vs30": float(row["vs30_m_s"]),
            "nehrp": row["nehrp_site_class"],
            "station_latitude_deg": float(row["station_latitude_deg"]),
            "station_longitude_deg": float(row["station_longitude_deg"]),
            "arrival_sample": float(row["selected_phase_arrival_sample"]),
            "left_pad": float(row["model_left_pad_samples"]),
            "right_pad": float(row["model_right_pad_samples"]),
            "trace_n_samples": float(row["trace_n_samples"]),
            "mean_hv": float(np.mean(spectrum)),
        }
        for name, idx in BANDS.items():
            rec[name] = float(np.mean(spectrum[idx]))
        rows.append(rec)
    return add_qc_flags(pd.DataFrame(rows).rename(columns={"mean_hv": "mean_amp"}))


def summarize_bootstrap(boot: pd.DataFrame) -> list[dict]:
    out = []
    for metric, group in boot.groupby("metric", sort=True):
        vals = pd.to_numeric(group["rho"], errors="coerce").dropna().to_numpy(float)
        out.append({
            "metric": metric,
            "n_bootstrap": int(len(vals)),
            "median_rho": _safe_float(np.median(vals)),
            "mean_rho": _safe_float(np.mean(vals)),
            "ci95_low": _safe_float(np.percentile(vals, 2.5)),
            "ci95_high": _safe_float(np.percentile(vals, 97.5)),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", default="outputs/expH_kiknet_dense_arrival_qc_events")
    parser.add_argument("--cache-dir", default="data/kiknet_dense_arrival_qc_events_v1/cache")
    parser.add_argument("--output-dir", default="outputs/expH_kiknet_dense_arrival_qc_events_postprocess")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jst = add_jst_bands(pd.read_csv(exp_dir / "per_station_event_records.csv"))
    jst_corr = summarize_jst_frequency(jst)
    boot = bootstrap_one_record(jst[jst["arrival_qc"]])

    conditions_path = cache_dir / "kiknet_measured_vs30_pwave_v1_conditions.csv"
    wave_path = cache_dir / "kiknet_measured_vs30_pwave_v1_X_model_20p60_streamnorm_float32.npy"
    standard_path = out_dir / "standard_hvsr_records.csv"
    if conditions_path.exists() and wave_path.exists():
        conditions = pd.read_csv(conditions_path)
        standard = compute_standard_hvsr(cache_dir, conditions)
    elif standard_path.exists():
        standard = add_qc_flags(pd.read_csv(standard_path))
    else:
        raise FileNotFoundError(
            "Standard HVSR cache is unavailable and no existing "
            f"{standard_path} file can be reused."
        )
    std_corr = summarize_jst_frequency(standard)

    jst_corr.to_csv(out_dir / "jst_frequency_correlations.csv", index=False)
    boot.to_csv(out_dir / "jst_one_record_per_station_bootstrap.csv", index=False)
    standard.to_csv(out_dir / "standard_hvsr_records.csv", index=False)
    std_corr.to_csv(out_dir / "standard_hvsr_frequency_correlations.csv", index=False)

    summary = {
        "frequency_bands_hz": {
            name: [float(FREQ_CENTERS[idx[0]]), float(FREQ_CENTERS[idx[-1]])]
            for name, idx in BANDS.items()
        },
        "jst_pooled": jst_corr[jst_corr["event_id"].eq("pooled")].replace({np.nan: None}).to_dict("records"),
        "standard_hvsr_pooled": std_corr[std_corr["event_id"].eq("pooled")].replace({np.nan: None}).to_dict("records"),
        "jst_one_record_bootstrap": summarize_bootstrap(boot),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
