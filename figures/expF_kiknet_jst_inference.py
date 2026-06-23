"""JsT inference on the KiK-net measured-Vs30 validation cache.

Reads the kiknet_measured_vs30_pwave_v1 cache, runs the fixed JsT checkpoint,
computes residual spectra per station, and correlates the station measurement
with profile-derived measured Vs30.

The script reports the all-event result and two arrival/window-QC sensitivities.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, mannwhitneyu, pearsonr, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from JsT import SeismicWaveformDataset, load_checkpoint_models
from JsT.ablation import AblationConditionEncoder


N_FREQ_BINS = 40
F_MIN, F_MAX = 0.3, 15.0
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])


@torch.no_grad()
def generate_waveform(dn, tokens, noise, steps=50):
    device = tokens.device
    net = dn.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()
    for i in range(steps):
        t = ts[i]
        t_next = ts[i + 1]
        xp = net(z, t.expand(z.shape[0]), tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
        z = z + (t_next - t) * v
    return z


def compute_hvsr(residual: np.ndarray, predicted: np.ndarray, sample_rate_hz: float, eps: float = 1e-12):
    """Return a 40-bin residual/predicted log spectrum using the cache sample rate."""
    per_comp = []
    freqs = np.fft.rfftfreq(residual.shape[1], d=1.0 / sample_rate_hz)
    for ch in range(3):
        sr = np.abs(np.fft.rfft(residual[ch]))
        sp = np.abs(np.fft.rfft(predicted[ch]))
        br = np.zeros(N_FREQ_BINS, dtype=np.float64)
        bp = np.zeros(N_FREQ_BINS, dtype=np.float64)
        for b in range(N_FREQ_BINS):
            mask = (freqs >= FREQ_EDGES[b]) & (freqs < FREQ_EDGES[b + 1])
            if mask.any():
                br[b] = np.mean(sr[mask])
                bp[b] = np.mean(sp[mask])
            else:
                br[b] = np.nan
                bp[b] = np.nan
        ratio = np.log10(np.maximum(br, eps) / np.maximum(bp, eps))
        per_comp.append(ratio)
    out = np.nanmean(np.asarray(per_comp), axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _corr(x: np.ndarray, y: np.ndarray, kind: str):
    if len(x) < 3:
        return None, None
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return None, None
    stat = spearmanr(x, y) if kind == "spearman" else pearsonr(x, y)
    return _safe_float(stat.statistic), _safe_float(stat.pvalue)


def _rank_z(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=float), method="average")
    ranks = ranks - np.mean(ranks)
    scale = np.std(ranks)
    return ranks / scale if scale > 0 else np.zeros_like(ranks, dtype=float)


def _linear_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < x.shape[1] + 2:
        return out
    design = np.column_stack([np.ones(mask.sum()), x[mask]])
    beta, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
    out[mask] = y[mask] - design @ beta
    return out


def _partial_spearman_latlon(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[float | None, float | None]:
    cols = [x_col, y_col, "station_latitude_deg", "station_longitude_deg"]
    if any(col not in df for col in cols):
        return None, None
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 5:
        return None, None
    x_rank = rankdata(sub[x_col].to_numpy(float), method="average")
    y_rank = rankdata(sub[y_col].to_numpy(float), method="average")
    cov = sub[["station_latitude_deg", "station_longitude_deg"]].to_numpy(float)
    cov = (cov - cov.mean(axis=0)) / np.where(cov.std(axis=0) > 0, cov.std(axis=0), 1.0)
    x_resid = _linear_residual(x_rank, cov)
    y_resid = _linear_residual(y_rank, cov)
    mask = np.isfinite(x_resid) & np.isfinite(y_resid)
    if mask.sum() < 5 or np.std(x_resid[mask]) == 0 or np.std(y_resid[mask]) == 0:
        return None, None
    stat = pearsonr(x_resid[mask], y_resid[mask])
    return _safe_float(stat.statistic), _safe_float(stat.pvalue)


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None, None
    ranks = rankdata(scores, method="average")
    auc = (float(ranks[labels].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    try:
        pval = mannwhitneyu(scores[labels], scores[~labels], alternative="greater").pvalue
    except ValueError:
        pval = np.nan
    return _safe_float(auc), _safe_float(pval)


def event_records_dataframe(event_records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in event_records:
        row = {
            "event_id": record["event_id"],
            "source_origin_time": record["source_origin_time"],
            "source_magnitude": record["source_magnitude"],
            "source_depth_km": record["source_depth_km"],
            "source_latitude_deg": record["source_latitude_deg"],
            "source_longitude_deg": record["source_longitude_deg"],
            "path_ep_distance_km": record["path_ep_distance_km"],
            "station_id": record["station_id"],
            "cache_index": record["cache_index"],
            "mean_amp": float(np.mean(record["hvsr"])),
            "vs30": record["vs30"],
            "nehrp": record["nehrp"],
            "station_latitude_deg": record["station_latitude_deg"],
            "station_longitude_deg": record["station_longitude_deg"],
            "arrival_sample": record["arrival_sample"],
            "left_pad": record["left_pad"],
            "right_pad": record["right_pad"],
            "trace_n_samples": record["trace_n_samples"],
            "sample_rate_hz": record["sample_rate_hz"],
        }
        for i, value in enumerate(record["hvsr"]):
            row[f"hvsr_bin_{i:02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def event_fixed_rank_correlation(
    df: pd.DataFrame,
    *,
    min_stations_per_event: int,
    n_permutations: int = 2000,
    seed: int = 20260622,
    latlon_partial: bool = False,
) -> dict:
    groups = []
    for event_id, group in df.groupby("event_id", sort=True):
        group = group.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["mean_amp", "vs30", "station_latitude_deg", "station_longitude_deg"]
        )
        if len(group) >= min_stations_per_event and group["mean_amp"].nunique() > 1 and group["vs30"].nunique() > 1:
            groups.append((event_id, group.copy()))

    if not groups:
        return {
            "min_stations_per_event": int(min_stations_per_event),
            "n_events": 0,
            "n_station_events": 0,
            "rank_r": None,
            "p_permutation_negative": None,
            "latlon_partial": bool(latlon_partial),
        }

    x_parts, y_parts, cov_parts = [], [], []
    group_slices = []
    start = 0
    for _, group in groups:
        xg = _rank_z(group["mean_amp"].to_numpy(float))
        yg = _rank_z(group["vs30"].to_numpy(float))
        cov = group[["station_latitude_deg", "station_longitude_deg"]].to_numpy(float)
        cov = cov - cov.mean(axis=0, keepdims=True)
        x_parts.append(xg)
        y_parts.append(yg)
        cov_parts.append(cov)
        stop = start + len(group)
        group_slices.append(slice(start, stop))
        start = stop

    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    cov = np.vstack(cov_parts)
    if latlon_partial:
        cov = cov / np.where(cov.std(axis=0) > 0, cov.std(axis=0), 1.0)
        x_eval = _linear_residual(x, cov)
        y_eval = _linear_residual(y, cov)
    else:
        x_eval = x
        y_eval = y
    mask = np.isfinite(x_eval) & np.isfinite(y_eval)
    rank_r = pearsonr(x_eval[mask], y_eval[mask]).statistic if mask.sum() >= 3 else np.nan

    rng = np.random.default_rng(seed + min_stations_per_event + (1000 if latlon_partial else 0))
    perm_stats = []
    for _ in range(n_permutations):
        yp = y.copy()
        for sl in group_slices:
            yp[sl] = rng.permutation(yp[sl])
        if latlon_partial:
            yp_eval = _linear_residual(yp, cov)
        else:
            yp_eval = yp
        pmask = np.isfinite(x_eval) & np.isfinite(yp_eval)
        if pmask.sum() >= 3 and np.std(yp_eval[pmask]) > 0:
            perm_stats.append(float(pearsonr(x_eval[pmask], yp_eval[pmask]).statistic))
    perm_stats = np.asarray(perm_stats, dtype=float)
    if np.isfinite(rank_r) and len(perm_stats):
        p_perm = (1 + int(np.sum(perm_stats <= rank_r))) / (len(perm_stats) + 1)
    else:
        p_perm = None

    return {
        "min_stations_per_event": int(min_stations_per_event),
        "n_events": int(len(groups)),
        "n_station_events": int(sum(len(g) for _, g in groups)),
        "rank_r": _safe_float(rank_r),
        "p_permutation_negative": _safe_float(p_perm),
        "latlon_partial": bool(latlon_partial),
    }


def summarize_single_event_correlations(
    event_records: list[dict],
    name: str,
    event_filter: Callable[[dict], bool],
    thresholds: tuple[int, ...] = (3, 5, 8, 10, 12, 15),
) -> tuple[dict, pd.DataFrame]:
    df = event_records_dataframe([r for r in event_records if event_filter(r)])
    per_event = []
    if len(df):
        for event_id, group in df.groupby("event_id", sort=True):
            if len(group) < 3:
                continue
            x = group["mean_amp"].to_numpy(float)
            y = group["vs30"].to_numpy(float)
            rho, pval = _corr(x, y, "spearman")
            pear, pear_p = _corr(x, y, "pearson")
            classes = group["nehrp"].astype(str).value_counts().to_dict()
            per_event.append({
                "event_id": str(event_id),
                "source_origin_time": str(group["source_origin_time"].iloc[0]),
                "source_magnitude": _safe_float(group["source_magnitude"].iloc[0]),
                "source_depth_km": _safe_float(group["source_depth_km"].iloc[0]),
                "n_stations": int(len(group)),
                "vs30_min": _safe_float(group["vs30"].min()),
                "vs30_max": _safe_float(group["vs30"].max()),
                "nehrp_classes": ";".join(f"{k}:{v}" for k, v in sorted(classes.items())),
                "spearman_rho": rho,
                "spearman_p": pval,
                "pearson_r": pear,
                "pearson_p": pear_p,
            })
    event_df = pd.DataFrame(per_event)

    threshold_summaries = {}
    for threshold in thresholds:
        sub = event_df[event_df["n_stations"] >= threshold].copy() if len(event_df) else event_df
        rhos = pd.to_numeric(sub["spearman_rho"], errors="coerce").dropna().to_numpy(float) if len(sub) else np.array([])
        n_neg = int(np.sum(rhos < 0))
        n_pos = int(np.sum(rhos > 0))
        n_nonzero = n_neg + n_pos
        sign_p = binomtest(n_neg, n_nonzero, 0.5, alternative="greater").pvalue if n_nonzero else None
        threshold_summaries[f"min{threshold}"] = {
            "min_stations_per_event": int(threshold),
            "n_events": int(len(sub)),
            "n_station_events": int(sub["n_stations"].sum()) if len(sub) else 0,
            "median_spearman_rho": _safe_float(np.median(rhos)) if len(rhos) else None,
            "mean_spearman_rho": _safe_float(np.mean(rhos)) if len(rhos) else None,
            "negative_rho_events": n_neg,
            "positive_rho_events": n_pos,
            "sign_test_p_negative": _safe_float(sign_p),
            "event_fixed_rank": event_fixed_rank_correlation(df, min_stations_per_event=threshold),
            "event_fixed_rank_latlon_partial": event_fixed_rank_correlation(
                df, min_stations_per_event=threshold, latlon_partial=True
            ),
        }

    return {
        "name": name,
        "n_station_events": int(len(df)),
        "n_events_with_at_least_3_stations": int(len(event_df)),
        "thresholds": threshold_summaries,
    }, event_df


def summarize_soft_hard_ordering(
    event_records: list[dict],
    name: str,
    event_filter: Callable[[dict], bool],
    thresholds: tuple[int, ...] = (3, 5, 8, 10, 12, 15),
) -> dict:
    df = event_records_dataframe([r for r in event_records if event_filter(r)])
    if len(df) == 0:
        return {"name": name, "thresholds": {}}
    class_map = {"A": "hard", "B": "hard", "D": "soft", "E": "soft"}
    df = df[df["nehrp"].astype(str).isin(class_map)].copy()
    df["soft_site"] = df["nehrp"].astype(str).map(class_map).eq("soft")

    summaries = {}
    for threshold in thresholds:
        groups = []
        per_event = []
        for _, group in df.groupby("event_id", sort=True):
            if len(group) < threshold or group["soft_site"].nunique() < 2:
                continue
            group = group.copy()
            group["event_centered_amp"] = group["mean_amp"] - group["mean_amp"].mean()
            groups.append(group)
            auc, _ = _auc_from_scores(group["mean_amp"].to_numpy(float), group["soft_site"].to_numpy(bool))
            per_event.append(auc)
        if groups:
            pooled = pd.concat(groups, ignore_index=True)
            auc_raw, p_raw = _auc_from_scores(pooled["mean_amp"].to_numpy(float), pooled["soft_site"].to_numpy(bool))
            auc_centered, p_centered = _auc_from_scores(
                pooled["event_centered_amp"].to_numpy(float), pooled["soft_site"].to_numpy(bool)
            )
            per_event_vals = np.asarray([v for v in per_event if v is not None], dtype=float)
        else:
            pooled = pd.DataFrame()
            auc_raw = p_raw = auc_centered = p_centered = None
            per_event_vals = np.array([])

        summaries[f"min{threshold}"] = {
            "min_stations_per_event": int(threshold),
            "n_events": int(len(groups)),
            "n_station_events": int(len(pooled)),
            "n_soft_records": int(pooled["soft_site"].sum()) if len(pooled) else 0,
            "n_hard_records": int((~pooled["soft_site"]).sum()) if len(pooled) else 0,
            "auc_raw_soft_gt_hard": auc_raw,
            "auc_raw_p": p_raw,
            "auc_event_centered_soft_gt_hard": auc_centered,
            "auc_event_centered_p": p_centered,
            "median_event_auc": _safe_float(np.median(per_event_vals)) if len(per_event_vals) else None,
        }
    return {"name": name, "thresholds": summaries}


def summarize_events(
    event_records: list[dict],
    name: str,
    event_filter: Callable[[dict], bool],
    max_events_per_station: int,
    min_events_per_station: int,
) -> dict:
    grouped = defaultdict(list)
    for record in event_records:
        if event_filter(record):
            grouped[record["station_id"]].append(record)

    per_station = []
    for station_id in sorted(grouped):
        records = sorted(grouped[station_id], key=lambda x: x["cache_index"])[:max_events_per_station]
        if len(records) < min_events_per_station:
            continue
        station_curve = np.mean([r["hvsr"] for r in records], axis=0)
        per_station.append({
            "station_id": station_id,
            "mean_amp": float(np.mean(station_curve)),
            "vs30": float(records[0]["vs30"]),
            "nehrp": records[0]["nehrp"],
            "n_events": int(len(records)),
            "station_latitude_deg": _safe_float(records[0]["station_latitude_deg"]),
            "station_longitude_deg": _safe_float(records[0]["station_longitude_deg"]),
            "arrival_sample_min": _safe_float(min(r["arrival_sample"] for r in records)),
            "arrival_sample_median": _safe_float(np.median([r["arrival_sample"] for r in records])),
            "arrival_sample_max": _safe_float(max(r["arrival_sample"] for r in records)),
            "left_pad_max": _safe_float(max(r["left_pad"] for r in records)),
            "right_pad_max": _safe_float(max(r["right_pad"] for r in records)),
        })

    df = pd.DataFrame(per_station)
    if len(df) >= 3:
        amps = df["mean_amp"].to_numpy(float)
        vs30 = df["vs30"].to_numpy(float)
        rho, pval = _corr(amps, vs30, "spearman")
        rho_log, p_log = _corr(amps, np.log10(vs30), "spearman")
        pear, pear_p = _corr(amps, vs30, "pearson")
        partial_rho, partial_p = _partial_spearman_latlon(df, "mean_amp", "vs30")
    else:
        rho = pval = rho_log = p_log = pear = pear_p = None
        partial_rho = partial_p = None

    per_nehrp = {}
    if len(df):
        for cls, sub in df.groupby("nehrp"):
            if len(sub) >= 3:
                rho_c, p_c = _corr(sub["mean_amp"].to_numpy(float), sub["vs30"].to_numpy(float), "spearman")
                per_nehrp[str(cls)] = {
                    "N": int(len(sub)),
                    "spearman_rho": rho_c,
                    "spearman_p": p_c,
                }

    selected_event_ids = {
        (r["station_id"], r["cache_index"])
        for station_records in grouped.values()
        for r in sorted(station_records, key=lambda x: x["cache_index"])[:max_events_per_station]
    }

    return {
        "name": name,
        "n_stations": int(len(df)),
        "n_events_used": int(sum(row["n_events"] for row in per_station)),
        "n_events_passing_filter": int(sum(1 for r in event_records if event_filter(r))),
        "n_station_events_selected_before_min_event_filter": int(len(selected_event_ids)),
        "min_events_per_station": int(min_events_per_station),
        "max_events_per_station": int(max_events_per_station),
        "vs30_range": [
            _safe_float(df["vs30"].min()) if len(df) else None,
            _safe_float(df["vs30"].max()) if len(df) else None,
        ],
        "spearman_rho": rho,
        "spearman_p": pval,
        "spearman_rho_log_vs30": rho_log,
        "spearman_p_log_vs30": p_log,
        "pearson_r": pear,
        "pearson_p": pear_p,
        "partial_spearman_latlon_rho": partial_rho,
        "partial_spearman_latlon_p": partial_p,
        "per_nehrp": per_nehrp,
        "per_station": per_station,
    }


def print_analysis(label: str, analysis: dict):
    print(f"\n[{label}]")
    print(f"  Stations: {analysis['n_stations']}  Events used: {analysis['n_events_used']}")
    print(f"  Spearman rho = {analysis['spearman_rho']}  p = {analysis['spearman_p']}")
    print(f"  Pearson r    = {analysis['pearson_r']}  p = {analysis['pearson_p']}")
    for cls, item in analysis["per_nehrp"].items():
        print(f"  NEHRP {cls}: N={item['N']}  rho={item['spearman_rho']}  p={item['spearman_p']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/kiknet_measured_vs30_pwave_v1")
    parser.add_argument("--output-dir", default="outputs/expF_kiknet_jst_inference")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-rate-hz", type=float, default=None)
    parser.add_argument("--max-events-per-station", type=int, default=5)
    parser.add_argument("--qc-max-left-pad-samples", type=int, default=800)
    parser.add_argument("--qc-max-right-pad-samples", type=int, default=0)
    parser.add_argument("--qc-min-events-per-station", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    print("Loading checkpoint...")
    ce, dn, _ = load_checkpoint_models(
        args.checkpoint, dev, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if dropped:
        ce = AblationConditionEncoder(ce, dropped)
    dn.eval()
    ce.eval()
    total_samples = 3200

    print("Loading KiK-net cache with MLAAPDE vocab...")
    ds_train_ref = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds = SeismicWaveformDataset(
        args.data_dir, split="testing", augment=False,
        cache_prefix="kiknet_measured_vs30_pwave_v1",
        condition_version="v2.1", field_policy="default",
        vocab_from=ds_train_ref,
    )
    conditions = ds.conditions.iloc[ds.indices].copy()
    conditions["station_id"] = (
        conditions["station_network_code"].fillna("KIKNET").astype(str)
        + "." + conditions["station_code"].fillna("UNKNOWN").astype(str)
    )

    if args.sample_rate_hz is None:
        sample_rate_hz = float(pd.to_numeric(conditions["trace_sampling_rate_hz"], errors="coerce").median())
    else:
        sample_rate_hz = float(args.sample_rate_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate_hz}")
    if F_MAX >= sample_rate_hz / 2:
        raise ValueError(f"F_MAX={F_MAX} Hz must be below Nyquist={sample_rate_hz / 2} Hz")

    print(f"  Events: {len(conditions)}, stations: {conditions['station_id'].nunique()}")
    print(f"  Sample rate for spectra: {sample_rate_hz:g} Hz")

    cache_to_ds_pos = {int(cache_idx): pos for pos, cache_idx in enumerate(ds.indices)}

    torch.manual_seed(42)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(42)
    noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=dev)

    event_records = []
    sta_groups = conditions.groupby("station_id", sort=True)
    print(f"Inferring {len(sta_groups)} stations ({len(conditions)} station-event rows)...")

    for si, (sta_id, group) in enumerate(sta_groups):
        for _, row in group.sort_values("cache_index").iterrows():
            cache_idx = int(row["cache_index"])
            ds_pos = cache_to_ds_pos.get(cache_idx)
            if ds_pos is None:
                continue
            wf_tensor, cond_dict = ds[ds_pos]
            wf_np = wf_tensor.numpy()
            cond_gpu = {k: v.unsqueeze(0).to(dev) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)
            predicted = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu().numpy()[0]
            residual = wf_np - predicted
            hvsr = compute_hvsr(residual, predicted, sample_rate_hz=sample_rate_hz)
            event_records.append({
                "station_id": sta_id,
                "event_id": str(row.get("event_id", cache_idx)),
                "source_origin_time": str(row.get("source_origin_time", "")),
                "source_magnitude": _safe_float(row.get("source_magnitude")),
                "source_depth_km": _safe_float(row.get("source_depth_km")),
                "source_latitude_deg": _safe_float(row.get("source_latitude_deg")),
                "source_longitude_deg": _safe_float(row.get("source_longitude_deg")),
                "path_ep_distance_km": _safe_float(row.get("path_ep_distance_km")),
                "cache_index": cache_idx,
                "hvsr": hvsr,
                "vs30": float(row["vs30_m_s"]),
                "nehrp": str(row["nehrp_site_class"]),
                "station_latitude_deg": _safe_float(row.get("station_latitude_deg")),
                "station_longitude_deg": _safe_float(row.get("station_longitude_deg")),
                "arrival_sample": float(row["selected_phase_arrival_sample"]),
                "left_pad": float(row["model_left_pad_samples"]),
                "right_pad": float(row["model_right_pad_samples"]),
                "trace_n_samples": float(row["trace_n_samples"]),
                "sample_rate_hz": sample_rate_hz,
            })
        if (si + 1) % 20 == 0:
            print(f"  {si+1}/{len(sta_groups)} stations done")

    def all_events(_record: dict) -> bool:
        return True

    def arrival_window_qc(record: dict) -> bool:
        return (
            record["arrival_sample"] >= 0
            and record["arrival_sample"] < record["trace_n_samples"]
            and record["left_pad"] <= args.qc_max_left_pad_samples
            and record["right_pad"] <= args.qc_max_right_pad_samples
        )

    def full_pre_window_qc(record: dict) -> bool:
        return (
            record["arrival_sample"] >= 800
            and record["arrival_sample"] < record["trace_n_samples"]
            and record["left_pad"] == 0
            and record["right_pad"] <= args.qc_max_right_pad_samples
        )

    analyses = {
        "all_events": summarize_events(
            event_records, "all_events", all_events,
            args.max_events_per_station, min_events_per_station=1,
        ),
        "qc_arrival_window_min1": summarize_events(
            event_records, "qc_arrival_window_min1", arrival_window_qc,
            args.max_events_per_station, min_events_per_station=args.qc_min_events_per_station,
        ),
        "qc_arrival_window_min2": summarize_events(
            event_records, "qc_arrival_window_min2", arrival_window_qc,
            args.max_events_per_station, min_events_per_station=2,
        ),
        "qc_full_pre_window_min1": summarize_events(
            event_records, "qc_full_pre_window_min1", full_pre_window_qc,
            args.max_events_per_station, min_events_per_station=1,
        ),
    }

    single_event_summaries = {}
    single_event_tables = {}
    soft_hard_summaries = {}
    for label, event_filter in [
        ("all_events", all_events),
        ("qc_arrival_window", arrival_window_qc),
        ("qc_full_pre_window", full_pre_window_qc),
    ]:
        single_event_summaries[label], single_event_tables[label] = summarize_single_event_correlations(
            event_records, label, event_filter
        )
        soft_hard_summaries[label] = summarize_soft_hard_ordering(event_records, label, event_filter)

    print("\n" + "=" * 70)
    print("JsT-HVSR vs KiK-net measured Vs30")
    print("=" * 70)
    for label, analysis in analyses.items():
        print_analysis(label, analysis)
    print("\nSingle-earthquake validation")
    for label, summary in single_event_summaries.items():
        for threshold in ("min8", "min10"):
            item = summary["thresholds"].get(threshold, {})
            print(
                f"  {label} {threshold}: events={item.get('n_events')} "
                f"median rho={item.get('median_spearman_rho')} "
                f"event-fixed r={item.get('event_fixed_rank', {}).get('rank_r')} "
                f"p_perm={item.get('event_fixed_rank', {}).get('p_permutation_negative')}"
            )

    all_result = analyses["all_events"]
    qc_values = pd.DataFrame([{
        "arrival_sample": r["arrival_sample"],
        "left_pad": r["left_pad"],
        "right_pad": r["right_pad"],
    } for r in event_records])
    qc_summary = {
        "n_event_records": int(len(event_records)),
        "arrival_sample": {
            "min": _safe_float(qc_values["arrival_sample"].min()),
            "median": _safe_float(qc_values["arrival_sample"].median()),
            "max": _safe_float(qc_values["arrival_sample"].max()),
        },
        "left_pad": {
            "min": _safe_float(qc_values["left_pad"].min()),
            "median": _safe_float(qc_values["left_pad"].median()),
            "max": _safe_float(qc_values["left_pad"].max()),
        },
        "right_pad": {
            "min": _safe_float(qc_values["right_pad"].min()),
            "median": _safe_float(qc_values["right_pad"].median()),
            "max": _safe_float(qc_values["right_pad"].max()),
        },
    }

    results = {
        "n_stations": all_result["n_stations"],
        "n_events_total": int(len(conditions)),
        "vs30_range": all_result["vs30_range"],
        "spearman_rho": all_result["spearman_rho"],
        "spearman_p": all_result["spearman_p"],
        "spearman_rho_log_vs30": all_result["spearman_rho_log_vs30"],
        "spearman_p_log_vs30": all_result["spearman_p_log_vs30"],
        "pearson_r": all_result["pearson_r"],
        "pearson_p": all_result["pearson_p"],
        "sample_rate_hz": sample_rate_hz,
        "frequency_range_hz": [F_MIN, F_MAX],
        "frequency_bins": int(N_FREQ_BINS),
        "max_events_per_station": int(args.max_events_per_station),
        "qc_thresholds": {
            "max_left_pad_samples": int(args.qc_max_left_pad_samples),
            "max_right_pad_samples": int(args.qc_max_right_pad_samples),
            "arrival_must_be_within_trace": True,
        },
        "qc_summary": qc_summary,
        "analyses": analyses,
        "single_event": single_event_summaries,
        "soft_hard_ordering": soft_hard_summaries,
        "per_station": all_result["per_station"],
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    for label, analysis in analyses.items():
        pd.DataFrame(analysis["per_station"]).to_csv(out_dir / f"per_station_{label}.csv", index=False)
    event_records_dataframe(event_records).to_csv(out_dir / "per_station_event_records.csv", index=False)
    for label, table in single_event_tables.items():
        table.to_csv(out_dir / f"per_event_single_earthquake_{label}.csv", index=False)

    print(f"\nSaved: {out_dir / 'results.json'}")
    print(f"Saved: {out_dir / 'per_station_event_records.csv'}")
    print(f"Saved per-station CSVs in {out_dir}")
    print("\nContext:")
    print("  JsT-HVSR vs proxy Vs30 (US): rho = -0.59 (N=86)")
    print("  Standard HVSR geom vs measured Vs30 (JP): rho = -0.43 (N=656)")
    print("  Standard HVSR F0 vs measured Vs30 (JP): rho = +0.62 (N=554)")
    print(f"  JsT-HVSR vs measured Vs30 (JP, all): rho = {all_result['spearman_rho']} (N={all_result['n_stations']})")
    print(
        "  JsT-HVSR vs measured Vs30 (JP, arrival/window QC): "
        f"rho = {analyses['qc_arrival_window_min1']['spearman_rho']} "
        f"(N={analyses['qc_arrival_window_min1']['n_stations']})"
    )


if __name__ == "__main__":
    main()
