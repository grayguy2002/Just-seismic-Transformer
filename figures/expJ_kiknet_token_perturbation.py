"""Exp J: token-level leakage perturbations for dense KiK-net inference.

This experiment reruns JsT inference on the dense KiK-net measured-Vs30 cache
while perturbing the condition tokens actually consumed by the denoiser.  It is
designed as a leakage stress test: if the measured-site ordering is only a
receiver/path metadata proxy, shuffled donor tokens should either reproduce the
same Vs30 association through the donor metadata or dominate the residual score.

Scenarios:
  true_tokens        : original condition tokens.
  receiver_shuffle   : token 7 replaced by a same-event donor station.
  path_shuffle       : tokens 3--6 replaced by a same-event donor station.
  path_receiver_shuffle : tokens 3--7 replaced by a same-event donor station.
  receiver_zero      : token 7 set to zero.

The script writes per-record residual spectra and a JSON summary.  It does not
train or tune the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from JsT import SeismicWaveformDataset, load_checkpoint_models
from JsT.ablation import AblationConditionEncoder


N_FREQ_BINS = 40
F_MIN, F_MAX = 0.3, 15.0
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])
BINS = [f"hvsr_bin_{i:02d}" for i in range(N_FREQ_BINS)]
BANDS = {
    "0.5-5Hz": [i for i, f in enumerate(FREQ_CENTERS) if 0.5 <= f < 5.0],
    "1-3Hz": [i for i, f in enumerate(FREQ_CENTERS) if 1.0 <= f < 3.0],
    "1-10Hz": [i for i, f in enumerate(FREQ_CENTERS) if 1.0 <= f < 10.0],
    "3-10Hz": [i for i, f in enumerate(FREQ_CENTERS) if 3.0 <= f < 10.0],
}


@torch.no_grad()
def generate_waveform(dn, tokens: torch.Tensor, noise: torch.Tensor, steps: int = 50) -> torch.Tensor:
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


def partial_rank_latlon(df: pd.DataFrame, score_col: str, vs30_col: str = "true_vs30") -> tuple[float | None, float | None]:
    cols = [score_col, vs30_col, "station_latitude_deg", "station_longitude_deg"]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 8:
        return None, None
    x = _rank(sub[score_col])
    y = _rank(sub[vs30_col])
    cov = sub[["station_latitude_deg", "station_longitude_deg"]].to_numpy(float)
    xr = _residualize(x, cov)
    yr = _residualize(y, cov)
    r, p, _ = _pearson(xr, yr)
    return r, p


def add_qc_and_bands(df: pd.DataFrame) -> pd.DataFrame:
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


def compute_hvsr_batch(
    observed: np.ndarray,
    predicted: np.ndarray,
    sample_rate_hz: float,
    eps: float = 1e-12,
) -> np.ndarray:
    residual = observed - predicted
    freqs = np.fft.rfftfreq(observed.shape[-1], d=1.0 / sample_rate_hz)
    out = np.zeros((observed.shape[0], N_FREQ_BINS), dtype=np.float32)
    for b in range(N_FREQ_BINS):
        mask = (freqs >= FREQ_EDGES[b]) & (freqs < FREQ_EDGES[b + 1])
        if not mask.any():
            continue
        sr = np.abs(np.fft.rfft(residual, axis=-1))[:, :, mask].mean(axis=2)
        sp = np.abs(np.fft.rfft(predicted, axis=-1))[:, :, mask].mean(axis=2)
        out[:, b] = np.log10(np.maximum(sr, eps) / np.maximum(sp, eps)).mean(axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def same_event_donors(conditions: pd.DataFrame, seed: int = 20260622) -> np.ndarray:
    rng = np.random.default_rng(seed)
    donor = np.arange(len(conditions), dtype=int)
    for _, group in conditions.groupby("event_id", sort=True):
        idx = group.index.to_numpy(dtype=int)
        if len(idx) <= 1:
            continue
        perm = idx.copy()
        # Retry a few times to avoid self-donors; fall back to a roll.
        for _ in range(20):
            rng.shuffle(perm)
            if np.all(perm != idx):
                break
        if np.any(perm == idx):
            perm = np.roll(idx, 1)
        donor[idx] = perm
    return donor


def perturb_tokens(tokens: torch.Tensor, base_tokens_all: torch.Tensor, donor_idx: np.ndarray, scenario: str) -> torch.Tensor:
    out = tokens.clone()
    if scenario == "true_tokens":
        return out
    if scenario == "receiver_shuffle":
        out[:, 7, :] = base_tokens_all[donor_idx, 7, :].to(tokens.device)
        return out
    if scenario == "path_shuffle":
        out[:, 3:7, :] = base_tokens_all[donor_idx, 3:7, :].to(tokens.device)
        return out
    if scenario == "path_receiver_shuffle":
        out[:, 3:8, :] = base_tokens_all[donor_idx, 3:8, :].to(tokens.device)
        return out
    if scenario == "receiver_zero":
        out[:, 7, :] = 0.0
        return out
    raise ValueError(f"Unknown scenario: {scenario}")


def station_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    sub = df[df["arrival_qc"]].copy()
    return (
        sub.groupby(["scenario", "station_id"], as_index=False)
        .agg(
            score=(metric, "mean"),
            true_vs30=("true_vs30", "first"),
            n_records=(metric, "size"),
            n_events=("event_id", "nunique"),
            station_latitude_deg=("station_latitude_deg", "first"),
            station_longitude_deg=("station_longitude_deg", "first"),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["score", "true_vs30"])
    )


def summarize(records: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    rows = []
    summary = {}
    for scenario, group in records.groupby("scenario", sort=True):
        entry = {}
        for scope, sub in [
            ("all", group),
            ("arrival_qc", group[group["arrival_qc"]]),
            ("full_pre_qc", group[group["full_pre_qc"]]),
        ]:
            rho, p, n = _spearman(sub[metric], sub["true_vs30"])
            pr, pp = partial_rank_latlon(sub, metric)
            donor_rho, donor_p, donor_n = _spearman(sub[metric], sub["donor_vs30"])
            rows.append(
                {
                    "scenario": scenario,
                    "scope": scope,
                    "aggregation": "records",
                    "N": n,
                    "statistic": "spearman_to_true_vs30",
                    "value": rho,
                    "p": p,
                    "partial_latlon_r": pr,
                    "partial_latlon_p": pp,
                    "donor_spearman": donor_rho,
                    "donor_spearman_p": donor_p,
                    "donor_N": donor_n,
                }
            )
            entry[f"{scope}_records"] = {
                "N": n,
                "rho_true_vs30": rho,
                "p_true_vs30": p,
                "partial_latlon_r": pr,
                "partial_latlon_p": pp,
                "rho_donor_vs30": donor_rho,
                "p_donor_vs30": donor_p,
            }

        means = station_means(group, metric)
        for min_events in [1, 2]:
            sub = means[means["n_events"] >= min_events]
            rho, p, n = _spearman(sub["score"], sub["true_vs30"])
            pr, pp = partial_rank_latlon(sub.rename(columns={"score": metric}), metric)
            rows.append(
                {
                    "scenario": scenario,
                    "scope": "arrival_qc",
                    "aggregation": f"station_mean_min{min_events}",
                    "N": n,
                    "statistic": "spearman_to_true_vs30",
                    "value": rho,
                    "p": p,
                    "partial_latlon_r": pr,
                    "partial_latlon_p": pp,
                    "donor_spearman": None,
                    "donor_spearman_p": None,
                    "donor_N": None,
                }
            )
            entry[f"station_mean_min{min_events}"] = {
                "N": n,
                "rho_true_vs30": rho,
                "p_true_vs30": p,
                "partial_latlon_r": pr,
                "partial_latlon_p": pp,
            }
        summary[scenario] = entry
    table = pd.DataFrame(rows)
    return table, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/run036/checkpoint-last.pth")
    parser.add_argument("--data-dir", default="data/kiknet_dense_arrival_qc_events_v1")
    parser.add_argument("--train-data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--output-dir", default="outputs/expJ_kiknet_token_perturbation")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample-rate-hz", type=float, default=None)
    parser.add_argument("--metric", default="1-10Hz")
    parser.add_argument(
        "--scenarios",
        default="true_tokens,receiver_shuffle,path_shuffle,path_receiver_shuffle,receiver_zero",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading checkpoint...")
    ce, dn, _ = load_checkpoint_models(
        args.checkpoint,
        dev,
        use_ema=True,
        sampling_method="heun",
        steps=args.steps,
        cfg_scale=1.0,
    )
    if dropped:
        ce = AblationConditionEncoder(ce, dropped)
    ce.eval()
    dn.eval()

    print("Loading datasets...")
    ds_train_ref = SeismicWaveformDataset(
        args.train_data_dir,
        split="training",
        augment=False,
        cache_prefix="pwave_v21",
        condition_version="v2.1",
        field_policy="default",
    )
    ds = SeismicWaveformDataset(
        args.data_dir,
        split="testing",
        augment=False,
        cache_prefix="kiknet_measured_vs30_pwave_v1",
        condition_version="v2.1",
        field_policy="default",
        vocab_from=ds_train_ref,
    )

    conditions = ds.conditions.iloc[ds.indices].copy().reset_index(drop=False).rename(columns={"index": "cache_row"})
    conditions["station_id"] = (
        conditions["station_network_code"].fillna("KIKNET").astype(str)
        + "."
        + conditions["station_code"].fillna("UNKNOWN").astype(str)
    )
    sample_rate_hz = (
        float(args.sample_rate_hz)
        if args.sample_rate_hz is not None
        else float(pd.to_numeric(conditions["trace_sampling_rate_hz"], errors="coerce").median())
    )
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate_hz}")
    if F_MAX >= sample_rate_hz / 2:
        raise ValueError(f"F_MAX={F_MAX} Hz must be below Nyquist={sample_rate_hz / 2} Hz")

    donor = same_event_donors(conditions)
    donor_meta = conditions.iloc[donor].reset_index(drop=True)
    print(f"Records: {len(conditions)}; stations: {conditions['station_id'].nunique()}; scenarios: {scenarios}")

    torch.manual_seed(42)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(42)
    total_samples = int(ds.waveforms.shape[2])
    noise_base = dn.noise_scale * torch.randn(1, 3, total_samples, device=dev)

    # Precompute all condition tokens once. This also fixes donor-token lookup.
    print("Encoding condition tokens...")
    token_chunks = []
    observed_chunks = []
    for start in range(0, len(ds), args.batch_size):
        stop = min(start + args.batch_size, len(ds))
        waveforms = []
        conds = []
        for pos in range(start, stop):
            wf, cond = ds[pos]
            waveforms.append(wf.numpy())
            conds.append({k: v.unsqueeze(0).to(dev) for k, v in cond.items()})
        # Stack manually to avoid importing a second collate path.
        batch_cond = {}
        for key in conds[0]:
            batch_cond[key] = torch.cat([c[key] for c in conds], dim=0)
        tokens = ce(batch_cond).detach().cpu()
        token_chunks.append(tokens)
        observed_chunks.append(np.stack(waveforms, axis=0))
    tokens_all_cpu = torch.cat(token_chunks, dim=0)
    observed_all = np.concatenate(observed_chunks, axis=0)
    if len(tokens_all_cpu) != len(conditions):
        raise RuntimeError("Token/condition length mismatch")

    rows = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario}")
        for start in range(0, len(conditions), args.batch_size):
            stop = min(start + args.batch_size, len(conditions))
            batch_tokens = tokens_all_cpu[start:stop].to(dev)
            batch_donor = donor[start:stop]
            batch_tokens = perturb_tokens(batch_tokens, tokens_all_cpu, batch_donor, scenario)
            noise = noise_base.expand(stop - start, -1, -1).clone()
            predicted = generate_waveform(dn, batch_tokens, noise, steps=args.steps).detach().cpu().numpy()
            hvsr = compute_hvsr_batch(observed_all[start:stop], predicted, sample_rate_hz)
            for local_i, spec in enumerate(hvsr):
                i = start + local_i
                row = conditions.iloc[i]
                drow = donor_meta.iloc[i]
                rec = {
                    "scenario": scenario,
                    "event_id": str(row["event_id"]),
                    "station_id": str(row["station_id"]),
                    "cache_index": int(row["cache_index"]),
                    "donor_index": int(donor[i]),
                    "donor_station_id": str(drow["station_id"]),
                    "true_vs30": float(row["vs30_m_s"]),
                    "donor_vs30": float(drow["vs30_m_s"]),
                    "nehrp": str(row["nehrp_site_class"]),
                    "donor_nehrp": str(drow["nehrp_site_class"]),
                    "station_latitude_deg": float(row["station_latitude_deg"]),
                    "station_longitude_deg": float(row["station_longitude_deg"]),
                    "donor_station_latitude_deg": float(drow["station_latitude_deg"]),
                    "donor_station_longitude_deg": float(drow["station_longitude_deg"]),
                    "source_latitude_deg": float(row["source_latitude_deg"]),
                    "source_longitude_deg": float(row["source_longitude_deg"]),
                    "source_depth_km": float(row["source_depth_km"]),
                    "source_magnitude": float(row["source_magnitude"]),
                    "path_ep_distance_km": float(row["path_ep_distance_km"]),
                    "arrival_sample": float(row["selected_phase_arrival_sample"]),
                    "left_pad": float(row["model_left_pad_samples"]),
                    "right_pad": float(row["model_right_pad_samples"]),
                    "trace_n_samples": float(row["trace_n_samples"]),
                    "sample_rate_hz": sample_rate_hz,
                }
                for b, value in enumerate(spec):
                    rec[f"hvsr_bin_{b:02d}"] = float(value)
                rows.append(rec)
            if (stop % (args.batch_size * 5) == 0) or stop == len(conditions):
                print(f"  {scenario}: {stop}/{len(conditions)}")

    records = add_qc_and_bands(pd.DataFrame(rows))
    if args.metric not in records.columns:
        raise ValueError(f"Metric {args.metric!r} not available")
    summary_table, summary = summarize(records, args.metric)

    records.to_csv(out_dir / "per_record_token_perturbation.csv", index=False)
    summary_table.to_csv(out_dir / "summary_table.csv", index=False)
    payload = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "n_records_base": int(len(conditions)),
        "n_stations": int(conditions["station_id"].nunique()),
        "sample_rate_hz": sample_rate_hz,
        "frequency_range_hz": [F_MIN, F_MAX],
        "metric": args.metric,
        "scenarios": scenarios,
        "dropped_tokens": dropped,
        "record_counts": {
            "all_records_per_scenario": int(len(conditions)),
            "arrival_qc_records_per_scenario": int(records[records["scenario"].eq(scenarios[0])]["arrival_qc"].sum()),
            "full_pre_qc_records_per_scenario": int(records[records["scenario"].eq(scenarios[0])]["full_pre_qc"].sum()),
        },
        "summary": summary,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
