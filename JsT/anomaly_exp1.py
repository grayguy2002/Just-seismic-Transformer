#!/usr/bin/env python3
"""Experiment 1: conditional generative anomaly scoring with JsT run019."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from JsT import Denoiser, SeismicConditionEncoder, load_checkpoint_models
from JsT.dataset import SeismicWaveformDataset, collate_conditions

EPS = 1e-8
RUN_NAME = "run019"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else project_root() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score held-out waveforms against JsT conditional generation distribution."
    )
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v1")
    parser.add_argument("--cache-prefix", default="pwave_v1")
    parser.add_argument("--condition-schema", choices=["auto", "legacy", "v2.1", "v3"], default="auto")
    parser.add_argument("--checkpoint", default=f"outputs/{RUN_NAME}/checkpoint-last.pth")
    parser.add_argument("--output-dir", default=f"outputs/anomaly_exp1_{RUN_NAME}")
    parser.add_argument("--split", default="testing")
    parser.add_argument("--max-samples", type=int, default=512, help="0 means the full split")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-gen", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampling-method", choices=["heun", "euler"], default="heun")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--envelope-kernel", type=int, default=41)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--rank-by", default="combined_z")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    use_ema: bool,
    sampling_method: str,
    steps: int,
    cfg_scale: float,
) -> tuple[SeismicConditionEncoder, Denoiser, dict[str, Any]]:
    return load_checkpoint_models(
        checkpoint_path,
        device,
        use_ema=use_ema,
        sampling_method=sampling_method,
        steps=steps,
        cfg_scale=cfg_scale,
    )


def checkpoint_condition_schema(ckpt: dict[str, Any]) -> str:
    cfg = ckpt.get("arch", {}).get("condition_encoder", {})
    ev = cfg.get("encoder_version", "v1")
    if ev in ("v2.1", "v3"):
        return "v2.1"  # v2.1 and v3 share the same data schema
    return "legacy"


def build_datasets(
    data_dir: Path,
    split: str,
    cache_prefix: str,
    condition_schema: str,
) -> tuple[SeismicWaveformDataset, SeismicWaveformDataset]:
    train = SeismicWaveformDataset(
        data_dir,
        split="training",
        augment=False,
        cache_prefix=cache_prefix,
        condition_version=condition_schema,
    )
    eval_ds = SeismicWaveformDataset(
        data_dir,
        split=split,
        augment=False,
        vocab_from=train,
        cache_prefix=cache_prefix,
        condition_version=condition_schema,
    )
    return train, eval_ds


def encode_conditions(
    ce: SeismicConditionEncoder,
    cond: dict[str, torch.Tensor],
) -> torch.Tensor:
    return ce(cond)


def validate_vocab_alignment(train_ds: SeismicWaveformDataset, ckpt_vocab: dict[str, list[str]]) -> None:
    checks = [
        ("magtype", train_ds.magtype_vocab),
        ("phase", train_ds.phase_vocab),
        ("channel", train_ds.channel_vocab),
        ("network", train_ds.network_vocab),
    ]
    if ckpt_vocab.get("station_id") is not None:
        checks.append(("station_id", train_ds.station_id_vocab))
    if ckpt_vocab.get("station_location") is not None:
        checks.append(("station_location", train_ds.station_location_vocab))
    if ckpt_vocab.get("source_magnitude_author") is not None:
        checks.append(("source_magnitude_author", train_ds.source_magnitude_author_vocab))
    if ckpt_vocab.get("phase_status") is not None:
        checks.append(("phase_status", train_ds.phase_status_vocab))
    mismatches = []
    for key, ds_vocab in checks:
        ckpt_values = list(ckpt_vocab[key])
        if list(ds_vocab) != ckpt_values:
            mismatches.append((key, list(ds_vocab), ckpt_values))
    if mismatches:
        msg = ["Dataset vocab does not match checkpoint vocab; categorical IDs would be invalid."]
        for key, ds_vocab, ckpt_values in mismatches:
            msg.append(f"{key}: dataset={ds_vocab} checkpoint={ckpt_values}")
        raise ValueError("\n".join(msg))


def selected_split_indices(n_total: int, start: int, max_samples: int) -> list[int]:
    if start < 0 or start >= n_total:
        raise ValueError(f"start-index {start} is outside split length {n_total}")
    end = n_total if max_samples == 0 else min(n_total, start + max_samples)
    if end <= start:
        raise ValueError("No samples selected")
    return list(range(start, end))


def move_cond(cond: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in cond.items()}


def relative_l2_stack(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = (a - b.unsqueeze(0)).flatten(2).norm(dim=-1)
    denom = b.flatten(1).norm(dim=-1).clamp_min(EPS)
    return diff / denom.unsqueeze(0)


def smooth_envelope(w: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return w.abs()
    if kernel % 2 == 0:
        kernel += 1
    original_shape = w.shape
    flat = w.abs().reshape(-1, original_shape[-2], original_shape[-1])
    smoothed = F.avg_pool1d(flat, kernel_size=kernel, stride=1, padding=kernel // 2)
    return smoothed.reshape(original_shape)


def spectral_log_amp(w: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.fft.rfft(w.float(), dim=-1).abs())


def metric_min_mean_nearest(dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_vals, nearest = dist.min(dim=0)
    return min_vals, dist.mean(dim=0), nearest


def compute_batch_metrics(real: torch.Tensor, gens: torch.Tensor, envelope_kernel: int) -> dict[str, torch.Tensor]:
    time_dist = relative_l2_stack(gens, real)
    time_min, time_mean, time_nearest = metric_min_mean_nearest(time_dist)

    env_dist = relative_l2_stack(smooth_envelope(gens, envelope_kernel), smooth_envelope(real, envelope_kernel))
    env_min, env_mean, env_nearest = metric_min_mean_nearest(env_dist)

    spec_dist = relative_l2_stack(spectral_log_amp(gens), spectral_log_amp(real))
    spec_min, spec_mean, spec_nearest = metric_min_mean_nearest(spec_dist)

    real_peak = real.abs().flatten(1).max(dim=1).values
    gen_peak = gens.abs().flatten(2).max(dim=2).values
    gen_peak_mean = gen_peak.mean(dim=0)
    gen_peak_std = gen_peak.std(dim=0, unbiased=False)
    gen_peak_median = gen_peak.median(dim=0).values
    gen_peak_min = gen_peak.min(dim=0).values
    gen_peak_max = gen_peak.max(dim=0).values
    peak_ratio = real_peak / gen_peak_median.clamp_min(EPS)
    peak_log_ratio_abs = torch.log(peak_ratio.clamp_min(EPS)).abs()
    peak_abs_z = (real_peak - gen_peak_mean).abs() / gen_peak_std.clamp_min(1e-4)

    combined_raw = (
        0.35 * time_min
        + 0.25 * env_min
        + 0.25 * spec_min
        + 0.15 * peak_log_ratio_abs
    )

    return {
        "time_rel_l2_min": time_min,
        "time_rel_l2_mean": time_mean,
        "time_rel_l2_nearest_k": time_nearest,
        "envelope_rel_l2_min": env_min,
        "envelope_rel_l2_mean": env_mean,
        "envelope_rel_l2_nearest_k": env_nearest,
        "spectral_rel_l2_min": spec_min,
        "spectral_rel_l2_mean": spec_mean,
        "spectral_rel_l2_nearest_k": spec_nearest,
        "real_peak_abs_norm": real_peak,
        "gen_peak_mean_abs_norm": gen_peak_mean,
        "gen_peak_std_abs_norm": gen_peak_std,
        "gen_peak_median_abs_norm": gen_peak_median,
        "gen_peak_min_abs_norm": gen_peak_min,
        "gen_peak_max_abs_norm": gen_peak_max,
        "peak_ratio_to_gen_median": peak_ratio,
        "peak_log_ratio_abs": peak_log_ratio_abs,
        "peak_abs_z": peak_abs_z,
        "combined_raw_weighted": combined_raw,
    }


def tensor_dict_to_numpy(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    out = {}
    for k, v in metrics.items():
        out[k] = v.detach().cpu().numpy()
    return out


def safe_value(row: pd.Series, key: str) -> Any:
    if key not in row.index:
        return None
    val = row[key]
    if pd.isna(val):
        return None
    if isinstance(val, np.generic):
        return val.item()
    return val


def base_record(
    eval_ds: SeismicWaveformDataset,
    split_name: str,
    split_index: int,
    cond_cpu: dict[str, torch.Tensor],
    batch_pos: int,
) -> dict[str, Any]:
    cache_index = int(eval_ds.indices[split_index])
    row = eval_ds.conditions.iloc[cache_index]
    record: dict[str, Any] = {
        "split": split_name,
        "split_index": int(split_index),
        "cache_index": cache_index,
    }

    metadata_keys = [
        "event_id",
        "phase_id",
        "waves_id",
        "trace_name",
        "station_code",
        "station_network_code",
        "source_id",
    ]
    condition_keys = [
        "source_magnitude",
        "source_depth_km",
        "path_ep_distance_deg",
        "path_ep_distance_km",
        "path_azimuth_deg",
        "path_back_azimuth_deg",
        "phase_travel_sec",
        "residual_travel_sec",
        "normalization_scale",
        "source_magnitude_type",
        "selected_phase",
        "trace_channel",
        "station_latitude_deg",
        "station_longitude_deg",
        "station_elevation_m",
        "source_latitude_deg",
        "source_longitude_deg",
    ]
    for key in metadata_keys + condition_keys:
        record[key] = safe_value(row, key)

    id_map = {
        "source_magnitude_type_id": "source_magnitude_type",
        "selected_phase_id": "selected_phase",
        "trace_channel_id": "trace_channel",
        "station_network_code_id": "station_network_code",
    }
    for out_key, cond_key in id_map.items():
        record[out_key] = int(cond_cpu[cond_key][batch_pos].item())
    return record


def add_combined_z(df: pd.DataFrame) -> pd.DataFrame:
    components = [
        "time_rel_l2_min",
        "envelope_rel_l2_min",
        "spectral_rel_l2_min",
        "peak_log_ratio_abs",
    ]
    z_cols = []
    for col in components:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        z_col = f"{col}_z"
        if not math.isfinite(std) or std < EPS:
            df[z_col] = 0.0
        else:
            df[z_col] = (df[col] - mean) / std
        z_cols.append(z_col)
    df["combined_z"] = df[z_cols].mean(axis=1)
    return df


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for col in cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values) == 0:
            continue
        summary[col] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        }
    return summary


def score_dataset(
    ce: SeismicConditionEncoder,
    dn: Denoiser,
    eval_ds: SeismicWaveformDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    split_indices = selected_split_indices(len(eval_ds), args.start_index, args.max_samples)
    subset = Subset(eval_ds, split_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_conditions,
    )

    records: list[dict[str, Any]] = []
    processed = 0
    t0 = time.time()
    print(f"Scoring {len(split_indices)} samples from split '{args.split}' with K={args.num_gen} generations")

    for batch_idx, (real_cpu, cond_cpu) in enumerate(loader):
        real = real_cpu.to(device, non_blocking=True)
        cond = move_cond(cond_cpu, device)
        with torch.no_grad():
            cond_tokens = encode_conditions(ce, cond)
            gens = []
            for k in range(args.num_gen):
                torch.manual_seed(args.seed + batch_idx * args.num_gen + k)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + batch_idx * args.num_gen + k)
                gens.append(dn.generate(cond_tokens, steps=args.steps))
            gen_stack = torch.stack(gens, dim=0)
            metrics = tensor_dict_to_numpy(compute_batch_metrics(real, gen_stack, args.envelope_kernel))

        bsz = real_cpu.shape[0]
        norm_scale = []
        for j in range(bsz):
            split_index = split_indices[processed + j]
            cache_index = int(eval_ds.indices[split_index])
            norm_scale.append(float(eval_ds.conditions.iloc[cache_index]["normalization_scale"]))
        for j in range(bsz):
            split_index = split_indices[processed + j]
            record = base_record(eval_ds, args.split, split_index, cond_cpu, j)
            for key, values in metrics.items():
                value = values[j]
                if np.issubdtype(np.asarray(value).dtype, np.integer):
                    record[key] = int(value)
                else:
                    record[key] = float(value)
            record["real_peak_counts"] = record["real_peak_abs_norm"] * float(norm_scale[j])
            record["gen_peak_mean_counts"] = record["gen_peak_mean_abs_norm"] * float(norm_scale[j])
            records.append(record)

        processed += bsz
        elapsed = time.time() - t0
        print(f"  processed {processed:5d}/{len(split_indices)} samples ({elapsed:.1f}s)", flush=True)

    df = pd.DataFrame.from_records(records)
    return add_combined_z(df)


def write_outputs(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_path: Path,
    ckpt: dict[str, Any],
    split_size: int,
    elapsed: float,
    device: torch.device,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{args.run_name}"
    scores_path = output_dir / f"scores_{suffix}.csv"
    top_path = output_dir / f"top_anomalies_{suffix}.csv"
    summary_path = output_dir / f"summary_{suffix}.json"

    df.to_csv(scores_path, index=False)
    rank_by = args.rank_by
    if rank_by not in df.columns:
        raise ValueError(f"rank-by column '{rank_by}' not found in results")
    df.sort_values(rank_by, ascending=False).head(args.top_n).to_csv(top_path, index=False)

    metric_cols = [
        "time_rel_l2_min",
        "envelope_rel_l2_min",
        "spectral_rel_l2_min",
        "peak_log_ratio_abs",
        "combined_raw_weighted",
        "combined_z",
        "real_peak_abs_norm",
        "gen_peak_mean_abs_norm",
        "real_peak_counts",
        "gen_peak_mean_counts",
    ]
    summary = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "split_size": int(split_size),
        "evaluated_samples": int(len(df)),
        "max_samples": int(args.max_samples),
        "start_index": int(args.start_index),
        "num_gen": int(args.num_gen),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "sampling_method": args.sampling_method,
        "cfg_scale": float(args.cfg_scale),
        "use_ema": bool(args.use_ema),
        "seed": int(args.seed),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
        "rank_by": rank_by,
        "metric_summary": numeric_summary(df, metric_cols),
        "outputs": {
            "scores_csv": str(scores_path),
            "top_anomalies_csv": str(top_path),
            "summary_json": str(summary_path),
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if args.num_gen <= 0:
        raise ValueError("--num-gen must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    data_dir = resolve_project_path(args.data_dir)
    checkpoint_path = resolve_project_path(args.checkpoint)
    output_dir = resolve_project_path(args.output_dir)
    device = select_device(args.device)

    print(f"Project root: {project_root()}")
    print(f"Data dir:     {data_dir}")
    print(f"Checkpoint:   {checkpoint_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Device:       {device}")

    t0 = time.time()
    ce, dn, ckpt = load_model(
        checkpoint_path,
        device=device,
        use_ema=args.use_ema,
        sampling_method=args.sampling_method,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
    )
    condition_schema = checkpoint_condition_schema(ckpt) if args.condition_schema == "auto" else args.condition_schema
    train_ds, eval_ds = build_datasets(data_dir, args.split, args.cache_prefix, condition_schema)
    validate_vocab_alignment(train_ds, ckpt["vocab"])
    print(f"Dataset sizes: training={len(train_ds)} {args.split}={len(eval_ds)}")

    df = score_dataset(ce, dn, eval_ds, args, device)
    elapsed = time.time() - t0
    summary = write_outputs(df, args, output_dir, checkpoint_path, ckpt, len(eval_ds), elapsed, device)

    print("Done.")
    print(f"Scores:        {summary['outputs']['scores_csv']}")
    print(f"Top anomalies: {summary['outputs']['top_anomalies_csv']}")
    print(f"Summary:       {summary['outputs']['summary_json']}")


if __name__ == "__main__":
    main()
