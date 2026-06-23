#!/usr/bin/env python3
"""Generate visual waveform audit figures for JsT/empirical disagreement samples."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from JsT.anomaly_exp1 import (
    build_datasets,
    checkpoint_condition_schema,
    compute_batch_metrics,
    encode_conditions,
    move_cond,
    resolve_project_path,
    select_device,
    spectral_log_amp,
    smooth_envelope,
    tensor_dict_to_numpy,
    validate_vocab_alignment,
)
from JsT.checkpoint import _build_from_arch, _build_legacy, _load_ema, load_checkpoint_models
from JsT.dataset import collate_conditions

RUNS = {
    "run020_last": "outputs/run020/checkpoint-last.pth",
    "run023_last": "outputs/run023/checkpoint-last.pth",
}
CHANNELS = ["E", "N", "Z"]


def load_audit_model(
    checkpoint: Path,
    device: torch.device,
    args: argparse.Namespace,
):
    return load_checkpoint_models(
        checkpoint,
        device=device,
        use_ema=args.use_ema,
        sampling_method=args.sampling_method,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual audit figures for disagreement samples.")
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--cache-prefix", default="pwave_v21")
    parser.add_argument("--input-dir", default="outputs/disagreement_audit_run020_run023")
    parser.add_argument("--output-dir", default="outputs/waveform_audit_run020_run023")
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--num-gen", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampling-method", choices=["heun", "euler"], default="heun")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--envelope-kernel", type=int, default=41)
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    return parser.parse_args()


def load_selection(input_dir: Path, samples_per_group: int) -> pd.DataFrame:
    specs = [
        ("jst_morphology_only", input_dir / "top_jst_morphology_vs_empirical_jst_only.csv"),
        ("empirical_only", input_dir / "top_jst_morphology_vs_empirical_empirical_only.csv"),
    ]
    frames = []
    for group, path in specs:
        df = pd.read_csv(path).head(samples_per_group).copy()
        df["audit_group"] = group
        frames.append(df)
    selected = pd.concat(frames, ignore_index=True)
    return selected.drop_duplicates("cache_index").reset_index(drop=True)


def build_eval_subset(data_dir: Path, cache_prefix: str, condition_schema: str, cache_indices: list[int]):
    train_ds, eval_ds = build_datasets(data_dir, "testing", cache_prefix, condition_schema)
    cache_to_split = {int(cache_index): i for i, cache_index in enumerate(eval_ds.indices.tolist())}
    split_indices = [cache_to_split[int(cache_index)] for cache_index in cache_indices]
    return train_ds, eval_ds, split_indices


def sample_model(
    run_name: str,
    checkpoint: Path,
    data_dir: Path,
    cache_prefix: str,
    cache_indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, float]], dict[int, np.ndarray]]:
    ce, dn, ckpt = load_audit_model(checkpoint, device, args)
    condition_schema = checkpoint_condition_schema(ckpt)
    train_ds, eval_ds, split_indices = build_eval_subset(data_dir, cache_prefix, condition_schema, cache_indices)
    validate_vocab_alignment(train_ds, ckpt["vocab"])
    subset = Subset(eval_ds, split_indices)
    loader = DataLoader(subset, batch_size=len(split_indices), shuffle=False, num_workers=0, collate_fn=collate_conditions)
    real_cpu, cond_cpu = next(iter(loader))
    real = real_cpu.to(device)
    cond = move_cond(cond_cpu, device)
    gens = []
    with torch.no_grad():
        cond_tokens = encode_conditions(ce, cond)
        for k in range(args.num_gen):
            torch.manual_seed(args.seed + 1000 * (0 if run_name == "run020_last" else 1) + k)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 1000 * (0 if run_name == "run020_last" else 1) + k)
            gens.append(dn.generate(cond_tokens, steps=args.steps))
        gen_stack = torch.stack(gens, dim=0)
        metrics = tensor_dict_to_numpy(compute_batch_metrics(real, gen_stack, args.envelope_kernel))
    gen_by_cache: dict[int, np.ndarray] = {}
    metric_by_cache: dict[int, dict[str, float]] = {}
    real_by_cache: dict[int, np.ndarray] = {}
    for pos, split_index in enumerate(split_indices):
        cache_index = int(eval_ds.indices[split_index])
        gen_by_cache[cache_index] = gen_stack[:, pos].detach().cpu().numpy()
        real_by_cache[cache_index] = real_cpu[pos].numpy()
        metric_by_cache[cache_index] = {key: float(values[pos]) for key, values in metrics.items() if np.asarray(values[pos]).ndim == 0}
    return gen_by_cache, metric_by_cache, real_by_cache


def envelope_np(x: np.ndarray, kernel: int) -> np.ndarray:
    t = torch.from_numpy(x[None]).float()
    return smooth_envelope(t, kernel).squeeze(0).numpy()


def spectrum_np(x: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(x[None]).float()
    return spectral_log_amp(t).squeeze(0).numpy()


def plot_sample(
    row: pd.Series,
    real: np.ndarray,
    gens_by_run: dict[str, np.ndarray],
    metrics_by_run: dict[str, dict[str, float]],
    output_path: Path,
    envelope_kernel: int,
) -> None:
    time = np.arange(real.shape[-1]) / 40.0 - 20.0
    freqs = np.fft.rfftfreq(real.shape[-1], d=1.0 / 40.0)
    real_env = envelope_np(real, envelope_kernel)
    real_spec = spectrum_np(real)
    fig = plt.figure(figsize=(18, 13))
    grid = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 0.95])

    for ch in range(3):
        ax = fig.add_subplot(grid[ch, 0])
        ax.plot(time, real[ch], color="black", lw=0.8, label="real")
        for run_name, gens in gens_by_run.items():
            gen_med = np.median(gens[:, ch, :], axis=0)
            gen_q10 = np.quantile(gens[:, ch, :], 0.10, axis=0)
            gen_q90 = np.quantile(gens[:, ch, :], 0.90, axis=0)
            color = "tab:blue" if run_name == "run020_last" else "tab:orange"
            ax.plot(time, gen_med, color=color, lw=0.7, label=f"{run_name} median" if ch == 0 else None)
            ax.fill_between(time, gen_q10, gen_q90, color=color, alpha=0.12)
        ax.axvline(0.0, color="red", lw=0.7, alpha=0.7)
        ax.set_ylabel(CHANNELS[ch])
        if ch == 0:
            ax.legend(loc="upper right", fontsize=8)
        if ch == 2:
            ax.set_xlabel("seconds from selected arrival")
        ax.set_title("Waveform" if ch == 0 else "")

    ax_env = fig.add_subplot(grid[0:2, 1])
    real_env_sum = real_env.mean(axis=0)
    ax_env.plot(time, real_env_sum, color="black", lw=1.2, label="real envelope")
    for run_name, gens in gens_by_run.items():
        envs = np.stack([envelope_np(g, envelope_kernel).mean(axis=0) for g in gens], axis=0)
        color = "tab:blue" if run_name == "run020_last" else "tab:orange"
        ax_env.plot(time, np.median(envs, axis=0), color=color, lw=1.0, label=f"{run_name} median")
        ax_env.fill_between(time, np.quantile(envs, 0.1, axis=0), np.quantile(envs, 0.9, axis=0), color=color, alpha=0.15)
    ax_env.axvline(0.0, color="red", lw=0.7, alpha=0.7)
    ax_env.set_title("Mean absolute envelope")
    ax_env.legend(fontsize=8)

    ax_spec = fig.add_subplot(grid[2:4, 1])
    ax_spec.plot(freqs, real_spec.mean(axis=0), color="black", lw=1.2, label="real spectrum")
    for run_name, gens in gens_by_run.items():
        specs = np.stack([spectrum_np(g).mean(axis=0) for g in gens], axis=0)
        color = "tab:blue" if run_name == "run020_last" else "tab:orange"
        ax_spec.plot(freqs, np.median(specs, axis=0), color=color, lw=1.0, label=f"{run_name} median")
        ax_spec.fill_between(freqs, np.quantile(specs, 0.1, axis=0), np.quantile(specs, 0.9, axis=0), color=color, alpha=0.15)
    ax_spec.set_xlim(0, 20)
    ax_spec.set_title("Mean log spectrum")
    ax_spec.set_xlabel("Hz")
    ax_spec.legend(fontsize=8)

    ax_text = fig.add_subplot(grid[4, :])
    ax_text.axis("off")
    metric_lines = []
    for run_name, metrics in metrics_by_run.items():
        metric_lines.append(
            f"{run_name}: time={metrics.get('time_rel_l2_min', np.nan):.3f}, "
            f"env={metrics.get('envelope_rel_l2_min', np.nan):.3f}, "
            f"spec={metrics.get('spectral_rel_l2_min', np.nan):.3f}, "
            f"peak_log={metrics.get('peak_log_ratio_abs', np.nan):.3f}, "
            f"combined={metrics.get('combined_raw_weighted', np.nan):.3f}"
        )
    text = (
        f"group={row.get('audit_group')} cache={int(row.cache_index)} event={row.get('event_id')} "
        f"station={row.get('station_network_code')}.{row.get('station_code')} phase={row.get('selected_phase')} channel={row.get('trace_channel')}\n"
        f"M={row.get('source_magnitude')}, depth={row.get('source_depth_km')} km, dist={row.get('path_ep_distance_deg'):.2f} deg, "
        f"log_peak={row.get('log_peak_counts'):.2f}, post/pre={row.get('post_pre_log_rms_ratio'):.2f}, centroid={row.get('spectral_centroid_hz'):.2f} Hz\n"
        f"empirical: global={row.get('global_empirical_score'):.3f}, bin={row.get('condition_bin_empirical_score'):.3f}, nn={row.get('condition_nn_empirical_score'):.3f}\n"
        + "\n".join(metric_lines)
    )
    ax_text.text(0.01, 0.98, text, va="top", ha="left", family="monospace", fontsize=10)
    fig.suptitle(f"{row.get('audit_group')} | cache {int(row.cache_index)}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    data_dir = resolve_project_path(args.data_dir)
    input_dir = resolve_project_path(args.input_dir)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    selected = load_selection(input_dir, args.samples_per_group)
    cache_indices = selected["cache_index"].astype(int).tolist()
    print(f"Selected {len(cache_indices)} unique audit samples")

    gen_by_run: dict[str, dict[int, np.ndarray]] = {}
    metrics_by_run: dict[str, dict[int, dict[str, float]]] = {}
    real_by_cache: dict[int, np.ndarray] = {}
    for run_name, checkpoint_rel in RUNS.items():
        print(f"Sampling {run_name}")
        gen, metrics, real = sample_model(
            run_name,
            resolve_project_path(checkpoint_rel),
            data_dir,
            args.cache_prefix,
            cache_indices,
            args,
            device,
        )
        gen_by_run[run_name] = gen
        metrics_by_run[run_name] = metrics
        real_by_cache.update(real)

    rows = []
    for _, row in selected.iterrows():
        cache_index = int(row.cache_index)
        safe_group = str(row.audit_group).replace("/", "_")
        path = output_dir / f"{safe_group}_cache{cache_index}.png"
        plot_sample(
            row,
            real_by_cache[cache_index],
            {run_name: gen_by_run[run_name][cache_index] for run_name in RUNS},
            {run_name: metrics_by_run[run_name][cache_index] for run_name in RUNS},
            path,
            args.envelope_kernel,
        )
        record = row.to_dict()
        record["figure"] = str(path)
        for run_name in RUNS:
            for key, value in metrics_by_run[run_name][cache_index].items():
                record[f"audit_{run_name}_{key}"] = value
        rows.append(record)
        print(f"wrote {path}")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "audit_manifest.csv", index=False)
    summary = {
        "n_samples": int(len(manifest)),
        "samples_per_group": int(args.samples_per_group),
        "num_gen": int(args.num_gen),
        "steps": int(args.steps),
        "runs": list(RUNS.keys()),
        "manifest": str(output_dir / "audit_manifest.csv"),
        "figures": manifest["figure"].tolist(),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(sanitize_json(summary), indent=2))
    print(f"Done. Manifest: {output_dir / 'audit_manifest.csv'}")


if __name__ == "__main__":
    main()
