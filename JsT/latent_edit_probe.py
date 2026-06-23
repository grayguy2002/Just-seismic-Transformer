#!/usr/bin/env python3
"""Probe JsT condition-token latent directions and direct token editing."""

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
from torch.utils.data import DataLoader

from JsT import Denoiser, SeismicConditionEncoder, load_checkpoint_models
from JsT.dataset import SeismicWaveformDataset, collate_conditions

RUN_NAME = "run019"
TIME = np.arange(-20, 60, 1 / 40)
EPS = 1e-8


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else project_root() / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JsT condition-token latent editing probe")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v1")
    parser.add_argument("--cache-prefix", default="pwave_v1")
    parser.add_argument("--condition-schema", choices=["auto", "legacy", "v2.1", "v3"], default="auto")
    parser.add_argument("--checkpoint", default=f"outputs/{RUN_NAME}/checkpoint-last.pth")
    parser.add_argument("--anomaly-scores", default=f"outputs/anomaly_exp1_{RUN_NAME}_full/scores_testing_{RUN_NAME}.csv")
    parser.add_argument("--output-dir", default=f"outputs/latent_edit_{RUN_NAME}")
    parser.add_argument("--split", default="testing")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alphas", default="-0.75,-0.375,0,0.375,0.75")
    return parser.parse_args()


def select_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: Path, device: torch.device, steps: int, cfg_scale: float) -> tuple[SeismicConditionEncoder, Denoiser, dict[str, Any]]:
    return load_checkpoint_models(
        checkpoint,
        device,
        use_ema=True,
        steps=steps,
        cfg_scale=cfg_scale,
    )


def checkpoint_condition_schema(ckpt: dict[str, Any]) -> str:
    cfg = ckpt.get("arch", {}).get("condition_encoder", {})
    ev = cfg.get("encoder_version", "v1")
    if ev in ("v2.1", "v3"):
        return "v2.1"
    return "legacy"


def build_eval_dataset(
    data_dir: Path,
    split: str,
    cache_prefix: str,
    condition_schema: str,
) -> SeismicWaveformDataset:
    train = SeismicWaveformDataset(
        data_dir,
        split="training",
        augment=False,
        cache_prefix=cache_prefix,
        condition_version=condition_schema,
    )
    return SeismicWaveformDataset(
        data_dir,
        split=split,
        augment=False,
        vocab_from=train,
        cache_prefix=cache_prefix,
        condition_version=condition_schema,
    )


def move_cond(cond: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in cond.items()}

def extract_tokens(ce: SeismicConditionEncoder, ds: SeismicWaveformDataset, args: argparse.Namespace, device: torch.device) -> tuple[np.ndarray, pd.DataFrame]:
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_conditions,
    )
    tokens = []
    rows = []
    offset = 0
    with torch.no_grad():
        for _, cond_cpu in loader:
            cond = move_cond(cond_cpu, device)
            tok = ce(cond).detach().cpu().numpy()
            tokens.append(tok)
            bsz = tok.shape[0]
            for j in range(bsz):
                split_index = offset + j
                cache_index = int(ds.indices[split_index])
                row = ds.conditions.iloc[cache_index]
                rows.append({
                    "split_index": split_index,
                    "cache_index": cache_index,
                    "event_id": _safe(row, "event_id"),
                    "station_network_code": _safe(row, "station_network_code"),
                    "station_code": _safe(row, "station_code"),
                    "selected_phase": _safe(row, "selected_phase"),
                    "source_magnitude": float(row["source_magnitude"]),
                    "source_depth_km": float(row["source_depth_km"]),
                    "path_ep_distance_deg": float(row["path_ep_distance_deg"]),
                    "path_azimuth_deg": float(row["path_azimuth_deg"]),
                    "path_back_azimuth_deg": float(row["path_back_azimuth_deg"]),
                    "residual_travel_sec": float(row["residual_travel_sec"]),
                    "normalization_scale": float(row["normalization_scale"]),
                })
            offset += bsz
            print(f"  extracted tokens {offset}/{len(ds)}", flush=True)
    return np.concatenate(tokens, axis=0), pd.DataFrame(rows)


def _safe(row: pd.Series, key: str) -> Any:
    if key not in row.index or pd.isna(row[key]):
        return None
    v = row[key]
    return v.item() if isinstance(v, np.generic) else v


def standardize(v: np.ndarray) -> np.ndarray:
    return (v - np.nanmean(v)) / (np.nanstd(v) + EPS)


def direction_by_quantile(
    tokens: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    group: str,
    group_token_indices: dict[str, list[int]],
    q: float = 0.15,
) -> dict[str, Any]:
    if group == "all":
        x = tokens.reshape(tokens.shape[0], -1)
        n_group_tokens = tokens.shape[1]
    else:
        idx = group_token_indices[group]
        x = tokens[:, idx, :].reshape(tokens.shape[0], -1)
        n_group_tokens = len(idx)
    y = meta[target].to_numpy(float)
    lo = y <= np.nanquantile(y, q)
    hi = y >= np.nanquantile(y, 1 - q)
    d = x[hi].mean(axis=0) - x[lo].mean(axis=0)
    proj = (x - x.mean(axis=0)) @ (d / (np.linalg.norm(d) + EPS))
    corr = float(np.corrcoef(proj, y)[0, 1]) if np.std(proj) > 0 and np.std(y) > 0 else float("nan")
    return {
        "target": target,
        "group": group,
        "direction": d.astype(np.float32),
        "n_group_tokens": int(n_group_tokens),
        "direction_norm": float(np.linalg.norm(d)),
        "low_n": int(lo.sum()),
        "high_n": int(hi.sum()),
        "low_mean_target": float(np.nanmean(y[lo])),
        "high_mean_target": float(np.nanmean(y[hi])),
        "projection_target_corr": corr,
    }


def pca2(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    coords = xc @ vt[:2].T
    explained = (s[:2] ** 2) / np.sum(s ** 2)
    return coords, explained


def waveform_metrics(w: np.ndarray) -> dict[str, float]:
    z = w[0]
    peak = float(np.abs(w).max())
    rms = float(np.sqrt(np.mean(w ** 2)))
    early = w[:, 800:1200]
    late = w[:, 1200:]
    early_rms = float(np.sqrt(np.mean(early ** 2)))
    late_rms = float(np.sqrt(np.mean(late ** 2)))
    spec = np.abs(np.fft.rfft(w, axis=-1)).mean(axis=0)
    freqs = np.fft.rfftfreq(w.shape[-1], d=1 / 40)
    centroid = float((freqs * spec).sum() / (spec.sum() + EPS))
    return {
        "peak_abs": peak,
        "rms": rms,
        "early_rms": early_rms,
        "late_rms": late_rms,
        "late_early_rms_ratio": late_rms / (early_rms + EPS),
        "spectral_centroid_hz": centroid,
        "z_peak_abs": float(np.abs(z).max()),
    }


def generate_one(
    dn: Denoiser,
    cond_tokens: torch.Tensor,
    seed: int,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        return dn.generate(cond_tokens, steps=steps)[0].detach().cpu().numpy()


def edit_tokens(
    base: np.ndarray,
    direction: dict[str, Any],
    alpha: float,
    group_token_indices: dict[str, list[int]],
) -> np.ndarray:
    out = base.copy()
    d = direction["direction"]
    group = direction["group"]
    if group == "all":
        out = out + alpha * d.reshape(out.shape)
    else:
        idx = group_token_indices[group]
        out[idx, :] = out[idx, :] + alpha * d.reshape(len(idx), out.shape[-1])
    return out


def select_anchors(meta: pd.DataFrame, anomaly_path: Path | None) -> pd.DataFrame:
    meta = meta.copy()
    if "combined_z" not in meta.columns:
        if anomaly_path is not None and anomaly_path.exists():
            scores = pd.read_csv(anomaly_path)
            cols = ["split_index", "combined_z", "time_rel_l2_min", "envelope_rel_l2_min", "spectral_rel_l2_min", "peak_log_ratio_abs"]
            meta = meta.merge(scores[[c for c in cols if c in scores.columns]], on="split_index", how="left")
        else:
            meta["combined_z"] = np.nan

    if meta["combined_z"].notna().any():
        candidates = [
            ("low_anomaly", meta["combined_z"].idxmin()),
            ("median_anomaly", (meta["combined_z"] - meta["combined_z"].median()).abs().idxmin()),
            ("high_anomaly", meta["combined_z"].idxmax()),
        ]
    else:
        candidates = [
            ("low_distance", meta["path_ep_distance_deg"].idxmin()),
            ("median_distance", (meta["path_ep_distance_deg"] - meta["path_ep_distance_deg"].median()).abs().idxmin()),
            ("high_distance", meta["path_ep_distance_deg"].idxmax()),
        ]
    anchors = []
    for label, idx in candidates:
        rec = meta.loc[idx].to_dict()
        rec["anchor_label"] = label
        anchors.append(rec)
    return pd.DataFrame(anchors)


def plot_pca(coords: np.ndarray, meta: pd.DataFrame, output_dir: Path) -> None:
    specs = [
        ("source_magnitude", "Magnitude"),
        ("path_ep_distance_deg", "Distance"),
        ("source_depth_km", "Depth"),
        ("combined_z", "Anomaly z"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (col, title) in zip(axes.ravel(), specs):
        if col in meta and meta[col].notna().any():
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=meta[col], s=5, cmap="viridis", alpha=0.65)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.65)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("JsT run019 condition-token latent space PCA", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_condition_token_pca.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_edits(edit_df: pd.DataFrame, waveforms: dict[tuple[str, str, float], np.ndarray], output_dir: Path) -> None:
    directions = list(edit_df["direction"].drop_duplicates())
    anchors = list(edit_df["anchor_label"].drop_duplicates())
    alphas = sorted(edit_df["alpha"].drop_duplicates())
    fig, axes = plt.subplots(len(directions), len(alphas), figsize=(3.2 * len(alphas), 2.3 * len(directions)), sharex=True)
    if len(directions) == 1:
        axes = axes[None, :]
    for i, direction in enumerate(directions):
        anchor = anchors[min(i, len(anchors) - 1)]
        for j, alpha in enumerate(alphas):
            ax = axes[i, j]
            w = waveforms[(anchor, direction, alpha)]
            ax.plot(TIME, w[0], lw=0.55, color="#1b4f72")
            ax.axvline(0, color="#d62728", ls="--", lw=0.5, alpha=0.45)
            ax.set_title(f"{direction}\n{anchor}, α={alpha:g}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Direct condition-token latent editing: Z channel", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_latent_token_edits_z.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def slope_and_corr(df: pd.DataFrame, metric: str) -> tuple[float, float]:
    x = df["alpha"].to_numpy(float)
    y = df[metric].to_numpy(float)
    if np.std(x) < EPS or np.std(y) < EPS:
        return 0.0, float("nan")
    slope = float(np.polyfit(x, y, 1)[0])
    corr = float(np.corrcoef(x, y)[0, 1])
    return slope, corr


def main() -> None:
    args = parse_args()
    data_dir = resolve_path(args.data_dir)
    checkpoint = resolve_path(args.checkpoint)
    anomaly_scores = resolve_path(args.anomaly_scores)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"Project root: {project_root()}")
    print(f"Checkpoint:   {checkpoint}")
    print(f"Output dir:   {output_dir}")
    print(f"Device:       {device}")

    ce, dn, ckpt = load_model(checkpoint, device, args.steps, args.cfg_scale)
    group_token_indices = {k: list(v) for k, v in ce.group_token_indices.items()}
    condition_schema = checkpoint_condition_schema(ckpt) if args.condition_schema == "auto" else args.condition_schema
    ds = build_eval_dataset(data_dir, args.split, args.cache_prefix, condition_schema)
    tokens, meta = extract_tokens(ce, ds, args, device)
    if anomaly_scores.exists():
        score_cols = ["split_index", "combined_z"]
        scores = pd.read_csv(anomaly_scores)
        meta = meta.merge(scores[score_cols], on="split_index", how="left")

    concat = tokens.reshape(tokens.shape[0], -1)
    coords, explained = pca2(concat)
    plot_pca(coords, meta, output_dir)

    directions = [
        direction_by_quantile(tokens, meta, "source_magnitude", "source", group_token_indices),
        direction_by_quantile(tokens, meta, "source_depth_km", "source", group_token_indices),
        direction_by_quantile(tokens, meta, "path_ep_distance_deg", "path", group_token_indices),
    ]
    if "combined_z" in meta and meta["combined_z"].notna().any():
        directions.extend([
            direction_by_quantile(tokens, meta, "combined_z", "source", group_token_indices),
            direction_by_quantile(tokens, meta, "combined_z", "path", group_token_indices),
            direction_by_quantile(tokens, meta, "combined_z", "receiver", group_token_indices),
            direction_by_quantile(tokens, meta, "combined_z", "all", group_token_indices),
        ])

    dir_rows = []
    for d in directions:
        row = {k: v for k, v in d.items() if k != "direction"}
        dir_rows.append(row)
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(output_dir / "latent_direction_summary.csv", index=False)

    anchors = select_anchors(meta, anomaly_scores)
    anchors.to_csv(output_dir / "anchors.csv", index=False)

    edit_rows = []
    waveforms: dict[tuple[str, str, float], np.ndarray] = {}
    for anchor_i, anchor in anchors.iterrows():
        split_index = int(anchor["split_index"])
        base_tokens = tokens[split_index]
        anchor_label = str(anchor["anchor_label"])
        for dir_i, direction in enumerate(directions):
            direction_name = f"{direction['group']}:{direction['target']}"
            baseline_w = None
            for alpha in alphas:
                edited = edit_tokens(base_tokens, direction, alpha, group_token_indices)
                cond_tok = torch.from_numpy(edited[None].astype(np.float32)).to(device)
                seed = args.seed + anchor_i * 1000 + dir_i
                w = generate_one(dn, cond_tok, seed, args.steps, device)
                if alpha == 0:
                    baseline_w = w
                waveforms[(anchor_label, direction_name, alpha)] = w
                rec = {
                    "anchor_label": anchor_label,
                    "split_index": split_index,
                    "event_id": anchor.get("event_id"),
                    "station_network_code": anchor.get("station_network_code"),
                    "station_code": anchor.get("station_code"),
                    "selected_phase": anchor.get("selected_phase"),
                    "anchor_combined_z": anchor.get("combined_z"),
                    "direction": direction_name,
                    "target": direction["target"],
                    "group": direction["group"],
                    "alpha": alpha,
                    **waveform_metrics(w),
                }
                if baseline_w is not None:
                    rec["rel_l2_from_alpha0"] = float(np.linalg.norm(w - baseline_w) / (np.linalg.norm(baseline_w) + EPS))
                else:
                    rec["rel_l2_from_alpha0"] = np.nan
                edit_rows.append(rec)

    edit_df = pd.DataFrame(edit_rows)
    # Fill baseline distances after all rows are available.
    for (anchor_label, direction_name), sub in edit_df.groupby(["anchor_label", "direction"]):
        base = waveforms[(anchor_label, direction_name, 0.0)]
        idxs = sub.index
        vals = []
        for idx in idxs:
            alpha = float(edit_df.loc[idx, "alpha"])
            w = waveforms[(anchor_label, direction_name, alpha)]
            vals.append(float(np.linalg.norm(w - base) / (np.linalg.norm(base) + EPS)))
        edit_df.loc[idxs, "rel_l2_from_alpha0"] = vals

    edit_df.to_csv(output_dir / "latent_edit_metrics.csv", index=False)
    plot_edits(edit_df, waveforms, output_dir)

    control_rows = []
    for direction_name, sub_dir in edit_df.groupby("direction"):
        row = {"direction": direction_name}
        for metric in ["peak_abs", "rms", "late_early_rms_ratio", "spectral_centroid_hz", "rel_l2_from_alpha0"]:
            slopes, corrs = [], []
            for _, sub_anchor in sub_dir.groupby("anchor_label"):
                slope, corr = slope_and_corr(sub_anchor.sort_values("alpha"), metric)
                slopes.append(slope)
                corrs.append(corr)
            row[f"{metric}_mean_slope"] = float(np.nanmean(slopes))
            row[f"{metric}_mean_corr"] = float(np.nanmean(corrs))
        control_rows.append(row)
    control_df = pd.DataFrame(control_rows)
    control_df.to_csv(output_dir / "latent_edit_control_summary.csv", index=False)

    summary = {
        "run_name": args.run_name,
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "n_samples": int(len(ds)),
        "token_shape": list(tokens.shape),
        "token_names": list(ce.token_names),
        "group_token_indices": group_token_indices,
        "pca_explained_first2": [float(explained[0]), float(explained[1])],
        "alphas": alphas,
        "directions": dir_rows,
        "anchors": anchors.to_dict(orient="records"),
        "outputs": {
            "latent_direction_summary": str(output_dir / "latent_direction_summary.csv"),
            "latent_edit_metrics": str(output_dir / "latent_edit_metrics.csv"),
            "latent_edit_control_summary": str(output_dir / "latent_edit_control_summary.csv"),
            "anchors": str(output_dir / "anchors.csv"),
            "pca_figure": str(output_dir / "fig_condition_token_pca.png"),
            "edit_figure": str(output_dir / "fig_latent_token_edits_z.png"),
        },
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("Done.")
    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
