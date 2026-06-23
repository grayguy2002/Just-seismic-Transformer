#!/usr/bin/env python3
"""Train JsT on the MLAAPDE P-wave cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

# Add parent to path so we can import JsT without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    JsT,
    SeismicConditionEncoder,
    ConditionSpec,
    Denoiser,
    SeismicWaveformDataset,
    collate_conditions,
    AblationConditionEncoder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(args, cond_encoder: SeismicConditionEncoder) -> JsT:
    return JsT(
        n_samples=args.n_samples,
        patch_size=args.patch_size,
        in_channels=3,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        bottleneck_dim=args.bottleneck_dim,
        attn_drop=args.attn_drop,
        proj_drop=args.proj_drop,
        n_cond_tokens=cond_encoder.n_tokens,
        cond_token_groups=cond_encoder.group_token_indices,
    )


def encode_conditions(
    cond_encoder: SeismicConditionEncoder,
    cond: dict[str, torch.Tensor],
) -> torch.Tensor:
    return cond_encoder(cond)


SAFE_WEIGHT_COLUMNS = [
    "selected_phase",
    "mag_bin",
    "depth_bin",
    "distance_bin",
    "trace_channel",
    "station_network_code",
]


def add_monitoring_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mag_bin"] = pd.cut(
        pd.to_numeric(out["source_magnitude"], errors="coerce"),
        [-np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, np.inf],
        labels=["<=3", "3-4", "4-5", "5-6", "6-7", ">7"],
    ).astype(str)
    out["depth_bin"] = pd.cut(
        pd.to_numeric(out["source_depth_km"], errors="coerce"),
        [-np.inf, 10.0, 35.0, 70.0, 150.0, 300.0, np.inf],
        labels=["<=10", "10-35", "35-70", "70-150", "150-300", ">300"],
    ).astype(str)
    out["distance_bin"] = pd.cut(
        pd.to_numeric(out["path_ep_distance_deg"], errors="coerce"),
        [-np.inf, 1.0, 3.0, 10.0, 30.0, 60.0, np.inf],
        labels=["<=1", "1-3", "3-10", "10-30", "30-60", ">60"],
    ).astype(str)
    for col in SAFE_WEIGHT_COLUMNS:
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


def build_train_sampler(
    train_dataset: SeismicWaveformDataset,
    weight_csv: str,
) -> tuple[WeightedRandomSampler | None, dict[str, float | int | str]]:
    if not weight_csv:
        return None, {}
    weight_path = Path(weight_csv)
    weights = pd.read_csv(weight_path)
    if "eligible" in weights.columns:
        eligible = weights["eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
        weights = weights[eligible].copy()
    meta = add_monitoring_bins(train_dataset.conditions.iloc[train_dataset.indices].copy())
    sample_weights = np.ones(len(meta), dtype=np.float64)
    matched_rows = 0
    for _, row in weights.iterrows():
        try:
            multiplier = float(row["recommended_sampling_multiplier"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(multiplier) or multiplier <= 1.0:
            continue
        cols = [c.strip() for c in str(row.get("bin_columns", "")).split(",") if c.strip()]
        cols = [c for c in cols if c in SAFE_WEIGHT_COLUMNS]
        if not cols:
            continue
        mask = np.ones(len(meta), dtype=bool)
        valid = True
        for col in cols:
            value = row.get(col)
            if pd.isna(value) or str(value) == "":
                valid = False
                break
            mask &= meta[col].to_numpy(str) == str(value)
        if not valid or not mask.any():
            continue
        matched_rows += 1
        sample_weights[mask] = np.maximum(sample_weights[mask], multiplier)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    report = {
        "weight_csv": str(weight_path),
        "eligible_rows": int(len(weights)),
        "matched_rows": int(matched_rows),
        "weighted_samples": int((sample_weights > 1.0).sum()),
        "max_weight": float(sample_weights.max()),
        "mean_weight": float(sample_weights.mean()),
    }
    return sampler, report


def parse_named_csvs(specs: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs or []:
        if "=" in spec:
            name, path = spec.split("=", 1)
            name = name.strip()
        else:
            path = spec
            name = Path(path).stem
        if not name:
            raise ValueError(f"Missing hard validation name in {spec}")
        safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
        out[safe_name] = Path(path)
    return out


def build_hard_val_loaders(
    specs: list[str] | None,
    data_dir: Path,
    train_dataset: SeismicWaveformDataset,
    args: argparse.Namespace,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, csv_path in parse_named_csvs(specs).items():
        table = pd.read_csv(csv_path)
        if "cache_index" not in table.columns:
            raise ValueError(f"Hard validation CSV lacks cache_index: {csv_path}")
        requested = pd.to_numeric(table["cache_index"], errors="coerce").dropna().astype(np.int64).tolist()
        requested = list(dict.fromkeys(requested))
        dataset = SeismicWaveformDataset(
            data_dir,
            split="testing",
            augment=False,
            vocab_from=train_dataset,
            cache_prefix=args.cache_prefix,
            condition_version=args.condition_schema,
            field_policy=args.field_policy,
        )
        test_ids = set(int(x) for x in dataset.indices)
        missing = [int(x) for x in requested if int(x) not in test_ids]
        if missing:
            raise ValueError(f"Hard validation CSV contains non-testing cache_index values: {missing[:20]}")
        dataset.indices = np.asarray(requested, dtype=np.int64)
        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_conditions,
        )
    return loaders


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    denoiser: Denoiser,
    cond_encoder: SeismicConditionEncoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_writer: SummaryWriter | None,
    encoder_weight: float = 0.0,
    log_freq: int = 20,
    grad_clip: float = 1.0,
) -> float:
    denoiser.train()
    cond_encoder.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for step, (x, cond) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        cond = {k: v.to(device, non_blocking=True) for k, v in cond.items()}

        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            cond_tokens = encode_conditions(cond_encoder, cond)
            result = denoiser(
                x,
                cond_tokens,
                encoder_weight=encoder_weight,
            )
            if isinstance(result, tuple):
                v_loss, enc_loss = result
                loss = v_loss + enc_loss * encoder_weight
            else:
                loss = result
                v_loss = loss
                enc_loss = torch.tensor(0.0, device=x.device)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(denoiser.parameters(), grad_clip)
            nn.utils.clip_grad_norm_(cond_encoder.parameters(), grad_clip)
        optimizer.step()

        denoiser.update_ema()

        loss_val = loss.item()
        v_loss_val = v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss
        total_loss += loss_val

        if log_writer is not None and step % log_freq == 0:
            global_step = epoch * n_batches + step
            log_writer.add_scalar("train/v_loss", v_loss_val, global_step)
            e_val = enc_loss.item() if isinstance(enc_loss, torch.Tensor) else enc_loss
            log_writer.add_scalar("train/enc_loss", e_val, global_step)
            log_writer.add_scalar("train/total_loss", loss_val, global_step)
            log_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

        if step % log_freq == 0:
            e_str = f" enc={enc_loss.item():.4f}" if encoder_weight > 0 else ""
            print(f"  epoch {epoch:3d}  [{step:4d}/{n_batches:4d}]  loss={loss_val:.6f}{e_str}")

    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Validation (reconstruction loss on a fixed batch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    denoiser: Denoiser,
    cond_encoder: SeismicConditionEncoder,
    dataloader: DataLoader,
    device: torch.device,
    n_batches: int = 3,
) -> float:
    denoiser.eval()
    cond_encoder.eval()
    total = 0.0
    count = 0
    for x, cond in dataloader:
        if count >= n_batches:
            break
        x = x.to(device, non_blocking=True)
        cond = {k: v.to(device, non_blocking=True) for k, v in cond.items()}
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            cond_tokens = encode_conditions(cond_encoder, cond)
            result = denoiser(x, cond_tokens)
            if isinstance(result, tuple):
                loss = result[0]
            else:
                loss = result
        total += loss.item()
        count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Generate samples for visual inspection
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    denoiser: Denoiser,
    cond_encoder: SeismicConditionEncoder,
    val_dataset: SeismicWaveformDataset,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    n_samples: int = 4,
    steps: int = 50,
):
    denoiser.eval()
    cond_encoder.eval()

    # Use EMA-1 weights for generation
    sd_backup = denoiser.state_dict()
    try:
        ema_sd = denoiser.apply_ema(1)
        denoiser.load_state_dict(ema_sd)
    except RuntimeError:
        pass  # EMA not available, use raw weights

    indices = np.random.choice(len(val_dataset), size=n_samples, replace=False)
    waveforms, conds = [], []
    for i in indices:
        w, c = val_dataset[i]
        waveforms.append(w)
        conds.append(c)
    real = torch.stack(waveforms).to(device)
    cond_batch = collate_conditions(list(zip(waveforms, conds)))[1]
    cond_batch = {k: v.to(device) for k, v in cond_batch.items()}
    cond_tokens = encode_conditions(cond_encoder, cond_batch)

    gen = denoiser.generate(cond_tokens, steps=steps)

    # Save as numpy for external plotting
    np.savez(
        out_dir / f"samples_epoch{epoch:04d}.npz",
        real=real.cpu().numpy(),
        gen=gen.cpu().numpy(),
    )

    # Restore original weights
    denoiser.load_state_dict(sd_backup)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train JsT")
    # Model
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=3200)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--attn-drop", type=float, default=0.0)
    parser.add_argument("--proj-drop", type=float, default=0.0)
    # Condition encoder
    parser.add_argument("--condition-version", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--condition-transformer-layers", type=int, default=0)
    parser.add_argument("--condition-transformer-heads", type=int, default=4)
    parser.add_argument("--condition-transformer-dropout", type=float, default=0.0)
    # Training
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--cond-drop-start-epoch", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-batches", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    # Diffusion
    parser.add_argument("--P-mean", type=float, default=-0.8)
    parser.add_argument("--P-std", type=float, default=0.8)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--t-eps", type=float, default=5e-2)
    parser.add_argument("--cond-drop-prob", type=float, default=0.40)
    parser.add_argument("--z-amp-jitter", type=float, default=0.0)
    parser.add_argument("--encoder-weight", type=float, default=0.0)
    parser.add_argument("--ema-decay1", type=float, default=0.9999)
    parser.add_argument("--ema-decay2", type=float, default=0.9996)
    # Sampling
    parser.add_argument("--sampling-method", default="heun", choices=["euler", "heun"])
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    # Data
    parser.add_argument("--data-dir", default="data/seisbench_mlaapde_pwave_v1")
    parser.add_argument("--cache-prefix", default="pwave_v1")
    parser.add_argument("--condition-schema", choices=["legacy", "v2.1"], default="v2.1")
    parser.add_argument("--field-policy", default="default")
    parser.add_argument("--drop-tokens", type=str, default="",
                        help="Comma-separated token indices to zero out (ablation)")
    parser.add_argument("--train-weight-csv", default="")
    parser.add_argument(
        "--hard-val-csv",
        action="append",
        default=None,
        help="Optional name=csv hard validation subset with testing cache_index values.",
    )
    parser.add_argument("--hard-val-batches", type=int, default=0)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    # Logging
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=200)
    parser.add_argument("--sample-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    # Resume
    parser.add_argument("--resume", default="")
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument("--resume-reset-ema", action="store_true")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_writer = SummaryWriter(log_dir=str(output_dir / "logs")) if SummaryWriter is not None else None
    if log_writer is None:
        print("TensorBoard is not installed; continuing without SummaryWriter logs.")

    # Save args
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    if args.condition_version == "v3" and args.condition_schema != "v2.1":
        raise ValueError("--condition-version v3 requires --condition-schema v2.1")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset = SeismicWaveformDataset(
        data_dir,
        split="training",
        augment=args.augment,
        cache_prefix=args.cache_prefix,
        condition_version=args.condition_schema,
        field_policy=args.field_policy,
    )
    val_dataset = SeismicWaveformDataset(
        data_dir,
        split="validation",
        augment=False,
        vocab_from=train_dataset,
        cache_prefix=args.cache_prefix,
        condition_version=args.condition_schema,
        field_policy=args.field_policy,
    )

    train_sampler, train_sampler_report = build_train_sampler(train_dataset, args.train_weight_csv)
    if train_sampler_report:
        print(f"Train sampler weights: {train_sampler_report}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_conditions,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_conditions,
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples:   {len(val_dataset):,}")
    hard_val_loaders = build_hard_val_loaders(args.hard_val_csv, data_dir, train_dataset, args)
    if hard_val_loaders:
        print("Hard validation subsets:")
        for name, loader in hard_val_loaders.items():
            print(f"  {name}: {len(loader.dataset):,} samples")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cond_spec = ConditionSpec(
        **train_dataset.condition_spec,
        hidden_dim=args.hidden_size,
        encoder_version=args.condition_version,
        use_condition_transformer=args.condition_transformer_layers > 0,
        condition_transformer_layers=args.condition_transformer_layers,
        condition_transformer_heads=args.condition_transformer_heads,
        condition_transformer_dropout=args.condition_transformer_dropout,
    )
    cond_encoder = SeismicConditionEncoder(cond_spec)
    n_ce = sum(p.numel() for p in cond_encoder.parameters() if p.requires_grad) / 1e6
    print(f"ConditionEncoder parameters: {n_ce:.2f}M")
    print(f"Condition version: {cond_encoder.encoder_version}")
    print(f"Condition tokens ({cond_encoder.n_tokens}): {cond_encoder.token_names}")
    print(f"Condition groups: {cond_encoder.group_token_indices}")

    # Token ablation: zero out specified token positions
    dropped_str = getattr(args, "drop_tokens", "") or ""
    _ablated_base_encoder = None
    if dropped_str:
        dropped = [int(x.strip()) for x in dropped_str.split(",") if x.strip()]
        _ablated_base_encoder = cond_encoder  # keep raw encoder for checkpoint saving
        cond_encoder = AblationConditionEncoder(cond_encoder, dropped)
        print(f"Ablation: dropped tokens {dropped}")
        print(f"Remaining groups: {cond_encoder.group_token_indices}")

    jst = make_model(args, cond_encoder)
    n_params = sum(p.numel() for p in jst.parameters() if p.requires_grad) / 1e6
    print(f"JsT parameters: {n_params:.2f}M")

    denoiser = Denoiser(
        jst,
        P_mean=args.P_mean,
        P_std=args.P_std,
        noise_scale=args.noise_scale,
        t_eps=args.t_eps,
        cond_drop_prob=args.cond_drop_prob,
        ema_decay1=args.ema_decay1,
        ema_decay2=args.ema_decay2,
        sampling_method=args.sampling_method,
        num_sampling_steps=args.sampling_steps,
        cfg_scale=args.cfg_scale,
        z_amp_jitter=args.z_amp_jitter,
    )
    denoiser.init_ema()
    denoiser.to(device)
    cond_encoder.to(device)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    all_params = list(denoiser.parameters()) + list(cond_encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --------------------------------------------------------------
    # Helper to build a checkpoint dict
    # --------------------------------------------------------------
    def _make_checkpoint(
        denoiser,
        cond_encoder,
        optimizer,
        scheduler,
        epoch,
        val_loss: float | None = None,
        best_val_loss: float | None = None,
        best_epoch: int | None = None,
        epochs_since_improvement: int = 0,
    ):
        ema1_dict = {}
        ema2_dict = {}
        if denoiser.ema_params1 is not None:
            for i, (name, _) in enumerate(denoiser.named_parameters()):
                ema1_dict[name] = denoiser.ema_params1[i].cpu()
                ema2_dict[name] = denoiser.ema_params2[i].cpu()
        # Save raw encoder state (not ablation wrapper) for portability
        ce_to_save = getattr(cond_encoder, "encoder", cond_encoder)
        return {
            "denoiser": denoiser.state_dict(),
            "cond_encoder": ce_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_since_improvement": epochs_since_improvement,
            "ema_params1": ema1_dict,
            "ema_params2": ema2_dict,
            "vocab": {
                "magtype": cond_spec.magnitude_types,
                "phase": cond_spec.phases,
                "channel": cond_spec.channels,
                "network": cond_spec.network_codes,
                "station_id": cond_spec.station_ids,
                "station_location": cond_spec.station_locations,
                "source_magnitude_author": cond_spec.source_magnitude_authors,
                "phase_status": cond_spec.phase_statuses,
            },
            "arch": {
                "format_version": 3,
                "jst": {
                    "n_samples": args.n_samples,
                    "patch_size": args.patch_size,
                    "in_channels": 3,
                    "hidden_size": args.hidden_size,
                    "depth": args.depth,
                    "num_heads": args.num_heads,
                    "mlp_ratio": args.mlp_ratio,
                    "bottleneck_dim": args.bottleneck_dim,
                    "attn_drop": args.attn_drop,
                    "proj_drop": args.proj_drop,
                    "n_cond_tokens": cond_encoder.n_tokens,
                    "cond_token_groups": cond_encoder.group_token_indices,
                    "film_groups": denoiser.net.film_groups,
                },
                "condition_encoder": cond_spec.to_config(),
                "denoiser": {
                    "P_mean": args.P_mean,
                    "P_std": args.P_std,
                    "noise_scale": args.noise_scale,
                    "t_eps": args.t_eps,
                    "cond_drop_prob": args.cond_drop_prob,
                    "ema_decay1": args.ema_decay1,
                    "ema_decay2": args.ema_decay2,
                    "sampling_method": args.sampling_method,
                    "num_sampling_steps": args.sampling_steps,
                    "cfg_scale": args.cfg_scale,
                    "z_amp_jitter": args.z_amp_jitter,
                },
            },
        }

    def _save_checkpoint(ckpt, output_dir, name):
        path = output_dir / name
        torch.save(ckpt, path)
        print(f"  Saved {path}")

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_since_improvement = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        denoiser.load_state_dict(ckpt["denoiser"])
        # Load raw encoder state dict (strip ablation wrapper prefix if present)
        ce_sd = {}
        for k, v in ckpt["cond_encoder"].items():
            ce_sd[k[8:] if k.startswith("encoder.") else k] = v
        cond_encoder.load_state_dict(ce_sd)
        if not args.resume_reset_optimizer:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = float(ckpt.get("best_val_loss") or float("inf"))
            best_epoch = int(ckpt.get("best_epoch", -1))
            epochs_since_improvement = int(ckpt.get("epochs_since_improvement", 0))
        if not args.resume_reset_ema and "ema_params1" in ckpt and ckpt["ema_params1"]:
            for i, (name, _) in enumerate(denoiser.named_parameters()):
                denoiser.ema_params1[i] = ckpt["ema_params1"][name].to(device)
                denoiser.ema_params2[i] = ckpt["ema_params2"][name].to(device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.cond_drop_prob > 0 and start_epoch >= args.cond_drop_start_epoch:
        denoiser.set_group_drop_probs(
            source=0.0,
            path=args.cond_drop_prob * 1.7,
            receiver=0.0,
        )
        denoiser.set_cond_drop_prob(args.cond_drop_prob)
        print(f"Initial group_drop src=0.00 path={args.cond_drop_prob*1.7:.2f} rcv=0.00")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"Training for {args.epochs} epochs ({len(train_loader)} batches/epoch)")
    if args.cond_drop_start_epoch > 0 and start_epoch < args.cond_drop_start_epoch:
        denoiser.set_cond_drop_prob(0.0)
        print(f"Phase 1 (epochs 0-{args.cond_drop_start_epoch-1}): NO condition dropout")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Phase 2: enable condition dropout with per-group probabilities
        if args.cond_drop_prob > 0 and epoch == args.cond_drop_start_epoch:
            denoiser.set_group_drop_probs(
                source=0.0,                               # source NEVER dropped
                path=args.cond_drop_prob * 1.7,           # 0.68 — path deprivation
                receiver=0.0,                             # receiver NEVER dropped
            )
            denoiser.set_cond_drop_prob(args.cond_drop_prob)
            print(f"Phase 2 (epochs {epoch}+): group_drop src=0.00 "
                  f"path={args.cond_drop_prob*1.7:.2f} rcv=0.00")

        avg_loss = train_one_epoch(
            denoiser, cond_encoder, train_loader, optimizer, device,
            epoch, log_writer, encoder_weight=args.encoder_weight,
            log_freq=args.log_freq, grad_clip=args.grad_clip,
        )
        scheduler.step()

        val_loss = validate(denoiser, cond_encoder, val_loader, device, n_batches=args.val_batches)
        hard_val_results: dict[str, dict[str, float]] = {}
        hard_val_batches = args.hard_val_batches or args.val_batches
        for name, loader in hard_val_loaders.items():
            hard_loss = validate(denoiser, cond_encoder, loader, device, n_batches=hard_val_batches)
            hard_val_results[name] = {"loss": hard_loss}
        if hard_val_results:
            for name, metrics in hard_val_results.items():
                print(f"  hard_val/{name} loss={metrics['loss']:.6f}")
        if hard_val_results and log_writer is not None:
            for name, metrics in hard_val_results.items():
                for key, value in metrics.items():
                    log_writer.add_scalar(f"hard_val/{name}/{key}", value, epoch)
        print(f"--- epoch {epoch:3d}  train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e} ---")

        if log_writer is not None:
            log_writer.add_scalar("val/loss", val_loss, epoch)

        improved = val_loss < best_val_loss - args.early_stopping_min_delta
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_improvement = 0
            ckpt = _make_checkpoint(
                denoiser,
                cond_encoder,
                optimizer,
                scheduler,
                epoch,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_since_improvement=epochs_since_improvement,
            )
            _save_checkpoint(ckpt, output_dir, "checkpoint-best.pth")
            print(f"  New best val_loss={best_val_loss:.6f} at epoch {best_epoch}")
        else:
            epochs_since_improvement += 1

        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            ckpt = _make_checkpoint(
                denoiser,
                cond_encoder,
                optimizer,
                scheduler,
                epoch,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_since_improvement=epochs_since_improvement,
            )
            _save_checkpoint(ckpt, output_dir, "checkpoint-last.pth")
            print(
                f"Early stopping at epoch {epoch}: best_val_loss={best_val_loss:.6f} "
                f"at epoch {best_epoch}, no improvement for {epochs_since_improvement} epochs."
            )
            break

        # Generate samples
        if (epoch % args.sample_freq == 0) or (epoch == args.epochs - 1):
            sample_dir = output_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            generate_samples(denoiser, cond_encoder, val_dataset, device, sample_dir, epoch)

        # Checkpoint
        if (epoch % args.save_freq == 0) or (epoch == args.epochs - 1):
            ckpt = _make_checkpoint(
                denoiser,
                cond_encoder,
                optimizer,
                scheduler,
                epoch,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                epochs_since_improvement=epochs_since_improvement,
            )
            _save_checkpoint(ckpt, output_dir, f"checkpoint-{epoch:04d}.pth")
            _save_checkpoint(ckpt, output_dir, "checkpoint-last.pth")

    elapsed = time.time() - t0
    print(f"Training complete.  Elapsed: {elapsed/3600:.1f}h")
    if log_writer is not None:
        log_writer.close()


if __name__ == "__main__":
    main()
