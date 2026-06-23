"""Checkpoint reconstruction helpers for JsT models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from .condition_encoder import ConditionSpec, SeismicConditionEncoder
from .denoiser import Denoiser
from .model import JsT


def _infer_legacy_jst_config(ckpt: dict[str, Any]) -> dict[str, Any]:
    sd = ckpt["denoiser"]
    cond_pos = sd["net.cond_pos_embed"]
    pos = sd["net.pos_embed"]
    final_w = sd["net.final_layer.linear.weight"]
    proj1_w = sd["net.x_embedder.proj1.weight"]
    hidden_size = int(cond_pos.shape[-1])
    n_cond_tokens = int(cond_pos.shape[1])
    patch_size = int(final_w.shape[0] // 3)
    n_samples = int(pos.shape[1] * patch_size)
    depth = len({int(k.split(".")[2]) for k in sd if k.startswith("net.blocks.")})
    return {
        "n_samples": n_samples,
        "patch_size": patch_size,
        "in_channels": 3,
        "hidden_size": hidden_size,
        "depth": depth,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "bottleneck_dim": int(proj1_w.shape[0]),
        "attn_drop": 0.0,
        "proj_drop": 0.0,
        "n_cond_tokens": n_cond_tokens,
        "cond_token_groups": {"source": [0], "path": [1], "receiver": [2]},
    }


def _legacy_condition_spec(ckpt: dict[str, Any], hidden_dim: int) -> ConditionSpec:
    vocab = ckpt["vocab"]
    return ConditionSpec(
        magnitude_types=vocab["magtype"],
        phases=vocab["phase"],
        channels=vocab["channel"],
        network_codes=vocab["network"],
        hidden_dim=hidden_dim,
        encoder_version="v1",
    )


def _build_from_arch(
    ckpt: dict[str, Any],
    sampling_method: str | None,
    steps: int | None,
    cfg_scale: float | None,
) -> tuple[SeismicConditionEncoder, Denoiser]:
    arch = ckpt["arch"]
    spec = ConditionSpec.from_config(arch["condition_encoder"])
    cond_encoder = SeismicConditionEncoder(spec)

    jst_cfg = dict(arch["jst"])
    jst = JsT(
        n_samples=jst_cfg["n_samples"],
        patch_size=jst_cfg["patch_size"],
        in_channels=jst_cfg.get("in_channels", 3),
        hidden_size=jst_cfg["hidden_size"],
        depth=jst_cfg["depth"],
        num_heads=jst_cfg["num_heads"],
        mlp_ratio=jst_cfg.get("mlp_ratio", 4.0),
        bottleneck_dim=jst_cfg.get("bottleneck_dim", 128),
        attn_drop=jst_cfg.get("attn_drop", 0.0),
        proj_drop=jst_cfg.get("proj_drop", 0.0),
        n_cond_tokens=jst_cfg["n_cond_tokens"],
        cond_token_groups=jst_cfg.get("cond_token_groups"),
        film_groups=jst_cfg.get("film_groups"),
    )

    den_cfg = dict(arch.get("denoiser", {}))
    if sampling_method is not None:
        den_cfg["sampling_method"] = sampling_method
    if steps is not None:
        den_cfg["num_sampling_steps"] = steps
    if cfg_scale is not None:
        den_cfg["cfg_scale"] = cfg_scale
    denoiser = Denoiser(jst, **den_cfg)
    return cond_encoder, denoiser


def _build_legacy(
    ckpt: dict[str, Any],
    sampling_method: str | None,
    steps: int | None,
    cfg_scale: float | None,
) -> tuple[SeismicConditionEncoder, Denoiser]:
    jst_cfg = _infer_legacy_jst_config(ckpt)
    spec = _legacy_condition_spec(ckpt, jst_cfg["hidden_size"])
    cond_encoder = SeismicConditionEncoder(spec)
    jst = JsT(**jst_cfg)
    denoiser = Denoiser(
        jst,
        sampling_method=sampling_method or "heun",
        num_sampling_steps=steps or 50,
        cfg_scale=1.0 if cfg_scale is None else cfg_scale,
    )
    return cond_encoder, denoiser


def _load_ema(denoiser: Denoiser, ckpt: dict[str, Any], ema_which: int, device: torch.device) -> None:
    key = f"ema_params{ema_which}"
    if not ckpt.get(key):
        return
    ema = {}
    for name, _ in denoiser.named_parameters():
        if name in ckpt[key]:
            ema[name] = ckpt[key][name].to(device)
    denoiser.load_state_dict(ema, strict=False)


def load_checkpoint_models(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    use_ema: bool = True,
    ema_which: int = 1,
    sampling_method: Literal["euler", "heun"] | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
) -> tuple[SeismicConditionEncoder, Denoiser, dict[str, Any]]:
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    if "arch" in ckpt:
        cond_encoder, denoiser = _build_from_arch(ckpt, sampling_method, steps, cfg_scale)
    else:
        cond_encoder, denoiser = _build_legacy(ckpt, sampling_method, steps, cfg_scale)

    cond_encoder.to(device)
    denoiser.to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    # Handle ablation-wrapped checkpoints with 'encoder.' prefix
    ce_sd = ckpt["cond_encoder"]
    if any(k.startswith("encoder.") for k in ce_sd):
        ce_sd = {k[8:] if k.startswith("encoder.") else k: v for k, v in ce_sd.items()}
    cond_encoder.load_state_dict(ce_sd)
    if use_ema:
        _load_ema(denoiser, ckpt, ema_which, device)
    denoiser.eval()
    cond_encoder.eval()
    return cond_encoder, denoiser, ckpt
