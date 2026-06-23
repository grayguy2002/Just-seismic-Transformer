"""Export one observed/predicted/residual JsT waveform for Fig. 4a artwork.

This script is intended to run where the JsT checkpoint and dense KiK-net cache
are available, typically lab54.  It reproduces the Exp H inference path for one
station-event record and writes compact source data for the local figure script.

Run from the project root:
  python3 manuscript/figures/export_fig4a_jst_waveform_sample.py \
    --checkpoint outputs/run036/checkpoint-last.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from JsT import SeismicWaveformDataset, load_checkpoint_models  # noqa: E402
from JsT.ablation import AblationConditionEncoder  # noqa: E402


EVENT_ID = "20140314020700"
STATION_ID = "KIKNET.TTRH02"
TOTAL_SAMPLES = 3200
SAMPLE_RATE_HZ = 40.0
N_FREQ_BINS = 40
F_MIN, F_MAX = 0.3, 15.0
FREQ_EDGES = np.logspace(np.log10(F_MIN), np.log10(F_MAX), N_FREQ_BINS + 1)
FREQ_CENTERS = np.sqrt(FREQ_EDGES[:-1] * FREQ_EDGES[1:])


@torch.no_grad()
def generate_waveform(dn, tokens: torch.Tensor, noise: torch.Tensor, steps: int = 50) -> torch.Tensor:
    device = tokens.device
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()
    for i in range(steps):
        t = ts[i]
        t_next = ts[i + 1]
        xp = dn.net(z, t.expand(z.shape[0]), tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
        z = z + (t_next - t) * v
    return z


def compute_hvsr(residual: np.ndarray, predicted: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    freqs = np.fft.rfftfreq(residual.shape[1], d=1.0 / SAMPLE_RATE_HZ)
    per_comp = []
    for ch in range(3):
        sr = np.abs(np.fft.rfft(residual[ch]))
        sp = np.abs(np.fft.rfft(predicted[ch]))
        ratio = []
        for b in range(N_FREQ_BINS):
            mask = (freqs >= FREQ_EDGES[b]) & (freqs < FREQ_EDGES[b + 1])
            if mask.any():
                ratio.append(np.log10(max(float(sr[mask].mean()), eps) / max(float(sp[mask].mean()), eps)))
            else:
                ratio.append(np.nan)
        per_comp.append(ratio)
    return np.nan_to_num(np.nanmean(np.asarray(per_comp), axis=0), nan=0.0, posinf=0.0, neginf=0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/run036/checkpoint-last.pth")
    parser.add_argument("--data-dir", default="data/kiknet_dense_arrival_qc_events_v1")
    parser.add_argument("--train-ref-dir", default="data/seisbench_mlaapde_pwave_v21_36m")
    parser.add_argument("--output-dir", default="outputs/fig4a_jst_waveform_sample")
    parser.add_argument("--drop-tokens", default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    ce, dn, _ = load_checkpoint_models(
        args.checkpoint,
        dev,
        use_ema=True,
        sampling_method="heun",
        steps=50,
        cfg_scale=1.0,
    )
    if dropped:
        ce = AblationConditionEncoder(ce, dropped)
    ce.eval()
    dn.eval()

    ds_train_ref = SeismicWaveformDataset(
        args.train_ref_dir,
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

    conditions = ds.conditions.iloc[ds.indices].copy()
    conditions["station_id"] = (
        conditions["station_network_code"].fillna("KIKNET").astype(str)
        + "."
        + conditions["station_code"].fillna("UNKNOWN").astype(str)
    )
    row = conditions[
        (conditions["station_id"].astype(str) == STATION_ID)
        & (conditions["event_id"].astype(str) == EVENT_ID)
    ]
    if row.empty:
        raise ValueError(f"Missing sample {STATION_ID} {EVENT_ID} in {args.data_dir}")
    row = row.iloc[0]

    cache_idx = int(row["cache_index"])
    cache_to_ds_pos = {int(cache_idx): pos for pos, cache_idx in enumerate(ds.indices)}
    ds_pos = cache_to_ds_pos[cache_idx]
    observed_tensor, cond_dict = ds[ds_pos]
    observed = observed_tensor.numpy().astype(np.float32)

    cond_gpu = {key: value.unsqueeze(0).to(dev) for key, value in cond_dict.items()}
    tokens = ce(cond_gpu)
    torch.manual_seed(42)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(42)
    noise = dn.noise_scale * torch.randn(1, 3, TOTAL_SAMPLES, device=dev)
    predicted = generate_waveform(dn, tokens, noise, steps=50).cpu().numpy()[0].astype(np.float32)
    residual = (observed - predicted).astype(np.float32)
    hvsr = compute_hvsr(residual, predicted).astype(np.float32)
    band_mask = (FREQ_CENTERS >= 1.0) & (FREQ_CENTERS < 10.0)

    np.savez_compressed(
        out_dir / "fig4a_jst_waveform_sample.npz",
        observed=observed,
        predicted=predicted,
        residual=residual,
        hvsr=hvsr,
        frequency_hz=FREQ_CENTERS.astype(np.float32),
        time_s=(np.arange(TOTAL_SAMPLES, dtype=np.float32) / SAMPLE_RATE_HZ - 20.0),
    )

    summary = {
        "station_id": STATION_ID,
        "event_id": EVENT_ID,
        "cache_index": cache_idx,
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "jst_1_10_hz_log10_score": float(np.mean(hvsr[band_mask])),
        "vs30_m_s": float(row["vs30_m_s"]),
        "nehrp": str(row["nehrp_site_class"]),
        "arrival_sample": float(row["selected_phase_arrival_sample"]),
        "source_magnitude": float(row["source_magnitude"]),
        "source_depth_km": float(row["source_depth_km"]),
        "path_ep_distance_km": float(row["path_ep_distance_km"]),
    }
    (out_dir / "fig4a_jst_waveform_sample_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(out_dir / "fig4a_jst_waveform_sample_summary.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
