#!/usr/bin/env python3
"""Verify that norm_scale post-processing produces realistic M2->M7 amplitude scaling."""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, numpy as np, pandas as pd
from JsT import JsT, SeismicConditionEncoder, ConditionSpec, Denoiser
from JsT.dataset import SeismicWaveformDataset, collate_conditions
from torch.utils.data import DataLoader

RUN = "run014"
ckpt = torch.load(f"outputs/{RUN}/checkpoint-last.pth", map_location="cuda")
vocab = ckpt["vocab"]
ds_train = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="training", augment=False)
ds = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="validation", augment=False, vocab_from=ds_train)
spec = ConditionSpec(vocab["magtype"], vocab["phase"], vocab["channel"], vocab["network"], hidden_dim=512)
jst = JsT(3200, 64, 3, 512, 8, 8).cuda()
ce = SeismicConditionEncoder(spec).cuda()
denoiser = Denoiser(jst).cuda()
denoiser.load_state_dict(ckpt["denoiser"]); ce.load_state_dict(ckpt["cond_encoder"])
if ckpt.get("ema_params1"):
    ema_sd = {}
    for name, _ in denoiser.named_parameters():
        ema_sd[name] = ckpt["ema_params1"][name].cuda()
    denoiser.load_state_dict(ema_sd, strict=False)
denoiser.eval(); ce.eval()

# -- Estimate norm_scale vs magnitude from training data --
train_cond = pd.read_csv("data/seisbench_mlaapde_pwave_v1/cache/pwave_v1_conditions.csv")
print("=== norm_scale by magnitude bin (training data) ===")
mag_medians = {}
for lo, hi in [(2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9)]:
    sub = train_cond[(train_cond["source_magnitude"]>=lo) & (train_cond["source_magnitude"]<hi)]
    med = sub["normalization_scale"].median()
    mag_medians[(lo+hi)/2] = med
    print(f"  M[{lo},{hi}): n={len(sub):5,}  norm_scale median={med:10.0f}")

# -- Test: generate with M=2.5 vs M=7.0, apply physical rescaling --
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_conditions)
_, anchor = next(iter(loader))
anchor = {k: v.cuda() for k, v in anchor.items()}

print("\n=== Amplitude-scaled generation ===")
for mag, target_norm in [(2.5, 1000.0), (3.5, 2000.0), (4.5, 5000.0), (5.5, 15000.0), (6.5, 50000.0), (7.5, 200000.0)]:
    c = copy.deepcopy(anchor)
    c["source_magnitude"] = torch.full((4,), mag, device="cuda", dtype=torch.float32)
    c["normalization_scale"] = torch.full((4,), target_norm, device="cuda", dtype=torch.float32)
    ct = ce(c)
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct, steps=50)  # g in [-1,1] range

    # Rescale back to physical counts
    g_physical = g * target_norm  # (B, 3, 3200) in counts
    peak = g_physical.abs().max(dim=2).values.max(dim=1).values  # (B,)
    rms = g_physical.pow(2).mean(dim=(1,2)).sqrt()  # (B,)
    print(f"  M={mag:.1f} norm={target_norm:8.0f}: peak_amp={peak.mean().item():.0f} counts  RMS={rms.mean().item():.0f}")

# -- Sweep: fix norm_scale, sweep magnitude only --
print("\n=== Fixed norm_scale=10000, sweep magnitude ===")
for mag in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
    c = copy.deepcopy(anchor)
    c["source_magnitude"] = torch.full((4,), mag, device="cuda", dtype=torch.float32)
    c["normalization_scale"] = torch.full((4,), 10000.0, device="cuda")
    ct = ce(c)
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct, steps=50)
    g_phys = g * 10000.0
    peak = g_phys.abs().max(dim=2).values.max(dim=1).values.mean().item()
    rms = g_phys.pow(2).mean(dim=(1,2)).sqrt().mean().item()
    std_norm = g.std(dim=(1,2)).mean().item()
    print(f"  M={mag:.1f}: norm_std={std_norm:.4f}  peak={peak:.0f}  RMS={rms:.0f}")
