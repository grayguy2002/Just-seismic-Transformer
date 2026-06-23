#!/usr/bin/env python3
"""Diagnose run003: is collapse ODE or encoder-driven?"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, numpy as np
from JsT import JsT, SeismicConditionEncoder, ConditionSpec, Denoiser
from JsT.dataset import SeismicWaveformDataset, collate_conditions
from torch.utils.data import DataLoader

ckpt = torch.load("outputs/run003/checkpoint-last.pth", map_location="cuda")
vocab = ckpt["vocab"]
ds_train = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="training", augment=False)
ds = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="validation", augment=False, vocab_from=ds_train)
spec = ConditionSpec(vocab["magtype"], vocab["phase"], vocab["channel"], vocab["network"], hidden_dim=512)
jst = JsT(3200, 64, 3, 512, 8, 8).cuda()
ce = SeismicConditionEncoder(spec).cuda()
denoiser = Denoiser(jst).cuda()
denoiser.load_state_dict(ckpt["denoiser"]); ce.load_state_dict(ckpt["cond_encoder"])

# Use EMA
if ckpt.get("ema_params1"):
    ema_sd = {}
    for name, _ in denoiser.named_parameters():
        ema_sd[name] = ckpt["ema_params1"][name].cuda()
    denoiser.load_state_dict(ema_sd, strict=False)
denoiser.eval(); ce.eval()

loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_conditions)
x, cond = next(iter(loader))
x = x.cuda(); cond = {k: v.cuda() for k, v in cond.items()}
ct = ce(cond)

# S1 (collapsed): 5 trials with different noise
print("=== S1: 5 runs, different noise ===")
for trial in range(5):
    torch.manual_seed(42 + trial)
    with torch.no_grad():
        g = denoiser.generate(ct[1:2], steps=50)
    s = g[0].std().item()
    print(f"  trial {trial}: std={s:.4f}  range=[{g[0].min().item():.2f},{g[0].max().item():.2f}]")

# S1: different step counts
print("\n=== S1: varying ODE steps ===")
for steps in [10, 25, 50, 75, 100, 150]:
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct[1:2], steps=steps)
    print(f"  steps={steps:3d}: std={g[0].std().item():.4f}  mean={g[0].mean().item():.4f}")

# S0 (stable): 5 trials
print("\n=== S0: 5 runs, different noise ===")
for trial in range(5):
    torch.manual_seed(42 + trial)
    with torch.no_grad():
        g = denoiser.generate(ct[0:1], steps=50)
    print(f"  trial {trial}: std={g[0].std().item():.4f}")

# S4 (collapsed): 5 trials
print("\n=== S4: 5 runs, different noise ===")
for trial in range(5):
    torch.manual_seed(42 + trial)
    with torch.no_grad():
        g = denoiser.generate(ct[4:5], steps=50)
    print(f"  trial {trial}: std={g[0].std().item():.4f}")
