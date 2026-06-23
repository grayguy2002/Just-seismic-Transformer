#!/usr/bin/env python3
"""Comprehensive condition-editing validation on run011 checkpoint.

Tests every condition dimension for:
  (A) Monotonicity — does std/amplitude change in the expected physical direction?
  (B) Disentanglement — does editing condition X leave other waveform properties intact?
  (C) Seed consistency — is the edit stable across random seeds?
"""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, numpy as np
from JsT import JsT, SeismicConditionEncoder, ConditionSpec, Denoiser
from JsT.dataset import SeismicWaveformDataset, collate_conditions
from torch.utils.data import DataLoader

ckpt = torch.load("outputs/run014/checkpoint-last.pth", map_location="cuda")
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
denoiser.cfg_scale = 1.0

# Take 3 real validation samples as anchors
loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=collate_conditions)
_, anchor_cond = next(iter(loader))
anchor_cond = {k: v.cuda() for k, v in anchor_cond.items()}

B = 3
N_STEPS = 50

def generate(c, seed=42):
    ct = ce(c)
    torch.manual_seed(seed)
    with torch.no_grad():
        return denoiser.generate(ct, steps=N_STEPS)

def stats(g):
    """Return (mean_std, mean_abs_amp, max_abs) across batch and channels."""
    std   = g.std(dim=(1,2)).mean().item()
    abs_amp = g.abs().mean(dim=(1,2)).mean().item()
    max_abs = g.abs().max(dim=2).values.max(dim=1).values.mean().item()
    return std, abs_amp, max_abs

# =========================================================================
# 1. MAGNITUDE: M 2.0 → 8.0
# =========================================================================
print("=" * 60)
print("1. MAGNITUDE SWEEP (expect: amplitude ↑ with M, std ↑)")
print("=" * 60)
for mag in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    c = copy.deepcopy(anchor_cond)
    c["source_magnitude"] = torch.full((B,), mag, device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    print(f"  M={mag:.0f}:  std={s:.4f}  abs_amp={a:.4f}  max_abs={m:.4f}")

# =========================================================================
# 2. DEPTH: 0 → 600 km
# =========================================================================
print("\n" + "=" * 60)
print("2. DEPTH SWEEP (expect: depth ↑ → simpler waveform, less surface waves)")
print("=" * 60)
for d in [0, 5, 15, 35, 70, 150, 300, 600]:
    c = copy.deepcopy(anchor_cond)
    c["source_depth_km"] = torch.full((B,), float(d), device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    print(f"  depth={d:4d}km: std={s:.4f}  abs_amp={a:.4f}  max_abs={m:.4f}")

# =========================================================================
# 3. DISTANCE: 0.1° → 95°
# =========================================================================
print("\n" + "=" * 60)
print("3. DISTANCE SWEEP (expect: ampl ↓ with distance, coda length changes)")
print("=" * 60)
for dist in [0.2, 1.0, 3.0, 10.0, 30.0, 60.0, 90.0]:
    c = copy.deepcopy(anchor_cond)
    c["path_ep_distance_deg"] = torch.full((B,), dist, device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    print(f"  dist={dist:5.1f} deg: std={s:.4f}  abs_amp={a:.4f}  max_abs={m:.4f}")

# =========================================================================
# 4. AZIMUTH CIRCULARITY: 0→360° → 0 (should produce same waveform)
# =========================================================================
print("\n" + "=" * 60)
print("4. AZIMUTH CIRCULARITY (expect: 0° ≈ 360°, distinct from 180°)")
print("=" * 60)
for az in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
    c = copy.deepcopy(anchor_cond)
    c["path_azimuth_deg"] = torch.full((B,), float(az), device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    mark = ""
    if az == 0:
        g0 = g.clone()
    elif az == 360:
        diff = (g - g0).norm() / g0.norm()
        mark = f"  |g360-g0|/|g0|={diff.item():.4f}"
    print(f"  az={az:3d} deg: std={s:.4f}  abs_amp={a:.4f}{mark}")

# =========================================================================
# 5. BACK-AZIMUTH: same circularity check
# =========================================================================
print("\n" + "=" * 60)
print("5. BACK-AZIMUTH CIRCULARITY")
print("=" * 60)
for baz in [0, 90, 180, 270, 360]:
    c = copy.deepcopy(anchor_cond)
    c["path_back_azimuth_deg"] = torch.full((B,), float(baz), device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    mark = ""
    if baz == 0:
        g0_baz = g.clone()
    elif baz == 360:
        diff = (g - g0_baz).norm() / g0_baz.norm()
        mark = f"  |g360-g0|/|g0|={diff.item():.4f}"
    print(f"  baz={baz:3d} deg: std={s:.4f}  abs_amp={a:.4f}{mark}")

# =========================================================================
# 6. DISENTANGLEMENT: does magnitude change affect arrival time?
# =========================================================================
print("\n" + "=" * 60)
print("6. DISENTANGLEMENT: arrival-time stability under M edit")
print("=" * 60)
# Compare P-wave arrival (first large positive peak after sample ~800 = 20s pre)
def arrival_sample(g, ch=0, start=800):
    w = g[0, ch, start:].cpu().numpy()
    # Find first sample where abs exceeds 50% of max
    thr = np.abs(w).max() * 0.5
    crossings = np.where(np.abs(w) > thr)[0]
    return start + (crossings[0] if len(crossings) > 0 else 0)

c_m2 = copy.deepcopy(anchor_cond)
c_m2["source_magnitude"][:] = 2.0
c_m7 = copy.deepcopy(anchor_cond)
c_m7["source_magnitude"][:] = 7.0
g2 = generate(c_m2); g7 = generate(c_m7)
for i in range(B):
    a2 = arrival_sample(g2[i:i+1]); a7 = arrival_sample(g7[i:i+1])
    print(f"  S{i}: M=2 arrival={a2}  M=7 arrival={a7}  shift={(a7-a2)/40:.3f}s")

# =========================================================================
# 7. RESIDUAL TT: negative→positive should shift arrival
# =========================================================================
print("\n" + "=" * 60)
print("7. RESIDUAL TRAVEL TIME (expect: neg=earlier, pos=later arrival)")
print("=" * 60)
for rtt in [-20, -10, 0, 10, 20]:
    c = copy.deepcopy(anchor_cond)
    c["residual_travel_sec"] = torch.full((B,), float(rtt), device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    arr = arrival_sample(g[0:1])
    print(f"  res_tt={rtt:+3.0f}s: std={s:.4f}  arrival_sample={arr}")

# =========================================================================
# 8. STATION ELEVATION
# =========================================================================
print("\n" + "=" * 60)
print("8. STATION ELEVATION (expect: higher → more amplification?)")
print("=" * 60)
for elev in [0, 500, 1000, 2000, 4000]:
    c = copy.deepcopy(anchor_cond)
    c["station_elevation_m"] = torch.full((B,), float(elev), device="cuda", dtype=torch.float32)
    g = generate(c)
    s, a, m = stats(g)
    print(f"  elev={elev:5d}m: std={s:.4f}  abs_amp={a:.4f}  max_abs={m:.4f}")

# =========================================================================
# 9. CHANNEL SWITCH
# =========================================================================
print("\n" + "=" * 60)
print("9. CHANNEL SWITCH (expect: BH narrower-band than HH)")
print("=" * 60)
for ch_name, ch_idx in [("BH", 0), ("HH", 1), ("OTHER", 2)]:
    c = copy.deepcopy(anchor_cond)
    c["trace_channel"] = torch.full((B,), ch_idx, device="cuda", dtype=torch.long)
    g = generate(c)
    s, a, m = stats(g)
    print(f"  {ch_name}: std={s:.4f}  abs_amp={a:.4f}  max_abs={m:.4f}")

# =========================================================================
# 10. SEED CONSISTENCY: 5 seeds same condition
# =========================================================================
print("\n" + "=" * 60)
print("10. SEED CONSISTENCY (expect: std-of-std < 0.05)")
print("=" * 60)
ct = ce(anchor_cond)
all_stds = []
for seed in range(10):
    torch.manual_seed(seed)
    with torch.no_grad():
        g = denoiser.generate(ct, steps=N_STEPS)
    all_stds.append(g.std(dim=(1,2)).cpu().numpy())
all_stds = np.array(all_stds)  # (10, B)
for i in range(B):
    print(f"  S{i}: mean_std={all_stds[:,i].mean():.4f}  std_of_std={all_stds[:,i].std():.4f}  "
          f"min={all_stds[:,i].min():.4f}  max={all_stds[:,i].max():.4f}")
print(f"  Overall mean_std={all_stds.mean():.4f}  std_of_std={all_stds.std():.4f}")

# =========================================================================
# 11. CFG SWEEP: does cfg_scale improve quality?
# =========================================================================
print("\n" + "=" * 60)
print("11. CFG SWEEP")
print("=" * 60)
for cfg in [1.0, 1.5, 2.0, 3.0, 4.0]:
    denoiser.cfg_scale = cfg
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct, steps=N_STEPS)
    s, a, m = stats(g)
    print(f"  CFG={cfg:.1f}: std={s:.4f}  abs_amp={a:.4f}  max_abs={m:.4f}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
