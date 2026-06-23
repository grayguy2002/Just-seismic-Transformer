#!/usr/bin/env python3
"""Quick edit test on run011 checkpoint."""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
from JsT import JsT, SeismicConditionEncoder, ConditionSpec, Denoiser
from JsT.dataset import SeismicWaveformDataset, collate_conditions
from torch.utils.data import DataLoader

ckpt = torch.load("outputs/run011/checkpoint-last.pth", map_location="cuda")
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

loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_conditions)
x, cond = next(iter(loader))
x = x.cuda(); cond = {k: v.cuda() for k, v in cond.items()}

# 1. Magnitude sweep
print("=== Magnitude edit ===")
for mag in [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
    c2 = copy.deepcopy(cond)
    c2["source_magnitude"] = torch.full((4,), mag, device=x.device, dtype=torch.float32)
    ct2 = ce(c2)
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct2, steps=50)
    print(f"  M={mag:.1f}: std={g.std(dim=(1,2)).mean().item():.4f}  amp=[{g[0].min().item():.2f},{g[0].max().item():.2f}]")

# 2. Distance sweep
print("\n=== Distance edit ===")
for dist in [0.5, 2.0, 10.0, 30.0, 60.0, 90.0]:
    c2 = copy.deepcopy(cond)
    c2["path_ep_distance_deg"] = torch.full((4,), dist, device=x.device, dtype=torch.float32)
    ct2 = ce(c2)
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct2, steps=50)
    print(f"  dist={dist:4.1f} deg: std={g.std(dim=(1,2)).mean().item():.4f}")

# 3. Phase consistency check
print("\n=== Phase check (same source, different phases) ===")
for ph_name, ph_idx in [("P", 0), ("Pn", 1), ("Pg", 2)]:
    c2 = copy.deepcopy(cond)
    c2["selected_phase"] = torch.full((4,), ph_idx, device=x.device, dtype=torch.long)
    ct2 = ce(c2)
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct2, steps=50)
    print(f"  {ph_name}: std={g.std(dim=(1,2)).mean().item():.3f}  amp=[{g[0].min().item():.2f},{g[0].max().item():.2f}]")

# 4. Conditional vs unconditional
print("\n=== Conditional vs Unconditional CFG test ===")
ct = ce(cond)
for cfg in [1.0, 1.5, 2.0, 3.0]:
    denoiser.cfg_scale = cfg
    torch.manual_seed(42)
    with torch.no_grad():
        g = denoiser.generate(ct, steps=50)
    print(f"  CFG={cfg:.1f}: std={g.std(dim=(1,2)).mean().item():.3f}  amp=[{g[0].min().item():.2f},{g[0].max().item():.2f}]")
