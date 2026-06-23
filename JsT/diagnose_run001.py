#!/usr/bin/env python3
"""Diagnose JsT run001 generation quality."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np, torch
from JsT import JsT, SeismicConditionEncoder, ConditionSpec, Denoiser
from JsT.dataset import SeismicWaveformDataset, collate_conditions
from torch.utils.data import DataLoader

# Use training split for vocab (frozen at training time)
ds_train = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="training", augment=False)
ds = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="validation", augment=False)
spec = ConditionSpec(magnitude_types=ds_train.magtype_vocab, phases=ds_train.phase_vocab,
                     channels=ds_train.channel_vocab, network_codes=ds_train.network_vocab, hidden_dim=512)
jst = JsT(3200, 64, 3, 512, 8, 8).cuda()
ce = SeismicConditionEncoder(spec).cuda()
denoiser = Denoiser(jst).cuda()

ckpt = torch.load("outputs/run001/checkpoint-0199.pth", map_location="cuda")
denoiser.load_state_dict(ckpt["denoiser"])
ce.load_state_dict(ckpt["cond_encoder"])
denoiser.eval(); ce.eval()

loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_conditions)
x, cond = next(iter(loader))
x = x.cuda(); cond = {k: v.cuda() for k, v in cond.items()}
ct = ce(cond)

with torch.no_grad():
    gen = denoiser.generate(ct, steps=50)
    null = jst.null_tokens.expand(8, -1, -1)
    gen_uncond = denoiser.generate(null, steps=50)

for i in range(4):
    r = x[i].cpu().numpy()
    g = gen[i].cpu().numpy()
    u = gen_uncond[i].cpu().numpy()
    print(f"Sample {i}:")
    print(f"  real:    [{r.min():.3f},{r.max():.3f}] std={r.std():.3f}")
    print(f"  cond:    [{g.min():.3f},{g.max():.3f}] std={g.std():.3f}")
    print(f"  uncond:  [{u.min():.3f},{u.max():.3f}] std={u.std():.3f}")

# Also test without EMA
denoiser.load_state_dict(ckpt["denoiser"])
denoiser.eval()
with torch.no_grad():
    gen_noema = denoiser.generate(ct, steps=50)
g2 = gen_noema[0].cpu().numpy()
print(f"\nNo EMA: [{g2.min():.3f},{g2.max():.3f}] std={g2.std():.3f}")

# Test different timestep counts
for steps in [10, 50, 100, 200]:
    with torch.no_grad():
        g = denoiser.generate(ct[:2], steps=steps)
    print(f"steps={steps}: [{g[0].min():.3f},{g[0].max():.3f}] std={g[0].std():.3f}")
