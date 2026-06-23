#!/usr/bin/env python3
"""Visualize condition-editing effects — run019 (final FiLM configuration)."""

import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch, numpy as np
from pathlib import Path
from JsT import load_checkpoint_models
from JsT.dataset import SeismicWaveformDataset, collate_conditions
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TIME = np.arange(-20, 60, 1 / 40)  # s
CH_LAB = ["Z", "N", "E"]
CH_CLR = ["#1b4f72", "#2e86c1", "#85c1e9"]  # seaborn-compatible

# ── helpers ──────────────────────────────────────────────────
def load_run(name, device="cuda"):
    ce, dn, _ = load_checkpoint_models(
        f"outputs/{name}/checkpoint-last.pth",
        torch.device(device),
        use_ema=True,
    )
    return ce, dn

def encode_conditions(ce, cond):
    device = next(ce.parameters()).device
    cond = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in cond.items()}
    return ce(cond)


def generate(ce, dn, cond, seed=42, steps=50):
    ct = encode_conditions(ce, cond)
    torch.manual_seed(seed)
    with torch.no_grad():
        g = dn.generate(ct, steps=steps)
    return g.cpu().numpy()  # (B, 3, 3200)

def vline(ax, t=0):
    ax.axvline(t, color="#d62728", ls="--", lw=0.5, alpha=0.4)

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=6)

def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))

def rotate_ne_rt(w, baz_deg):
    theta = np.deg2rad(baz_deg)
    n, e = w[1], w[2]
    radial = n * np.cos(theta) + e * np.sin(theta)
    transverse = -n * np.sin(theta) + e * np.cos(theta)
    return radial, transverse

print("Loading model (run019)...")
ce, dn = load_run("run019")

# Validation dataset
ds_train = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="training", augment=False)
ds = SeismicWaveformDataset("data/seisbench_mlaapde_pwave_v1", split="validation", augment=False, vocab_from=ds_train)
_, anchor = ds[3]
anchor = {k: torch.as_tensor(v).cuda().unsqueeze(0) for k, v in anchor.items()}

# ══════════════════════════════════════════════════════════════
# FIG 1: MAGNITUDE — 3×3 grid (mag×channel)
# ══════════════════════════════════════════════════════════════
print("Fig 1: Magnitude…")
fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharex=True, sharey="row")
for i, mag in enumerate([2.5, 4.5, 6.5]):
    c = copy.deepcopy(anchor)
    c["source_magnitude"] = torch.full((1,), mag, device="cuda", dtype=torch.float32)
    w = generate(ce, dn, c)[0]
    for ch in range(3):
        ax = axes[ch, i]
        ax.plot(TIME, w[ch], color=CH_CLR[ch], lw=0.6)
        vline(ax)
        style(ax)
        if i == 0:
            ax.set_ylabel(CH_LAB[ch], fontsize=9)
    axes[0, i].set_title(f"M = {mag:.1f}", fontsize=10)
fig.suptitle("Magnitude Editing — final FiLM configuration (run019)", fontsize=12, y=0.98)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig01_magnitude.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIG 2: DISTANCE — 1 row, 4 panels (Z channel only)
# ══════════════════════════════════════════════════════════════
print("Fig 2: Distance…")
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
for j, dist in enumerate([0.2, 3.0, 30.0, 90.0]):
    c = copy.deepcopy(anchor)
    c["path_ep_distance_deg"] = torch.full((1,), dist, device="cuda", dtype=torch.float32)
    w = generate(ce, dn, c)[0, 0]
    axes[j].plot(TIME, w, color=CH_CLR[0], lw=0.6)
    vline(axes[j])
    style(axes[j])
    axes[j].set_title(f"Δ = {dist:.1f}°", fontsize=10)
axes[0].set_ylabel("Z", fontsize=10)
fig.suptitle("Distance Editing — Vertical channel", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig02_distance.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIG 3: AZIMUTH + BAZ HORIZONTAL DIAGNOSTIC
# ══════════════════════════════════════════════════════════════
print("Fig 3: Azimuth / BAZ…")
angles = [0, 90, 180, 270]

c_az0 = copy.deepcopy(anchor)
c_az0["path_azimuth_deg"] = torch.full((1,), 0.0, device="cuda", dtype=torch.float32)
w_az0 = generate(ce, dn, c_az0)[0]
c_az360 = copy.deepcopy(anchor)
c_az360["path_azimuth_deg"] = torch.full((1,), 360.0, device="cuda", dtype=torch.float32)
w_az360 = generate(ce, dn, c_az360)[0]
az360_rel = rel_l2(w_az360[1:], w_az0[1:])

c_baz0 = copy.deepcopy(anchor)
c_baz0["path_back_azimuth_deg"] = torch.full((1,), 0.0, device="cuda", dtype=torch.float32)
w_baz0 = generate(ce, dn, c_baz0)[0]
c_baz360 = copy.deepcopy(anchor)
c_baz360["path_back_azimuth_deg"] = torch.full((1,), 360.0, device="cuda", dtype=torch.float32)
w_baz360 = generate(ce, dn, c_baz360)[0]
baz360_rel = rel_l2(w_baz360[1:], w_baz0[1:])

fig, axes = plt.subplots(2, 4, figsize=(14, 5.2), sharex=True, sharey="row")
for j, deg in enumerate(angles):
    ca = copy.deepcopy(anchor)
    ca["path_azimuth_deg"] = torch.full((1,), float(deg), device="cuda", dtype=torch.float32)
    wa = generate(ce, dn, ca)[0]
    axes[0, j].plot(TIME, wa[1], color=CH_CLR[1], lw=0.55, label="N")
    axes[0, j].plot(TIME, wa[2], color=CH_CLR[2], lw=0.55, label="E")
    vline(axes[0, j]); style(axes[0, j])
    axes[0, j].set_title(f"az = {deg}°", fontsize=10)
    if j == 0:
        axes[0, j].legend(frameon=False, fontsize=7, loc="upper right")

    cb = copy.deepcopy(anchor)
    cb["path_back_azimuth_deg"] = torch.full((1,), float(deg), device="cuda", dtype=torch.float32)
    wb = generate(ce, dn, cb)[0]
    radial, transverse = rotate_ne_rt(wb, deg)
    axes[1, j].plot(TIME, radial, color="#1b4f72", lw=0.55, label="R")
    axes[1, j].plot(TIME, transverse, color="#d4ac0d", lw=0.55, label="T")
    vline(axes[1, j]); style(axes[1, j])
    axes[1, j].set_title(f"baz = {deg}°", fontsize=10)
    if j == 0:
        axes[1, j].legend(frameon=False, fontsize=7, loc="upper right")

axes[0, 0].set_ylabel("Azimuth\nN/E", fontsize=9)
axes[1, 0].set_ylabel("Back-azimuth\nR/T", fontsize=9)
fig.suptitle(
    f"Azimuth / Back-Azimuth Editing — horizontal components, 0° vs 360° rel-L2: "
    f"az={az360_rel:.3f}, baz={baz360_rel:.3f}",
    fontsize=12,
)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig03_azimuth_baz.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIG 4: DEPTH + ELEVATION
# ══════════════════════════════════════════════════════════════
print("Fig 4: Depth + Elevation…")
fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharex=True, sharey="row")
for j, d in enumerate([5, 35, 150, 600]):
    c = copy.deepcopy(anchor)
    c["source_depth_km"] = torch.full((1,), float(d), device="cuda", dtype=torch.float32)
    w = generate(ce, dn, c)[0, 0]
    axes[0, j].plot(TIME, w, color=CH_CLR[2], lw=0.6)
    vline(axes[0, j]); style(axes[0, j])
    axes[0, j].set_title(f"depth = {d} km", fontsize=10)

for j, e in enumerate([0, 500, 1000, 4000]):
    c = copy.deepcopy(anchor)
    c["station_elevation_m"] = torch.full((1,), float(e), device="cuda", dtype=torch.float32)
    w = generate(ce, dn, c)[0, 0]
    axes[1, j].plot(TIME, w, color=CH_CLR[2], lw=0.6)
    vline(axes[1, j]); style(axes[1, j])
    axes[1, j].set_title(f"elev = {e} m", fontsize=10)

axes[0,0].set_ylabel("Depth", fontsize=10)
axes[1,0].set_ylabel("Elevation", fontsize=10)
fig.suptitle("Depth & Elevation Editing", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig04_depth_elevation.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIG 5: CFG SWEEP
# ══════════════════════════════════════════════════════════════
print("Fig 5: CFG sweep…")
ct_anchor = encode_conditions(ce, anchor)
cfg_values = [1.0, 1.5, 2.0, 3.0, 4.0]
cfg_peaks = []
fig, axes = plt.subplots(1, len(cfg_values), figsize=(16, 3), sharey=True)
for j, cfg in enumerate(cfg_values):
    dn.cfg_scale = cfg
    torch.manual_seed(42)
    with torch.no_grad():
        g = dn.generate(ct_anchor, steps=50)
    w = g[0, 0].cpu().numpy()
    axes[j].plot(TIME, w, color=CH_CLR[0], lw=0.6)
    vline(axes[j]); style(axes[j])
    axes[j].set_title(f"CFG = {cfg:.1f}", fontsize=10)
    peak = np.abs(w).max()
    cfg_peaks.append(peak)
    axes[j].text(0.98, 0.95, f"peak={peak:.2f}", transform=axes[j].transAxes,
                 ha="right", va="top", fontsize=7, color="gray")
dn.cfg_scale = 1.0
fig.suptitle("Classifier-Free Guidance — conditional amplification", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig05_cfg_sweep.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIG 6: SEED CONSISTENCY
# ══════════════════════════════════════════════════════════════
print("Fig 6: Seed consistency…")
fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharex=True, sharey=True)
c0 = copy.deepcopy(anchor)
c0["source_magnitude"] = torch.full((1,), 6.5, device="cuda", dtype=torch.float32)
for j, s in enumerate([0, 1, 42, 99]):
    g = generate(ce, dn, c0, seed=s)[0, 0]
    axes[j].plot(TIME, g, color=CH_CLR[0], lw=0.6)
    vline(axes[j]); style(axes[j])
    axes[j].set_title(f"seed = {s}", fontsize=10)
fig.suptitle("Seed Consistency — Same condition, 4 random seeds", fontsize=12)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig06_seed_consistency.png", dpi=200, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIG 7: SUMMARY TABLE
# ══════════════════════════════════════════════════════════════
print("Fig 7: Summary table…")
fig, ax = plt.subplots(figsize=(11.5, 4.4))
ax.axis("off")
rows = [
    ("Magnitude",        "Strong",        "Clear morphology/amplitude response in normalized space; physical counts need norm_scale."),
    ("Distance",         "Strong",        "Path token controls envelope/coda; response is visible but not strictly monotonic."),
    ("Depth",            "Visible",       "Deep events generate simpler, shorter-coda waveforms in this anchor."),
    ("Azimuth",          "Visible",       f"Horizontal N/E components change; 0-360 circular rel-L2={az360_rel:.3f}."),
    ("Back-azimuth",     "Visible",       f"R/T decomposition changes with baz; 0-360 circular rel-L2={baz360_rel:.3f}."),
    ("CFG scale",        "Moderate",      f"CFG 1.0-4.0 changes Z peak {cfg_peaks[0]:.2f}->{cfg_peaks[-1]:.2f} for this anchor."),
    ("Elevation",        "Weak",          "Small but nonzero response; needs broader station/site validation."),
    ("Phase (P/Pn/Pg)",  "Weak/Redundant","Mostly derived from distance and depth; little independent controllability."),
    ("Channel (BH/HH)",  "Weak",          "40 Hz data suppresses high-frequency instrument-band differences."),
    ("Residual TT",      "Weak/Absorbed", "Residual timing information appears absorbed by path representation."),
]
tab = ax.table(cellText=rows, colLabels=["Condition", "Editability", "Evidence / interpretation"],
               cellLoc="left", loc="center", colWidths=[0.16, 0.13, 0.71])
tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1.15, 1.5)
for i in range(len(rows)):
    for j in range(3):
        tab[i+1, j].set_fontsize(7)
for j in range(3):
    tab[0, j].set_fontsize(9); tab[0, j].set_facecolor("#1b4f72")
    tab[0, j].set_text_props(color="white", fontweight="bold")
fig.suptitle("JsT Condition Editing — 19-run sweep, run019 final checkpoint", fontsize=12, y=0.9)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig07_summary.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\nDone — {len(list(OUT_DIR.glob('*.png')))} figures saved to {OUT_DIR.resolve()}")
for p in sorted(OUT_DIR.glob("*.png")):
    print(f"  {p.name:30s}  {p.stat().st_size/1024:6.0f} KB")
