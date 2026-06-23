"""Experiment 5: Scaled whiteness scoring with dense candidate grid.

Key optimisations over Experiment 4:
  1. 300+ candidates via Latin Hypercube (not 50 random)
  2. Pre-compute ODE: 1 forward + 1 reverse per candidate (not per-window)
  3. 20 trials (not 10) for statistical power
  4. Per-window L2 is masking on pre-computed waveforms (zero ODE cost)
  5. Rank correlation: whiteness ranking vs physical parameter ranking
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, ConditionSpec, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


# ---------------------------------------------------------------------------
# ODE primitives (same as previous experiments)
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_ode(denoiser, noise, cond_tokens, steps=50):
    device = noise.device
    B = noise.shape[0]
    net = denoiser.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        xp = net(z, t.expand(B), cond_tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


@torch.no_grad()
def reverse_ode(denoiser, waveform, cond_tokens, steps=30):
    device = waveform.device
    B = waveform.shape[0]
    net = denoiser.net
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    z = waveform.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        xp = net(z, t.expand(B), cond_tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def build_cond_dict(source_info, station, phase_idx=0, sample_rate_hz=40.0, pad_sec=20.0):
    dkm = station["path_ep_distance_km"]
    tt = dkm / 8.0 + max(source_info["source_depth_km"], 0) / 6.0
    arr = (pad_sec + tt) * sample_rate_hz
    return {
        "source_magnitude": torch.tensor(source_info["source_magnitude"], dtype=torch.float32),
        "source_depth_km": torch.tensor(source_info["source_depth_km"], dtype=torch.float32),
        "path_ep_distance_deg": torch.tensor(station["path_ep_distance_deg"], dtype=torch.float32),
        "path_ep_distance_km": torch.tensor(dkm, dtype=torch.float32),
        "path_azimuth_deg": torch.tensor(station["path_azimuth_deg"], dtype=torch.float32),
        "path_back_azimuth_deg": torch.tensor(station["path_back_azimuth_deg"], dtype=torch.float32),
        "phase_travel_sec": torch.tensor(tt, dtype=torch.float32),
        "residual_travel_sec": torch.tensor(0.0, dtype=torch.float32),
        "source_latitude_deg": torch.tensor(source_info["source_latitude_deg"], dtype=torch.float32),
        "source_longitude_deg": torch.tensor(source_info["source_longitude_deg"], dtype=torch.float32),
        "station_latitude_deg": torch.tensor(station["station_latitude_deg"], dtype=torch.float32),
        "station_longitude_deg": torch.tensor(station["station_longitude_deg"], dtype=torch.float32),
        "station_elevation_m": torch.tensor(station.get("station_elevation_m", 0.0), dtype=torch.float32),
        "source_magnitude_type": torch.tensor(0, dtype=torch.long),
        "selected_phase": torch.tensor(phase_idx, dtype=torch.long),
        "trace_channel": torch.tensor(0, dtype=torch.long),
        "station_network_code": torch.tensor(0, dtype=torch.long),
        "selected_phase_arrival_sample": torch.tensor(arr, dtype=torch.float32),
        "selected_phase_arrival_sample_present": torch.tensor(1.0, dtype=torch.float32),
    }


def station_from_pos(src_lat, src_lon, sta_lat, sta_lon):
    dx = (sta_lon - src_lon) * 111.195 * np.cos(np.deg2rad(src_lat))
    dy = (sta_lat - src_lat) * 111.195
    dkm = np.sqrt(dx**2 + dy**2)
    az = np.rad2deg(np.arctan2(dx, dy)) % 360
    return {
        "station_latitude_deg": sta_lat, "station_longitude_deg": sta_lon,
        "station_elevation_m": 0.0,
        "path_ep_distance_deg": dkm / 111.195, "path_ep_distance_km": dkm,
        "path_azimuth_deg": az, "path_back_azimuth_deg": (az + 180) % 360,
    }


# ---------------------------------------------------------------------------
# Noise whiteness scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_noise_whiteness(noise_tensor):
    n = noise_tensor.cpu().float().numpy()
    B, C, T = n.shape
    # Autocorrelation
    ac1s = []
    for ch in range(C):
        d = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
        ac1s.append(abs(np.corrcoef(d[:-1], d[1:])[0, 1]) if len(d) > 1 else 1.0)
    ac1 = np.mean(ac1s)
    # Kurtosis
    from scipy import stats
    ks = []
    for ch in range(C):
        d = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
        ks.append(abs(stats.kurtosis(d, fisher=True)))
    k = np.mean(ks)
    return float(ac1), float(k)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearmanr(x, y):
    """Simple Spearman rank correlation."""
    from scipy.stats import rankdata
    rx = rankdata(x); ry = rankdata(y)
    n = len(rx)
    return 1.0 - 6.0 * np.sum((rx - ry)**2) / (n * (n**2 - 1))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment_5(
    checkpoint_path: str,
    device: torch.device,
    n_trials: int = 20,
    n_candidates: int = 300,
    windows_sec: list[float] | None = None,
    drop_tokens: list[int] | None = None,
) -> dict:
    if windows_sec is None:
        windows_sec = [8.0, 16.0, 40.0]

    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    total_samples = 3200; sr = 40.0

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    print(f"Experiment 5: Scaled whiteness scoring")
    print(f"  Trials: {n_trials}, Candidates: {n_candidates}, Windows: {windows_sec}")
    print()

    # Build Latin Hypercube candidate pool once (relative to arbitrary reference)
    # In each trial we shift the pool to centre on true params
    np.random.seed(42)
    # LHS in 4D [-1, 1], then scaled per trial
    lhs_points = np.zeros((n_candidates, 4))
    for d in range(4):
        lhs_points[:, d] = (np.random.permutation(n_candidates) + np.random.rand(n_candidates)) / n_candidates

    # Ranges (half-widths around truth)
    ranges_hw = np.array([2.0, 200.0, 3.0, 3.0])  # mag, depth_km, lat_deg, lon_deg

    all_summary = []

    for trial in range(n_trials):
        cache_idx = int(torch.randint(0, len(ds_test), (1,)).item())
        waveform_tensor, cond_dict = ds_test[cache_idx]

        true_params = {
            "source_magnitude": float(cond_dict["source_magnitude"].item()),
            "source_depth_km": float(cond_dict["source_depth_km"].item()),
            "source_latitude_deg": float(cond_dict["source_latitude_deg"].item()),
            "source_longitude_deg": float(cond_dict["source_longitude_deg"].item()),
        }
        sta_lat = float(cond_dict["station_latitude_deg"].item())
        sta_lon = float(cond_dict["station_longitude_deg"].item())

        # Build candidate grid: LHS points scaled to ranges around truth
        candidates = []
        for i in range(n_candidates):
            u = lhs_points[i] * 2 - 1  # [-1, 1]
            mg = true_params["source_magnitude"] + u[0] * ranges_hw[0]
            dp = max(0.0, true_params["source_depth_km"] + u[1] * ranges_hw[1])
            lt = true_params["source_latitude_deg"] + u[2] * ranges_hw[2]
            ln = true_params["source_longitude_deg"] + u[3] * ranges_hw[3]
            if dp > 700: dp = 700.0
            si = {"source_magnitude": mg, "source_depth_km": dp,
                  "source_latitude_deg": lt, "source_longitude_deg": ln,
                  "source_magnitude_type": "mw"}
            sta = station_from_pos(lt, ln, sta_lat, sta_lon)
            cond = build_cond_dict(si, sta)
            # Physical error
            perr = np.sqrt(
                ((mg - true_params["source_magnitude"]) / 2.0)**2 +
                ((dp - true_params["source_depth_km"]) / 100.0)**2 +
                ((lt - true_params["source_latitude_deg"]) / 2.0)**2 +
                ((ln - true_params["source_longitude_deg"]) / 2.0)**2
            )
            candidates.append({"cond": cond, "param_error": perr,
                               "mag_err": abs(mg - true_params["source_magnitude"]),
                               "depth_err": abs(dp - true_params["source_depth_km"])})

        # ---- Pre-compute ODEs ----
        t0 = time.time()

        # True waveform (once)
        cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
        true_tokens = ce(cond_gpu)
        seed_base = 42 + trial * 1000
        torch.manual_seed(seed_base)
        if device.type == "cuda": torch.cuda.manual_seed_all(seed_base)
        noise_syn = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
        W_true = forward_ode(dn, noise_syn, true_tokens, steps=50)

        # For each candidate: encode tokens, compute forward W and reverse noise
        for ci, cand in enumerate(candidates):
            cb = {k: v.unsqueeze(0).to(device) for k, v in cand["cond"].items()}
            ct = ce(cb)
            # Forward from SAME noise as true
            W = forward_ode(dn, noise_syn, ct, steps=50)
            # Reverse
            zr = reverse_ode(dn, W_true, ct, steps=30)
            ac1, kurt = compute_noise_whiteness(zr)
            cand["w_cand"] = W  # keep on GPU
            cand["whiteness"] = ac1 + kurt * 0.1
            cand["ac1"] = ac1
            cand["kurt"] = kurt

        elapsed_ode = time.time() - t0

        # ---- Score per window (masking only, no new ODE) ----
        per_window = {}
        for win_sec in windows_sec:
            win_s = min(int(win_sec * sr), total_samples)
            for cand in candidates:
                obs_diff = (cand["w_cand"] - W_true)[:, :, :win_s]
                cand["wf_l2"] = float((obs_diff**2).mean().sqrt() / max((W_true[:,:,:win_s]**2).mean().sqrt().item(), 1e-8))

            # Rank by each metric
            by_whiteness = sorted(candidates, key=lambda c: c["whiteness"])
            by_wf_l2 = sorted(candidates, key=lambda c: c["wf_l2"])
            by_combined = sorted(candidates, key=lambda c: c["wf_l2"] + c["whiteness"] * 0.5)

            res = {}
            for label, ranked in [("whiteness", by_whiteness), ("wf_l2", by_wf_l2), ("combined", by_combined)]:
                top1 = ranked[0]; top5 = ranked[:5]; top50 = ranked[:50]
                res[f"{label}_top1_perr"] = top1["param_error"]
                res[f"{label}_top1_mag"] = top1["mag_err"]
                res[f"{label}_top1_depth"] = top1["depth_err"]
                res[f"{label}_top5_perr"] = float(np.mean([c["param_error"] for c in top5]))
                res[f"{label}_top5_mag"] = float(np.mean([c["mag_err"] for c in top5]))
                res[f"{label}_top5_depth"] = float(np.mean([c["depth_err"] for c in top5]))
                res[f"{label}_top50_perr"] = float(np.mean([c["param_error"] for c in top50]))
                # Rank correlation: whiteness rank vs param_error rank
                perrs = np.array([c["param_error"] for c in ranked])
                metric_vals = np.array([c["whiteness"] if label=="whiteness" else (c["wf_l2"] if label=="wf_l2" else (c["wf_l2"]+c["whiteness"]*0.5)) for c in ranked])
                res[f"{label}_rank_corr"] = float(spearmanr(metric_vals, perrs))

            per_window[f"{win_sec:.0f}s"] = res

        all_summary.append({
            "trial": trial + 1, "true_params": true_params,
            "n_candidates": n_candidates, "ode_time_s": float(elapsed_ode),
            "windows": per_window,
        })

        print(f"  Trial {trial+1:2d}/{n_trials}  "
              f"M={true_params['source_magnitude']:.1f}  "
              f"depth={true_params['source_depth_km']:.0f}km  "
              f"ODE={elapsed_ode:.0f}s  "
              f"whiteness_top1_mag={per_window['8s']['whiteness_top1_mag']:.2f}  "
              f"wf_l2_top1_mag={per_window['8s']['wf_l2_top1_mag']:.2f}")

    # ---- Aggregate ----
    print(f"\n{'='*90}")
    print(f"  EXPERIMENT 5 — AGGREGATE ({n_trials} trials × {n_candidates} candidates)")
    print(f"{'='*90}\n")

    for win_key in [f"{w:.0f}s" for w in windows_sec]:
        for metric in ["whiteness", "wf_l2"]:
            topl = f"{metric}_top1_perr"
            magl = f"{metric}_top1_mag"
            depl = f"{metric}_top1_depth"
            t5l = f"{metric}_top5_mag"
            corl = f"{metric}_rank_corr"
            vals = [t["windows"][win_key][topl] for t in all_summary]
            mags = [t["windows"][win_key][magl] for t in all_summary]
            deps = [t["windows"][win_key][depl] for t in all_summary]
            t5m  = [t["windows"][win_key][t5l] for t in all_summary]
            cors = [t["windows"][win_key][corl] for t in all_summary]
            print(f"  {win_key:>5s} {metric:>12s}: perr={np.mean(vals):.3f}±{np.std(vals):.3f}  "
                  f"top1_mag={np.mean(mags):.2f}±{np.std(mags):.2f}  "
                  f"top5_mag={np.mean(t5m):.2f}  "
                  f"top1_depth={np.mean(deps):.0f}±{np.std(deps):.0f}km  "
                  f"rank_corr={np.mean(cors):.3f}")

    # Key delta
    print(f"\n{'='*90}")
    print(f"  WHITENESS vs WF_L2 — SCALED (300 candidates)")
    print(f"{'='*90}\n")
    for win_key in [f"{w:.0f}s" for w in windows_sec]:
        wf  = [t["windows"][win_key]["wf_l2_top1_perr"] for t in all_summary]
        wh  = [t["windows"][win_key]["whiteness_top1_perr"] for t in all_summary]
        d = (np.mean(wf) - np.mean(wh)) / np.mean(wf) * 100
        wh_mag = np.mean([t["windows"][win_key]["whiteness_top1_mag"] for t in all_summary])
        wf_mag = np.mean([t["windows"][win_key]["wf_l2_top1_mag"] for t in all_summary])
        wh_cor = np.mean([t["windows"][win_key]["whiteness_rank_corr"] for t in all_summary])
        wf_cor = np.mean([t["windows"][win_key]["wf_l2_rank_corr"] for t in all_summary])
        print(f"  {win_key:>5s}: whiteness_perr={np.mean(wh):.3f}  wf_l2_perr={np.mean(wf):.3f}  "
              f"delta={d:+.0f}%  |  wh_mag={wh_mag:.2f}  wf_mag={wf_mag:.2f}  |  "
              f"cor_wh={wh_cor:.3f}  cor_wf={wf_cor:.3f}")

    # Final verdict
    overall_delta = np.mean([
        (np.mean([t["windows"][f"{w:.0f}s"]["wf_l2_top1_perr"] for t in all_summary])
         - np.mean([t["windows"][f"{w:.0f}s"]["whiteness_top1_perr"] for t in all_summary]))
        / np.mean([t["windows"][f"{w:.0f}s"]["wf_l2_top1_perr"] for t in all_summary]) * 100
        for w in windows_sec
    ])
    print(f"\n  OVERALL whiteness advantage: {overall_delta:+.0f}%")
    if overall_delta > 15:
        print("  FINAL: Whiteness is a RELIABLE candidate scoring metric.")
    elif overall_delta > 5:
        print("  FINAL: Whiteness provides modest improvement over wf_l2.")
    else:
        print("  FINAL: Whiteness does NOT improve candidate selection at scale.")

    return all_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Scaled whiteness scoring")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/candidate_score_exp5")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-candidates", type=int, default=300)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_experiment_5(
        args.checkpoint, device,
        n_trials=args.n_trials, n_candidates=args.n_candidates,
        drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lightweight save: only per-trial summary, not full candidate data
    with open(output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/summary.json")
