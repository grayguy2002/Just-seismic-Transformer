"""Experiment 4: Candidate scoring via reverse-ODE noise whiteness.

Samples candidates in PHYSICAL parameter space (lat, lon, depth, mag),
encodes them to tokens, then scores each via:
  1. Waveform L2 on observed window (standard metric)
  2. Reverse-ODE noise whiteness (autocorr + kurtosis)

If noise whiteness can correctly rank candidates, the best-performing
candidate by whiteness should correlate with physical parameter accuracy.

Unlike Experiments 1-3, this operates in 4-D parameter space (not 4096-D
token space), making discrete search computationally feasible.
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


@torch.no_grad()
def reverse_ode(denoiser, waveform, cond_tokens, steps=30):
    """ODE backward: clean waveform → initial noise."""
    device = waveform.device
    B = waveform.shape[0]
    net = denoiser.net
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)
    z = waveform.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        x_pred = net(z, t.expand(B), cond_tokens)
        v = (x_pred - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


@torch.no_grad()
def forward_ode(denoiser, noise, cond_tokens, steps=50):
    """Forward ODE: noise → waveform."""
    device = noise.device
    B = noise.shape[0]
    net = denoiser.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        x_pred = net(z, t.expand(B), cond_tokens)
        v = (x_pred - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


def compute_noise_whiteness(noise_tensor: torch.Tensor) -> dict:
    """Compute scalar whiteness metrics on recovered noise."""
    n = noise_tensor.cpu().float().numpy()
    B, C, T = n.shape
    results = {}
    # Per-channel lag-1 autocorrelation
    ac1_vals = []
    for ch in range(C):
        ch_data = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
        ac1 = np.corrcoef(ch_data[:-1], ch_data[1:])[0, 1] if len(ch_data) > 1 else 0
        ac1_vals.append(abs(ac1))
    results["autocorr_l1"] = float(np.mean(ac1_vals))

    # Excess kurtosis (absolute)
    from scipy import stats
    k_vals = []
    for ch in range(C):
        ch_data = n[0, ch, :] if B == 1 else n[:, ch, :].flatten()
        k_vals.append(abs(stats.kurtosis(ch_data, fisher=True)))
    results["kurtosis"] = float(np.mean(k_vals))

    # Combined: lower = whiter
    results["whiteness_score"] = results["autocorr_l1"] + results["kurtosis"] * 0.1
    return results


def build_candidate_encoder_input(
    source_info: dict,
    station: dict,
    phase_idx: int = 0,
    channel: str = "BH",
    network: str = "XX",
    sample_rate_hz: float = 40.0,
    window_padding_sec: float = 20.0,
) -> dict[str, torch.Tensor]:
    """Build condition dict for one station with given source parameters."""
    dist_km = station["path_ep_distance_km"]
    depth_km = source_info["source_depth_km"]
    tt_sec = dist_km / 8.0 + depth_km / 6.0
    arrival_sample = (window_padding_sec + tt_sec) * sample_rate_hz

    return {
        "source_magnitude": torch.tensor(source_info["source_magnitude"], dtype=torch.float32),
        "source_depth_km": torch.tensor(depth_km, dtype=torch.float32),
        "path_ep_distance_deg": torch.tensor(station["path_ep_distance_deg"], dtype=torch.float32),
        "path_ep_distance_km": torch.tensor(dist_km, dtype=torch.float32),
        "path_azimuth_deg": torch.tensor(station["path_azimuth_deg"], dtype=torch.float32),
        "path_back_azimuth_deg": torch.tensor(station["path_back_azimuth_deg"], dtype=torch.float32),
        "phase_travel_sec": torch.tensor(tt_sec, dtype=torch.float32),
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
        "selected_phase_arrival_sample": torch.tensor(arrival_sample, dtype=torch.float32),
        "selected_phase_arrival_sample_present": torch.tensor(1.0, dtype=torch.float32),
    }


def generate_station_info(
    source_lat: float, source_lon: float,
    sta_lat: float, sta_lon: float,
) -> dict:
    """Compute station geometry from positions."""
    dx_km = (sta_lon - source_lon) * 111.195 * np.cos(np.deg2rad(source_lat))
    dy_km = (sta_lat - source_lat) * 111.195
    dist_km = np.sqrt(dx_km**2 + dy_km**2)
    dist_deg = dist_km / 111.195
    azimuth = np.rad2deg(np.arctan2(dx_km, dy_km)) % 360
    return {
        "station_latitude_deg": sta_lat,
        "station_longitude_deg": sta_lon,
        "station_elevation_m": 0.0,
        "path_ep_distance_deg": dist_deg,
        "path_ep_distance_km": dist_km,
        "path_azimuth_deg": azimuth,
        "path_back_azimuth_deg": (azimuth + 180) % 360,
    }


def run_experiment_4(
    checkpoint_path: str,
    device: torch.device,
    n_trials: int = 10,
    n_candidates_per_param: int = 5,
    windows_sec: list[float] | None = None,
    n_stations: int = 1,
    drop_tokens: list[int] | None = None,
) -> dict:
    """Candidate scoring via noise whiteness.

    For each trial:
    1. Take a real waveform + true source params from test set
    2. Build N_candidates * 4 params discrete candidates around truth
    3. For each candidate: encode → reverse ODE → whiteness score
    4. Also compute waveform L2 on observed window
    5. Check: does best whiteness score → best params?
    """
    if windows_sec is None:
        windows_sec = [8.0, 16.0, 40.0, 80.0]

    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    total_samples = 3200
    sample_rate_hz = 40.0

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    # Sampling ranges for each parameter
    param_ranges = {
        "source_magnitude": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],   # Δmag from truth
        "source_depth_km": [-50, -20, -5, 0, 5, 20, 50, 100, 200],      # Δdepth
        "source_latitude_deg": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],  # Δlat
        "source_longitude_deg": [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], # Δlon
    }

    print(f"Experiment 4: Candidate scoring via noise whiteness")
    print(f"  Trials: {n_trials}")
    print(f"  Candidates per param: {n_candidates_per_param} → {n_candidates_per_param**4} total candidates")
    print(f"  Windows: {windows_sec}")
    print(f"  Stations: {n_stations}")
    print()

    all_trial_results = []

    for trial in range(n_trials):
        cache_idx = int(torch.randint(0, len(ds_test), (1,)).item())
        waveform_tensor, cond_dict = ds_test[cache_idx]
        waveform = waveform_tensor.unsqueeze(0).to(device)

        # True physical parameters from condition dict
        true_params = {
            "source_magnitude": float(cond_dict["source_magnitude"].item()),
            "source_depth_km": float(cond_dict["source_depth_km"].item()),
            "source_latitude_deg": float(cond_dict["source_latitude_deg"].item()),
            "source_longitude_deg": float(cond_dict["source_longitude_deg"].item()),
        }
        # True station info
        true_sta = generate_station_info(
            true_params["source_latitude_deg"], true_params["source_longitude_deg"],
            float(cond_dict["station_latitude_deg"].item()),
            float(cond_dict["station_longitude_deg"].item()),
        )

        print(f"\n{'='*70}")
        print(f"  Trial {trial+1}/{n_trials}")
        print(f"  Truth: M={true_params['source_magnitude']:.1f}, "
              f"depth={true_params['source_depth_km']:.0f}km, "
              f"lat={true_params['source_latitude_deg']:.1f}, "
              f"lon={true_params['source_longitude_deg']:.1f}")
        print(f"  Building candidates...")

        # Build candidate pool: random combinations within range
        np.random.seed(42 + trial * 100)
        candidates = []
        for _ in range(200):  # sample 200, keep diverse set
            mg = true_params["source_magnitude"] + np.random.choice(param_ranges["source_magnitude"])
            dp = max(0.0, true_params["source_depth_km"] + np.random.choice(param_ranges["source_depth_km"]))
            lt = true_params["source_latitude_deg"] + np.random.choice(param_ranges["source_latitude_deg"])
            ln = true_params["source_longitude_deg"] + np.random.choice(param_ranges["source_longitude_deg"])
            if mg < 0.5 or mg > 9.5: continue
            if dp > 700: continue

            source_info = {
                "source_magnitude": mg,
                "source_depth_km": dp,
                "source_latitude_deg": lt,
                "source_longitude_deg": ln,
                "source_magnitude_type": "mw",
            }
            # Station adjusts position for lat/lon shift
            sta = generate_station_info(lt, ln,
                                        float(cond_dict["station_latitude_deg"].item()),
                                        float(cond_dict["station_longitude_deg"].item()))
            cond = build_candidate_encoder_input(source_info, sta)

            # Parameter error (L2 norm of z-scores)
            mag_err = (mg - true_params["source_magnitude"]) / 2.0
            depth_err = (dp - true_params["source_depth_km"]) / 100.0
            lat_err = (lt - true_params["source_latitude_deg"]) / 2.0
            lon_err = (ln - true_params["source_longitude_deg"]) / 2.0
            param_error = np.sqrt(mag_err**2 + depth_err**2 + lat_err**2 + lon_err**2)

            candidates.append({
                "source_info": source_info, "cond": cond, "param_error": param_error,
                "mag": mg, "depth": dp, "lat": lt, "lon": ln,
            })

        # Keep 50 diverse candidates
        candidates.sort(key=lambda c: c["param_error"])
        # Take: 5 closest + 15 medium + 30 farthest
        selected = candidates[:5] + candidates[75:90] + candidates[-30:]
        np.random.shuffle(selected)
        n_candidates = len(selected)
        print(f"  Scoring {n_candidates} candidates...")

        # Compute W_true for waveform L2 comparison
        cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
        true_tokens = ce(cond_gpu)

        # Generate synthetic waveform from true tokens
        seed_base = 42 + trial * 1000
        with torch.no_grad():
            torch.manual_seed(seed_base)
            if device.type == "cuda": torch.cuda.manual_seed_all(seed_base)
            noise_syn = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
            W_syn = forward_ode(dn, noise_syn, true_tokens, steps=50)

        # Score each candidate on each window
        for win_sec in windows_sec:
            win_samples = int(min(win_sec * sample_rate_hz, total_samples))

            scores = []
            t0 = time.time()

            for ci, cand in enumerate(candidates):
                cond_batch = {k: v.unsqueeze(0).to(device) for k, v in cand["cond"].items()}
                cand_tokens = ce(cond_batch)

                # Waveform L2 on observed window
                with torch.no_grad():
                    W_cand = forward_ode(dn, noise_syn, cand_tokens, steps=50)
                    obs_diff = (W_cand - W_syn)[:, :, :win_samples]
                    obs_l2 = (obs_diff ** 2).mean().sqrt().item()
                    obs_norm = (W_syn[:, :, :win_samples] ** 2).mean().sqrt().item()
                    wf_l2 = obs_l2 / max(obs_norm, 1e-8)

                # Reverse ODE noise whiteness
                z_rec = reverse_ode(dn, W_syn, cand_tokens, steps=30)
                whiteness = compute_noise_whiteness(z_rec)

                scores.append({
                    "candidate_idx": ci,
                    "param_error": cand["param_error"],
                    "mag_err": abs(cand["mag"] - true_params["source_magnitude"]),
                    "depth_err": abs(cand["depth"] - true_params["source_depth_km"]),
                    "wf_l2": float(wf_l2),
                    "whiteness": whiteness["whiteness_score"],
                    "autocorr_l1": whiteness["autocorr_l1"],
                    "kurtosis": whiteness["kurtosis"],
                })

            elapsed = time.time() - t0

            # ---- Analysis ----
            # Rank candidates by each metric
            by_whiteness = sorted(scores, key=lambda s: s["whiteness"])
            by_wf_l2 = sorted(scores, key=lambda s: s["wf_l2"])
            by_combined = sorted(scores, key=lambda s: s["wf_l2"] + s["whiteness"] * 0.5)

            # Top-1 and Top-5 by each metric
            for rank_name, ranked in [("whiteness", by_whiteness), ("wf_l2", by_wf_l2), ("combined", by_combined)]:
                top1 = ranked[0]
                top5 = ranked[:5]
                avg_param_err_top1 = top1["param_error"]
                avg_param_err_top5 = np.mean([s["param_error"] for s in top5])
                avg_mag_err_top5 = np.mean([s["mag_err"] for s in top5])
                avg_depth_err_top5 = np.mean([s["depth_err"] for s in top5])

                print(f"  win={win_sec:.0f}s {rank_name:>12s}: top1_param_err={avg_param_err_top1:.3f}  "
                      f"top5_param_err={avg_param_err_top5:.3f}  "
                      f"top5_mag_err={avg_mag_err_top5:.2f}  "
                      f"top5_depth_err={avg_depth_err_top5:.0f}km  "
                      f"({elapsed:.0f}s)")

            all_trial_results.append({
                "trial": trial + 1, "win_sec": win_sec,
                "scores": scores,
                "true_params": true_params,
            })

    # ---- Aggregate analysis ----
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT 4 — AGGREGATE SUMMARY")
    print(f"{'='*80}")

    for win_sec in windows_sec:
        print(f"\n  Window {win_sec:.0f}s:")
        for rank_name in ["whiteness", "wf_l2", "combined"]:
            trial_vals = []
            for t in all_trial_results:
                if t["win_sec"] != win_sec: continue
                scores = t["scores"]
                if rank_name == "whiteness":
                    ranked = sorted(scores, key=lambda s: s["whiteness"])
                elif rank_name == "wf_l2":
                    ranked = sorted(scores, key=lambda s: s["wf_l2"])
                else:
                    ranked = sorted(scores, key=lambda s: s["wf_l2"] + s["whiteness"] * 0.5)
                trial_vals.append(ranked[0]["param_error"])
            mean_err = np.mean(trial_vals)
            print(f"    {rank_name:>12s}: avg top-1 param_error = {mean_err:.3f}")

    return all_trial_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: Candidate scoring via whiteness")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/candidate_score_exp4")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_experiment_4(
        args.checkpoint, device,
        n_trials=args.n_trials,
        drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics only
    summary = []
    for t in results:
        entry = {
            "trial": t["trial"], "win_sec": t["win_sec"],
            "true_params": t["true_params"],
            "n_candidates": len(t["scores"]),
        }
        for rank_name, key_fn in [("whiteness", "whiteness"), ("wf_l2", "wf_l2")]:
            ranked = sorted(t["scores"], key=lambda s: s[key_fn])
            entry[f"{rank_name}_top1_param_err"] = ranked[0]["param_error"]
            entry[f"{rank_name}_top1_mag_err"] = ranked[0]["mag_err"]
            entry[f"{rank_name}_top1_depth_err"] = ranked[0]["depth_err"]
        summary.append(entry)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir}/summary.json")
