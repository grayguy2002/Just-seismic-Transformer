"""JsT seismic network simulation with physical consistency.

Generate waveforms across a synthetic station grid (e.g. 10x10 stations)
sharing the same source event. All stations use identical initial noise
to preserve coherent source-time-function randomness, with condition
differences driving the physically-varying path/receiver responses.

Key insight:
  - Same noise seed → same source-time-function randomness across stations
  - Different condition tokens → physically-varying propagation effects
  - This gives us "one earthquake, many recordings" with physical consistency
"""

from __future__ import annotations

import sys
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    JsT,
    SeismicConditionEncoder,
    ConditionSpec,
    Denoiser,
    SeismicWaveformDataset,
    collate_conditions,
    load_checkpoint_models,
)


def build_grid_network(
    n_stations_per_side: int = 10,
    grid_spacing_km: float = 50.0,
    source_lat: float = 35.0,
    source_lon: float = 139.0,
    source_depth_km: float = 10.0,
    source_magnitude: float = 5.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a regular grid of stations around a single source event.

    Parameters
    ----------
    n_stations_per_side: grid dimension (e.g. 10 → 10x10 = 100 stations)
    grid_spacing_km: spacing between adjacent stations in km
    source_lat, source_lon: source epicenter (degrees)
    source_depth_km: source depth
    source_magnitude: magnitude

    Returns
    -------
    source_info: dict with source parameters
    stations: list of 100 station dicts with lat, lon, distance, azimuth
    """
    # Convert grid to lat/lon offsets
    # 1 degree latitude ≈ 111.195 km
    deg_per_km = 1.0 / 111.195
    half = (n_stations_per_side - 1) / 2

    stations = []
    for i in range(n_stations_per_side):
        for j in range(n_stations_per_side):
            # Grid in km relative to source
            dx_km = (j - half) * grid_spacing_km
            dy_km = (i - half) * grid_spacing_km

            # Convert to lat/lon (approximate, flat-earth at these scales)
            sta_lat = source_lat + dy_km * deg_per_km
            sta_lon = source_lon + dx_km * deg_per_km / np.cos(np.deg2rad(source_lat))

            # Distance and azimuth
            dist_km = np.sqrt(dx_km**2 + dy_km**2)
            dist_deg = dist_km / 111.195
            azimuth = np.rad2deg(np.arctan2(dx_km, dy_km)) % 360
            back_azimuth = (azimuth + 180) % 360

            stations.append({
                "station_latitude_deg": sta_lat,
                "station_longitude_deg": sta_lon,
                "station_elevation_m": 0.0,
                "path_ep_distance_deg": dist_deg,
                "path_ep_distance_km": dist_km,
                "path_azimuth_deg": azimuth,
                "path_back_azimuth_deg": back_azimuth,
                "grid_i": i,
                "grid_j": j,
            })

    source_info = {
        "source_latitude_deg": source_lat,
        "source_longitude_deg": source_lon,
        "source_depth_km": source_depth_km,
        "source_magnitude": source_magnitude,
        "source_magnitude_type": "mw",
        "n_stations": len(stations),
        "grid_spacing_km": grid_spacing_km,
    }

    return source_info, stations


def build_condition_dict(
    source_info: dict,
    station: dict,
    phase: str = "P",
    channel: str = "BH",
    network: str = "XX",
    sample_rate_hz: float = 40.0,
    window_padding_sec: float = 20.0,
) -> dict[str, torch.Tensor]:
    """Build a single condition dict for one station.

    Sets selected_phase_arrival_sample to match distance-dependent travel
    time, so the model places the P-wave energy at the correct time.
    """
    dist_km = station["path_ep_distance_km"]
    depth_km = source_info["source_depth_km"]

    # Approximate P-wave travel time
    tt_sec = dist_km / 8.0 + depth_km / 6.0
    # Arrival sample: padding + travel time, converted to samples
    arrival_sample = (window_padding_sec + tt_sec) * sample_rate_hz

    cond = {
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
        "selected_phase": torch.tensor(0, dtype=torch.long),
        "trace_channel": torch.tensor(0, dtype=torch.long),
        "station_network_code": torch.tensor(0, dtype=torch.long),
        # KEY: tell the model WHERE to place the P-wave
        "selected_phase_arrival_sample": torch.tensor(arrival_sample, dtype=torch.float32),
        "selected_phase_arrival_sample_present": torch.tensor(1.0, dtype=torch.float32),
    }

    return cond


@torch.no_grad()
def generate_network(
    denoiser: Denoiser,
    cond_encoder: SeismicConditionEncoder,
    source_info: dict,
    stations: list[dict],
    *,
    steps: int = 50,
    shared_noise_seed: int = 42,
    batch_size: int = 25,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    use_amp: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate waveforms for all stations with shared source-time-function.

    Parameters
    ----------
    denoiser: trained Denoiser
    cond_encoder: trained SeismicConditionEncoder
    source_info: dict from build_grid_network()
    stations: list of station dicts from build_grid_network()
    steps: ODE integration steps
    shared_noise_seed: fixed seed for all stations (ensures shared source)
    batch_size: stations per GPU batch
    verbose: print progress

    Returns
    -------
    waveforms: (n_stations, 3, T) float32 numpy array
    metadata: dict with station positions, distances, etc as numpy arrays
    """
    if device is None:
        device = next(denoiser.parameters()).device

    n_stations = len(stations)
    C = denoiser.net.in_channels
    T = denoiser.net.n_samples
    noise_scale = denoiser.noise_scale

    # Build all condition dicts
    all_conds = [build_condition_dict(source_info, s) for s in stations]

    all_waveforms = []

    n_batches = (n_stations + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_stations)
        n_batch = end - start

        if verbose:
            print(f"  batch {batch_idx + 1}/{n_batches}: stations {start}-{end-1}")

        # Collate conditions
        batch_conds = all_conds[start:end]
        # Create dummy waveforms for collation
        dummy_waves = [torch.zeros(3, T) for _ in range(n_batch)]
        _, cond_batch = collate_conditions(list(zip(dummy_waves, batch_conds)))
        cond_batch = {k: v.to(device, non_blocking=True) for k, v in cond_batch.items()}

        # Encode conditions
        ct = cond_encoder(cond_batch)

        # ---- KEY: Shared noise for all stations ----
        # Shared seed → same source-time-function across stations
        # Independent seed → each station gets its own source realization
        if shared_noise_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(shared_noise_seed)
            noise = noise_scale * torch.randn(
                1, C, T, device=device, dtype=dtype, generator=generator
            ).expand(n_batch, -1, -1)
        else:
            noise = noise_scale * torch.randn(
                n_batch, C, T, device=device, dtype=dtype
            )

        # ---- Generate ----
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu",
                                dtype=torch.bfloat16) if use_amp else torch.enable_grad():
            ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
            stepper = denoiser._heun_step if denoiser.method == "heun" else denoiser._euler_step

            z = noise
            for i in range(steps - 1):
                z = stepper(z, ts[i], ts[i + 1], ct)
            z = denoiser._euler_step(z, ts[-2], ts[-1], ct)

        all_waveforms.append(z.cpu().float().numpy())

    # Concatenate results
    waveforms = np.concatenate(all_waveforms, axis=0)  # (n_stations, 3, T)

    # Build metadata
    metadata = {
        "station_lat": np.array([s["station_latitude_deg"] for s in stations]),
        "station_lon": np.array([s["station_longitude_deg"] for s in stations]),
        "distance_km": np.array([s["path_ep_distance_km"] for s in stations]),
        "azimuth_deg": np.array([s["path_azimuth_deg"] for s in stations]),
        "back_azimuth_deg": np.array([s["path_back_azimuth_deg"] for s in stations]),
        "grid_i": np.array([s["grid_i"] for s in stations]),
        "grid_j": np.array([s["grid_j"] for s in stations]),
        "source_magnitude": source_info["source_magnitude"],
        "source_depth_km": source_info["source_depth_km"],
        "shared_noise_seed": shared_noise_seed,
        "n_stations": n_stations,
        "ode_steps": steps,
    }

    return waveforms, metadata


def compute_physical_consistency(
    waveforms: np.ndarray,
    metadata: dict[str, np.ndarray],
    sample_rate_hz: float = 40.0,
    p_wave_velocity_kms: float = 8.0,
) -> dict[str, Any]:
    """Evaluate physical consistency of generated waveforms across the network.

    Parameters
    ----------
    waveforms: (n_stations, 3, T) float32
    metadata: from generate_network()

    Returns
    -------
    dict with:
        - travel_time_correlation: Pearson r between distance and first-arrival time
        - peak_amplitude_distance_corr: r between log(peak) and log(distance)
        - amplitude_decay_exponent: fitted geometric spreading exponent
        - waveform_similarity_vs_distance: correlation between station-pair
          waveform similarity and station-pair distance
        - channel_coherence: cross-channel correlation matrix
        - per_station: list of per-station metrics
    """
    from scipy.signal import find_peaks
    from scipy.stats import pearsonr

    n_stations = waveforms.shape[0]
    n_channels = waveforms.shape[1]
    n_samples = waveforms.shape[2]
    distances = metadata["distance_km"]
    time_axis = np.arange(n_samples) / sample_rate_hz

    # 1. Per-station metrics
    first_arrivals = []
    peak_amplitudes = []
    snr_values = []

    for i in range(n_stations):
        w = waveforms[i, 0, :]  # vertical component

        # Energy-based arrival detection (more robust for synthetic data)
        # Find where cumulative energy crosses a threshold of total energy
        energy = w ** 2
        cumulative = np.cumsum(energy)
        total = cumulative[-1]
        if total > 1e-12:
            # First arrival = where cumulative energy crosses 0.5% of total
            arrival_idx = np.searchsorted(cumulative, 0.005 * total)
        else:
            arrival_idx = 0

        # Also try peak-derivative picker: max acceleration
        accel = np.abs(np.diff(w, 2))
        accel_idx = np.argmax(accel[:int(0.6 * n_samples)]) if len(accel) > 0 else 0

        # Use minimum of the two
        first_arrival = min(arrival_idx, accel_idx + 2)

        # Peak amplitude in P-wave window after arrival
        p_window_end = min(first_arrival + int(10.0 * sample_rate_hz), n_samples)
        p_window = slice(first_arrival, p_window_end)
        peak = np.max(np.abs(w[p_window])) if p_window_end > first_arrival + 1 else np.max(np.abs(w))

        # SNR: signal / pre-arrival noise
        pre_window = slice(max(0, first_arrival - int(2*sample_rate_hz)), max(1, first_arrival))
        pre_noise = np.mean(w[pre_window] ** 2) if pre_window.start < pre_window.stop else 1e-12
        signal_power = np.mean(w[p_window] ** 2) if p_window_end > first_arrival + 1 else 1e-12
        snr = 10 * np.log10(max(signal_power / pre_noise, 1e-12))

        first_arrivals.append(first_arrival)
        peak_amplitudes.append(peak)
        snr_values.append(snr)

    first_arrivals = np.array(first_arrivals)
    peak_amplitudes = np.array(peak_amplitudes)
    snr_values = np.array(snr_values)

    # 2. Travel time correlation with distance
    valid = (first_arrivals > 0) & np.isfinite(distances)
    if valid.sum() >= 5:
        tt_corr, tt_pval = pearsonr(first_arrivals[valid], distances[valid])
    else:
        tt_corr, tt_pval = 0.0, 1.0

    # 3. Amplitude decay: log(peak) ~ b * log(distance) + c
    valid = (peak_amplitudes > 1e-8) & (distances > 1.0) & np.isfinite(distances)
    if valid.sum() >= 5:
        log_peak = np.log10(peak_amplitudes[valid])
        log_dist = np.log10(distances[valid])
        # Linear regression
        X = np.column_stack([np.ones_like(log_dist), log_dist])
        coeffs = np.linalg.lstsq(X, log_peak, rcond=None)[0]
        decay_exponent = float(coeffs[1])
        amp_corr, amp_pval = pearsonr(log_peak, log_dist)
    else:
        decay_exponent = 0.0
        amp_corr = 0.0

    # 4. Waveform similarity vs distance
    # Compute pairwise correlation for a subset of stations
    n_pairs = min(50, n_stations)
    pair_dists = []
    pair_corrs = []
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            # Correlation between aligned waveforms (channel 0)
            wi = waveforms[i, 0, :]
            wj = waveforms[j, 0, :]
            corr = np.corrcoef(wi, wj)[0, 1] if np.std(wi) > 0 and np.std(wj) > 0 else 0
            pair_corrs.append(corr)
            pair_dists.append(np.abs(distances[i] - distances[j]))

    pair_dists = np.array(pair_dists)
    pair_corrs = np.array(pair_corrs)
    if len(pair_corrs) >= 3:
        sim_corr, _ = pearsonr(pair_dists, pair_corrs)
    else:
        sim_corr = 0.0

    # 5. Channel coherence
    # All channels should show similar patterns
    ch_corrs = []
    for i in range(min(20, n_stations)):
        w = waveforms[i]  # (3, T)
        c01 = np.corrcoef(w[0], w[1])[0, 1]
        c02 = np.corrcoef(w[0], w[2])[0, 1]
        c12 = np.corrcoef(w[1], w[2])[0, 1]
        ch_corrs.append([c01, c02, c12])
    ch_corrs = np.array(ch_corrs)

    # 6. Travel time fit: arrival_time = intercept + distance / velocity
    valid = (first_arrivals > 0) & (distances > 0.1) & np.isfinite(distances)
    if valid.sum() >= 5:
        t_arr = first_arrivals[valid] / sample_rate_hz
        X = np.column_stack([np.ones_like(distances[valid]), distances[valid]])
        coeffs = np.linalg.lstsq(X, t_arr, rcond=None)[0]
        fitted_intercept = float(coeffs[0])
        fitted_slowness = float(coeffs[1])  # s/km
        fitted_velocity = 1.0 / fitted_slowness if abs(fitted_slowness) > 1e-8 else 0.0
        tt_r2 = 1.0 - np.sum((t_arr - X @ coeffs) ** 2) / np.sum((t_arr - np.mean(t_arr)) ** 2)
    else:
        fitted_velocity = 0.0
        tt_r2 = 0.0

    return {
        # Travel time
        "travel_time_distance_corr": float(tt_corr),
        "fitted_p_velocity_kms": float(fitted_velocity),
        "expected_p_velocity_kms": p_wave_velocity_kms,
        "velocity_error_pct": float(
            100 * abs(fitted_velocity - p_wave_velocity_kms) / p_wave_velocity_kms
        ) if fitted_velocity > 0 else float("inf"),
        "travel_time_r2": float(tt_r2),
        # Amplitude
        "amplitude_distance_corr": float(amp_corr),
        "amplitude_decay_exponent": decay_exponent,
        "expected_decay_exponent": -1.0,  # geometric spreading in 3D
        # Waveform similarity
        "pairwise_similarity_distance_corr": float(sim_corr),
        "mean_pairwise_corr": float(np.mean(pair_corrs)) if len(pair_corrs) > 0 else 0.0,
        # Channel
        "mean_channel_corr": float(np.mean(ch_corrs)),
        "channel_corr_std": float(np.std(ch_corrs)),
        # Per-station
        "first_arrival_samples": first_arrivals.tolist(),
        "peak_amplitudes": peak_amplitudes.tolist(),
        "snr_db": snr_values.tolist(),
        # Summary
        "n_valid_stations": int(valid.sum()),
        "grid_positions": [(int(s["grid_i"]), int(s["grid_j"]))
                          for s in [{"grid_i": i, "grid_j": j}
                                   for i, j in zip(metadata["grid_i"], metadata["grid_j"])]],
    }


def save_results(
    waveforms: np.ndarray,
    metadata: dict,
    consistency: dict,
    output_dir: str | Path,
    prefix: str = "network_sim",
):
    """Save generated waveforms and metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save waveforms (potentially large - use compressed format)
    np.savez_compressed(
        output_dir / f"{prefix}_waveforms.npz",
        waveforms=waveforms,
        **{k: v for k, v in metadata.items() if isinstance(v, (np.ndarray, np.generic))},
    )

    # Save metadata and consistency results as JSON
    import json
    serializable = {}
    for k, v in metadata.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool)):
            serializable[k] = v
        else:
            serializable[k] = str(v)

    with open(output_dir / f"{prefix}_results.json", "w") as f:
        json.dump({
            "metadata": serializable,
            "consistency": {k: v for k, v in consistency.items()
                           if not isinstance(v, (np.ndarray, list)) or len(str(v)) < 500},
        }, f, indent=2)

    # Full consistency dump
    import json as _json
    class NumpyEncoder(_json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_dir / f"{prefix}_consistency.json", "w") as f:
        _json.dump(consistency, f, indent=2, cls=NumpyEncoder)

    print(f"Saved to {output_dir}/")


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    use_ema: bool = True,
    drop_tokens: list[int] | None = None,
) -> tuple[SeismicConditionEncoder, Denoiser, dict[str, Any]]:
    """Load trained JsT model with optional token ablation."""
    from JsT.ablation import AblationConditionEncoder

    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=use_ema,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )

    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)

    return ce, dn, ckpt


# ============================================================================
# Main: run network simulation and evaluate
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JsT seismic network simulation")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pth")
    parser.add_argument("--output-dir", default="outputs/network_sim")
    parser.add_argument("--grid-size", type=int, default=10, help="Stations per side (10 = 100 stations)")
    parser.add_argument("--grid-spacing-km", type=float, default=50.0)
    parser.add_argument("--source-magnitude", type=float, default=5.5)
    parser.add_argument("--source-depth-km", type=float, default=10.0)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--shared-noise-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10",
                        help="Comma-separated token indices to zero out")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix", default="network_sim")
    parser.add_argument("--compare-independent", action="store_true",
                        help="Also generate with independent noise seeds for comparison")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse dropped tokens
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []
    if dropped:
        print(f"Dropping tokens: {dropped}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ce, dn, ckpt = load_model(args.checkpoint, device, use_ema=args.use_ema, drop_tokens=dropped)
    print(f"Model loaded. Encoder version: {getattr(ce, 'encoder_version', 'v3')}")

    # Build grid
    source_info, stations = build_grid_network(
        n_stations_per_side=args.grid_size,
        grid_spacing_km=args.grid_spacing_km,
        source_magnitude=args.source_magnitude,
        source_depth_km=args.source_depth_km,
    )
    print(f"Grid: {args.grid_size}x{args.grid_size} = {len(stations)} stations")
    print(f"Source: M{source_info['source_magnitude']}, depth={source_info['source_depth_km']}km")

    # ---- Shared-noise generation ----
    print("\n=== Shared-noise generation ===")
    t0 = time.time()
    waveforms, metadata = generate_network(
        dn, ce, source_info, stations,
        steps=args.ode_steps,
        shared_noise_seed=args.shared_noise_seed,
        batch_size=args.batch_size,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"Generated {len(stations)} waveforms in {elapsed:.1f}s")

    # Evaluate physical consistency
    print("\n=== Physical consistency evaluation ===")
    consistency = compute_physical_consistency(waveforms, metadata)

    # Print key metrics
    print(f"\n{'Metric':<45s} {'Value':>15s} {'Expected':>15s}")
    print("-" * 75)
    print(f"{'Travel time vs distance corr':<45s} {consistency['travel_time_distance_corr']:15.4f} {'>0.7':>15s}")
    print(f"{'Fitted P-wave velocity (km/s)':<45s} {consistency['fitted_p_velocity_kms']:15.2f} {consistency['expected_p_velocity_kms']:15.1f}")
    print(f"{'Velocity error (%)':<45s} {consistency['velocity_error_pct']:15.1f} {'<20':>15s}")
    print(f"{'Travel time R²':<45s} {consistency['travel_time_r2']:15.3f} {'>0.5':>15s}")
    print(f"{'Amplitude vs distance corr':<45s} {consistency['amplitude_distance_corr']:15.4f} {'< -0.5':>15s}")
    print(f"{'Amplitude decay exponent':<45s} {consistency['amplitude_decay_exponent']:15.3f} {consistency['expected_decay_exponent']:15.1f}")
    print(f"{'Pairwise similarity vs dist corr':<45s} {consistency['pairwise_similarity_distance_corr']:15.4f} {'< -0.2':>15s}")
    print(f"{'Mean cross-channel correlation':<45s} {consistency['mean_channel_corr']:15.3f} {'>0.3':>15s}")
    print(f"{'Mean SNR (dB)':<45s} {np.mean(consistency['snr_db']):15.1f} {'>5':>15s}")
    print(f"{'Valid stations':<45s} {consistency['n_valid_stations']:15d}")

    # Save
    save_results(waveforms, metadata, consistency, args.output_dir, args.prefix)

    # ---- Optional: Independent-noise comparison ----
    if args.compare_independent:
        print("\n=== Independent-noise generation (comparison) ===")
        t0 = time.time()
        waveforms_ind, metadata_ind = generate_network(
            dn, ce, source_info, stations,
            steps=args.ode_steps,
            shared_noise_seed=None,  # None → each station gets independent noise
            batch_size=args.batch_size,
            device=device,
        )
        elapsed = time.time() - t0

        consistency_ind = compute_physical_consistency(waveforms_ind, metadata_ind)

        print(f"\n{'Metric':<45s} {'Shared Noise':>15s} {'Independent':>15s}")
        print("-" * 75)
        for key, label in [
            ("travel_time_distance_corr", "TT vs distance corr"),
            ("amplitude_distance_corr", "Amp vs distance corr"),
            ("pairwise_similarity_distance_corr", "Similarity vs dist corr"),
            ("mean_channel_corr", "Cross-channel corr"),
        ]:
            print(f"{label:<45s} {consistency[key]:15.4f} {consistency_ind[key]:15.4f}")

        save_results(waveforms_ind, metadata_ind, consistency_ind, args.output_dir, f"{args.prefix}_independent")

    print("\nDone.")
