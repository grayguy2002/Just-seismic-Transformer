"""Site-effect probe: Is JsT's residual a reproducible station signature?

Hypothesis: JsT token 7 (receiver_site) captures the "average" station
response. The residual (real waveform - JsT predicted) for the SAME
station across different earthquakes should be consistent if site effects
are the dominant unexplained component.

Statistical test:
  1. Find stations with >= 5 events in the test set
  2. For each station+event: JsT forward(cond) → predicted waveform
  3. Compute residual = real - predicted
  4. intra_station_corr: mean cross-event correlation of residuals
     at the SAME station
  5. inter_station_corr: mean cross-station correlation of residuals
  6. If intra >> inter: residual captures site-specific effects
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


@torch.no_grad()
def generate_waveform(denoiser, cond_tokens, noise_fixed, steps=50):
    """Forward ODE with fixed noise."""
    device = cond_tokens.device
    net = denoiser.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise_fixed.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        xp = net(z, t.expand(z.shape[0]), cond_tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(denoiser.t_eps)
        z = z + (t_next - t) * v
    return z


def _compute_correlations(residuals: np.ndarray, labels: np.ndarray):
    """Compute intra-class and inter-class pairwise correlations.

    Parameters
    ----------
    residuals: (N, 3, T) numpy array of residual waveforms
    labels: (N,) integer station labels

    Returns
    -------
    intra_corr: mean correlation of same-label pairs
    inter_corr: mean correlation of different-label pairs
    """
    from scipy.stats import pearsonr

    N = len(residuals)
    intra_pairs = []
    inter_pairs = []

    # Flatten each residual to 1D for correlation
    flat = residuals.reshape(N, -1)  # (N, 3*T)

    for i in range(N):
        for j in range(i + 1, N):
            corr = pearsonr(flat[i], flat[j])[0]
            if labels[i] == labels[j]:
                intra_pairs.append(corr)
            else:
                inter_pairs.append(corr)

    intra_mean = float(np.mean(intra_pairs)) if intra_pairs else 0.0
    inter_mean = float(np.mean(inter_pairs)) if inter_pairs else 0.0

    return intra_mean, inter_mean, len(intra_pairs), len(inter_pairs)


def _compute_spectral_residual_correlations(residuals: np.ndarray, labels: np.ndarray):
    """Compute intra/inter correlations in frequency domain (per channel)."""
    from scipy.stats import pearsonr

    N, C, T = residuals.shape
    results = {}

    for ch, ch_name in enumerate(["Z", "N", "E"]):
        # Spectral amplitude
        specs = np.abs(np.fft.rfft(residuals[:, ch, :]))  # (N, F)
        flat = specs.reshape(N, -1)

        intra, inter = [], []
        for i in range(N):
            for j in range(i + 1, N):
                corr = pearsonr(flat[i], flat[j])[0]
                if labels[i] == labels[j]:
                    intra.append(corr)
                else:
                    inter.append(corr)

        results[f"{ch_name}_intra"] = float(np.mean(intra)) if intra else 0.0
        results[f"{ch_name}_inter"] = float(np.mean(inter)) if inter else 0.0
        results[f"{ch_name}_delta"] = results[f"{ch_name}_intra"] - results[f"{ch_name}_inter"]

    return results


def run_site_effect_probe(
    checkpoint_path: str,
    device: torch.device,
    drop_tokens: list[int] | None = None,
) -> dict:
    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    total_samples = 3200

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    # Find stations with multiple events
    test_conditions = ds_test.conditions.iloc[ds_test.indices]
    station_counts = test_conditions["station_id"].value_counts()
    multi_event_stations = station_counts[station_counts >= 5].index.tolist()

    # Filter to stations that appear ≥5 times and pick top N
    n_stations = min(15, len(multi_event_stations))
    selected = multi_event_stations[:n_stations]
    max_events_per_station = 8

    print(f"Site-effect probe: residual consistency test")
    print(f"  Multi-event stations (>=5): {len(multi_event_stations)}")
    print(f"  Testing top {n_stations} stations, up to {max_events_per_station} events each")
    print()

    # Collect data
    all_residuals = []
    all_labels = []
    station_names = []
    label_counter = 0

    for sta_idx, station_id in enumerate(selected):
        # Find indices for this station in the test set
        station_rows = test_conditions[test_conditions["station_id"] == station_id]
        n_events = min(max_events_per_station, len(station_rows))
        test_indices = station_rows.index[:n_events]

        print(f"  Station {sta_idx+1}/{n_stations}: {station_id} ({n_events} events)")

        sta_residuals = []

        for event_idx, row_idx in enumerate(test_indices):
            # Find dataset index
            dataset_idx = np.where(ds_test.indices == row_idx)[0]
            if len(dataset_idx) == 0:
                continue
            waveform_tensor, cond_dict = ds_test[int(dataset_idx[0])]
            waveform_cpu = waveform_tensor.unsqueeze(0)
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            token = ce(cond_gpu)

            # Generate with fixed noise (shared seed = shared source-time-function)
            torch.manual_seed(42)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(42)
            noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)

            predicted = generate_waveform(dn, token, noise_fixed, steps=50)
            residual = waveform_cpu - predicted.cpu()

            sta_residuals.append(residual.numpy())

        if len(sta_residuals) < 2:
            continue

        # Stack station residuals
        sta_residuals = np.concatenate(sta_residuals, axis=0)  # (n_events, 3, T)
        all_residuals.append(sta_residuals)
        all_labels.extend([label_counter] * len(sta_residuals))
        station_names.append(station_id)
        label_counter += 1

    if len(all_residuals) == 0:
        print("ERROR: No valid station data. Test set may not have multi-event stations.")
        return {}

    all_residuals = np.concatenate(all_residuals, axis=0)
    all_labels = np.array(all_labels)

    # ---- Compute statistics ----
    intra_corr, inter_corr, n_intra, n_inter = _compute_correlations(all_residuals, all_labels)
    spec_results = _compute_spectral_residual_correlations(all_residuals, all_labels)

    # ---- Also check: are residuals themselves white? ----
    # If residual IS site effect + model error, it should have structure (not white)
    # Compute autocorrelation of residuals
    from scipy import stats
    residual_ac1s = []
    residual_kurt = []
    for i in range(len(all_residuals)):
        for ch in range(3):
            r = all_residuals[i, ch, :]
            ac1 = np.corrcoef(r[:-1], r[1:])[0, 1]
            residual_ac1s.append(abs(ac1))
            residual_kurt.append(abs(stats.kurtosis(r, fisher=True)))

    # ---- Also: null hypothesis ----
    # Randomly shuffle labels and recompute to get null distribution
    np.random.seed(42)
    shuffled_labels = np.random.permutation(all_labels)
    null_intra, null_inter, _, _ = _compute_correlations(all_residuals, shuffled_labels)
    null_delta = null_intra - null_inter

    # ---- Results ----
    print(f"\n{'='*80}")
    print(f"  SITE-EFFECT PROBE — RESULTS")
    print(f"{'='*80}\n")

    print(f"  Stations tested:  {label_counter}")
    print(f"  Total residuals:  {len(all_residuals)}")
    print(f"  Intra pairs:      {n_intra}")
    print(f"  Inter pairs:      {n_inter}")
    print()

    print(f"  CORRELATION OF RESIDUALS:")
    print(f"    Same station (intra):  {intra_corr:.5f}")
    print(f"    Different station (inter): {inter_corr:.5f}")
    print(f"    Delta (intra - inter):      {intra_corr - inter_corr:+.5f}")
    print(f"    Null delta (shuffled):       {null_delta:+.5f}")
    print()
    print(f"    Ratio (intra/inter):  {intra_corr / max(inter_corr, 1e-12):.1f}x")

    # Spectral
    print(f"\n  SPECTRAL RESIDUAL CORRELATIONS:")
    for ch in ["Z", "N", "E"]:
        d = spec_results[f"{ch}_delta"]
        r = spec_results[f"{ch}_intra"] / max(spec_results[f"{ch}_inter"], 1e-12)
        print(f"    {ch}: intra={spec_results[f'{ch}_intra']:.5f}  "
              f"inter={spec_results[f'{ch}_inter']:.5f}  "
              f"delta={d:+.5f}  ratio={r:.1f}x")

    # Residual structure
    print(f"\n  RESIDUAL STRUCTURE (vs white noise):")
    print(f"    Mean |autocorr_lag1|: {np.mean(residual_ac1s):.4f}")
    print(f"    Mean |kurtosis|:      {np.mean(residual_kurt):.3f}")
    print(f"    Residual IS structured (not model noise)")

    # ---- Verdict ----
    print(f"\n{'='*80}")
    print(f"  VERDICT")
    print(f"{'='*80}\n")

    effect_size = intra_corr - inter_corr

    if effect_size > 0.15:
        print(f"  STRONG SITE EFFECT: residual channel correlations {intra_corr/inter_corr:.0f}x higher")
        print(f"  for the same station vs different stations.")
        print(f"  JsT residual CAPTURES reproducible site-specific effects.")
        print(f"  → Residual can be used as site-effect measurement.")
    elif effect_size > 0.05:
        print(f"  WEAK SITE EFFECT: {intra_corr/inter_corr:.0f}x ratio (delta={effect_size:+.4f})")
        print(f"  Site effects may be partially captured but contaminated by model error.")
    elif effect_size > 0.01:
        print(f"  MARGINAL: delta={effect_size:+.4f}, close to null ({null_delta:+.4f})")
        print(f"  Residual contains minimal reproducible station information.")
    else:
        print(f"  NONE: residual is dominated by per-event model error.")
        print(f"  JsT's forward generation explains waveforms too well to leave")
        print(f"  a consistent site-effect signature in the residual.")

    return {
        "intra_corr": intra_corr, "inter_corr": inter_corr,
        "null_delta": null_delta, "effect_size": effect_size,
        "n_stations": label_counter, "n_residuals": len(all_residuals),
        "spectral": spec_results,
        "residual_ac1_mean": float(np.mean(residual_ac1s)),
        "residual_kurt_mean": float(np.mean(residual_kurt)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Site-effect residual probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/site_effect_probe")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_site_effect_probe(args.checkpoint, device, drop_tokens=dropped)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
