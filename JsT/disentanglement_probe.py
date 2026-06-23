"""Physical disentanglement probe: does JsT learn separable physics?

For each source token (0-2), we encode a real condition, then perturb
ONE token in isolation and regenerate with the SAME noise seed.
If the token space is disentangled:
  - token 0 (source_size): changes amplitude, preserves arrival time
  - token 1 (source_location): changes arrival/distance characteristics
  - token 2 (radiation_proxy): changes radiation pattern / first motion

Key signature of disentanglement: varying token[i] changes the waveform
in ways that are physically interpretable for that token's semantics,
while NOT changing features controlled by tokens j≠i.
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, ConditionSpec, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder

TOKEN_NAMES_V3 = [
    "source_size", "source_location_depth", "source_radiation_proxy",
    "path_geometry", "path_travel_time", "selected_phase_label",
    "path_region_proxy", "receiver_site",
    "station_identity", "instrument", "receiver_orientation",
]

def _cosine(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    an = a / (np.linalg.norm(a) + 1e-12)
    bn = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(an, bn))


def _peak_amplitude(w):
    """Peak absolute amplitude (per channel, then mean)."""
    return float(np.mean(np.max(np.abs(w), axis=-1)))


def _first_arrival_sample(w, sr=40.0):
    """Cumulative energy threshold picker."""
    energy = np.cumsum(w ** 2)
    total = energy[-1]
    if total < 1e-12:
        return 0
    return int(np.searchsorted(energy, 0.01 * total))


def _spectral_centroid(w, sr=40.0):
    """Weighted mean frequency."""
    spec = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), d=1.0/sr)
    if spec.sum() < 1e-12:
        return 0.0
    return float(np.sum(freqs * spec) / spec.sum())


def _channel_correlation(w):
    """Mean cross-channel correlation."""
    c01 = np.corrcoef(w[0], w[1])[0,1]
    c02 = np.corrcoef(w[0], w[2])[0,1]
    c12 = np.corrcoef(w[1], w[2])[0,1]
    return float(np.mean([c01, c02, c12]))


def run_disentanglement_probe(
    checkpoint_path: str,
    device: torch.device,
    n_samples: int = 5,
    perturbation_scales: list[float] | None = None,
    token_indices: list[int] | None = None,
    drop_tokens: list[int] | None = None,
) -> dict:
    """Probe whether JsT tokens are disentangled.

    For each test sample and each token:
      - Encode true condition tokens
      - Perturb token[i] by ±scale * std(token_i)
      - Regenerate with SAME noise seed
      - Measure: does the waveform change in physically interpretable ways?
    """
    if perturbation_scales is None:
        perturbation_scales = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    if token_indices is None:
        token_indices = [0, 1, 2]  # source tokens

    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    net = dn.net
    n_tokens = net.n_cond_tokens
    total_samples = 3200
    sr = 40.0

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    # Pre-compute per-token std over a subset to get meaningful perturbation scales
    print("Computing token statistics...")
    token_samples = []
    np.random.seed(42)
    sample_indices = np.random.choice(len(ds_test), min(100, len(ds_test)), replace=False)
    for idx in sample_indices:
        _, cond_dict = ds_test[int(idx)]
        cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
        token_samples.append(ce(cond_gpu).cpu())
    all_tokens = torch.cat(token_samples, dim=0)  # (N, n_tokens, hidden)
    token_std = all_tokens.std(dim=0)  # (n_tokens, hidden) — per-token, per-dim std

    print(f"Token perturbation std mean per token:")
    for ti in range(min(len(token_indices) + 5, n_tokens)):
        name = TOKEN_NAMES_V3[ti] if ti < len(TOKEN_NAMES_V3) else f"token_{ti}"
        print(f"  {ti:2d} {name:<30s}: std={token_std[ti].mean().item():.4f}")

    all_results = []

    for sample_idx in range(n_samples):
        cache_idx = int(torch.randint(0, len(ds_test), (1,)).item())
        waveform, cond_dict = ds_test[cache_idx]
        cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
        true_tokens = ce(cond_gpu)

        # Fixed noise seed for this sample (ensures same source-time-function)
        seed_base = 42 + sample_idx * 1000
        noise_fixed = None  # lazy init

        print(f"\n{'='*70}")
        print(f"  Sample {sample_idx+1}/{n_samples}: M={cond_dict['source_magnitude'].item():.1f}  "
              f"depth={cond_dict['source_depth_km'].item():.0f}km  "
              f"dist={cond_dict['path_ep_distance_deg'].item():.1f}deg")
        print(f"{'='*70}")

        sample_results = {"sample": sample_idx + 1, "cache_idx": cache_idx,
                          "tokens": {}, "metrics": {}}

        for ti in token_indices:
            token_name = TOKEN_NAMES_V3[ti] if ti < len(TOKEN_NAMES_V3) else f"token_{ti}"
            print(f"\n  Token {ti}: {token_name}")

            waveforms = {}   # scale -> (3, T) numpy
            metrics = {}     # scale -> {peak, arrival, centroid, ch_corr, cos_to_base}
            base_w = None

            # Process scale=0 first so base_w is always defined
            ordered_scales = sorted(perturbation_scales, key=abs)

            for scale in ordered_scales:
                # Perturb token ti by scale * std
                perturbed = true_tokens.clone()
                perturbed[0, ti, :] += scale * token_std[ti].to(device)

                # Generate with fixed noise
                if noise_fixed is None:
                    torch.manual_seed(seed_base)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed_base)
                    noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)

                # Forward ODE
                with torch.no_grad():
                    ts = torch.linspace(0.0, 1.0, 51, device=device)
                    z = noise_fixed.clone()
                    for i in range(50):
                        t = ts[i]; t_next = ts[i + 1]
                        xp = net(z, t.expand(1), perturbed)
                        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
                        z = z + (t_next - t) * v

                w = z.cpu().float().numpy()[0]  # (3, T)
                waveforms[scale] = w

                if scale == 0.0:
                    base_w = w

                # Compute physical metrics
                peak = _peak_amplitude(w)
                arrival = _first_arrival_sample(w[0])  # Z channel
                centroid = _spectral_centroid(w[0])
                ch_corr = _channel_correlation(w)
                cos_sim = _cosine(w, base_w) if scale != 0.0 else 1.0

                metrics[scale] = {
                    "peak": float(peak), "arrival_sample": arrival,
                    "spectral_centroid": float(centroid),
                    "channel_corr": float(ch_corr), "cosine_to_base": cos_sim,
                }

                if abs(scale) >= 2.0 or scale == 0.0:
                    print(f"    scale={scale:+4.1f}: peak={peak:.4f}  arrival={arrival}  "
                          f"centroid={centroid:.1f}Hz  ch_corr={ch_corr:.3f}  cos={cos_sim:.3f}")

            sample_results["tokens"][f"token_{ti}"] = {
                "name": token_name,
                "waveforms": {str(s): w.tolist() for s, w in waveforms.items()},
                "metrics": {str(s): m for s, m in metrics.items()},
            }

        all_results.append(sample_results)

    # ---- Aggregate analysis ----
    print(f"\n{'='*90}")
    print(f"  DISENTANGLEMENT ANALYSIS — AGGREGATE ({n_samples} samples)")
    print(f"{'='*90}\n")

    # For each token, compute sensitivity metrics
    for ti in token_indices:
        token_name = TOKEN_NAMES_V3[ti] if ti < len(TOKEN_NAMES_V3) else f"token_{ti}"
        peak_slopes = []; arrival_slopes = []; centroid_slopes = []; cos_extremes = []

        for r in all_results:
            td = r["tokens"].get(f"token_{ti}", {})
            mets = td.get("metrics", {})
            if not mets: continue
            scales = sorted([float(s) for s in mets.keys()])
            if len(scales) < 3: continue
            peaks = [mets[str(s)]["peak"] for s in scales]
            arrivals = [mets[str(s)]["arrival_sample"] for s in scales]
            centroids = [mets[str(s)]["spectral_centroid"] for s in scales]
            peak_slopes.append(abs(np.polyfit(scales, peaks, 1)[0]))
            arrival_slopes.append(abs(np.polyfit(scales, arrivals, 1)[0]))
            centroid_slopes.append(abs(np.polyfit(scales, centroids, 1)[0]))
            ext_scales = [s for s in scales if abs(s) >= 2.0]
            if ext_scales: cos_extremes.append(np.mean([mets[str(s)]["cosine_to_base"] for s in ext_scales]))

        print(f"\n  Token {ti}: {token_name}")
        print(f"    amplitude sensitivity:     {np.mean(peak_slopes):.4f} ±{np.std(peak_slopes):.4f}")
        print(f"    arrival time sensitivity:  {np.mean(arrival_slopes):.3f} ±{np.std(arrival_slopes):.3f}")
        print(f"    spectral sensitivity:      {np.mean(centroid_slopes):.3f} ±{np.std(centroid_slopes):.3f}")
        print(f"    shape preservation (cos):  {np.mean(cos_extremes):.3f} ±{np.std(cos_extremes):.3f}" if cos_extremes else "    shape preservation: N/A")

    # Cross-token disentanglement matrix
    print(f"\n{'='*90}")
    print(f"  CROSS-TOKEN DISENTANGLEMENT")
    print(f"{'='*90}\n")

    for ti in token_indices:
        token_name = TOKEN_NAMES_V3[ti] if ti < len(TOKEN_NAMES_V3) else f"token_{ti}"
        peak_s = []; arr_s = []; cent_s = []; cos_e = []
        for r in all_results:
            td = r["tokens"].get(f"token_{ti}", {})
            mets = td.get("metrics", {})
            if not mets: continue
            scales = sorted([float(s) for s in mets.keys()])
            if len(scales) < 3: continue
            peaks = [mets[str(s)]["peak"] for s in scales]
            arrivals = [mets[str(s)]["arrival_sample"] for s in scales]
            centroids = [mets[str(s)]["spectral_centroid"] for s in scales]
            peak_s.append(abs(np.polyfit(scales, peaks, 1)[0]))
            arr_s.append(abs(np.polyfit(scales, arrivals, 1)[0]))
            cent_s.append(abs(np.polyfit(scales, centroids, 1)[0]))
            ext = [s for s in scales if abs(s) >= 2.0]
            if ext: cos_e.append(np.mean([mets[str(s)]["cosine_to_base"] for s in ext]))
        print(f"  {ti:2d} {token_name:<30s}  amp_sens={np.mean(peak_s):.4f}  arr_sens={np.mean(arr_s):.4f}  "
              f"spec_sens={np.mean(cent_s):.4f}" + (f"  shape_cos={np.mean(cos_e):.3f}" if cos_e else ""))

    # Verdict
    print(f"\n{'='*90}")
    print(f"  VERDICT")
    print(f"{'='*90}\n")

    # Disentanglement if:
    # 1. Token 0 (source_size) has HIGHEST amplitude sensitivity
    # 2. Token 1 (source_location_depth) has HIGHEST arrival sensitivity
    # 3. No single token dominates ALL metrics
    # 4. Shape preservation (cos) stays high (>0.5) even at extreme scales

    if token_indices == [0, 1, 2]:
        amp_by_token = {}; arr_by_token = {}
        for ti in token_indices:
            amp_s = []; arr_s = []
            for r in all_results:
                td = r["tokens"].get(f"token_{ti}", {})
                mets = td.get("metrics", {})
                if not mets: continue
                scales = sorted([float(s) for s in mets.keys()])
                if len(scales) < 3: continue
                peaks = [mets[str(s)]["peak"] for s in scales]
                arrivals = [mets[str(s)]["arrival_sample"] for s in scales]
                amp_s.append(abs(np.polyfit(scales, peaks, 1)[0]))
                arr_s.append(abs(np.polyfit(scales, arrivals, 1)[0]))
            amp_by_token[ti] = np.mean(amp_s) if amp_s else 0.0
            arr_by_token[ti] = np.mean(arr_s) if arr_s else 0.0

        if max(amp_by_token.values()) > 0:
            best_amp_token = max(amp_by_token, key=amp_by_token.get)
            amp_ratio = amp_by_token[best_amp_token] / max(sum(amp_by_token.values()), 1e-12)
            print(f"  Amplitude control: token {best_amp_token} ({TOKEN_NAMES_V3[best_amp_token]}) "
                  f"has {amp_ratio*100:.0f}% of total sensitivity")

        if max(arr_by_token.values()) > 0:
            best_arr_token = max(arr_by_token, key=arr_by_token.get)
            arr_ratio = arr_by_token[best_arr_token] / max(sum(arr_by_token.values()), 1e-12)
            print(f"  Arrival control:   token {best_arr_token} ({TOKEN_NAMES_V3[best_arr_token]}) "
                  f"has {arr_ratio*100:.0f}% of total sensitivity")

        if best_amp_token == 0 and best_arr_token == 1:
            print("\n  PERFECT DISENTANGLEMENT: token 0 controls amplitude, token 1 controls arrival.")
            print("  JsT has learned separable physical representations.")
        elif best_amp_token != best_arr_token:
            print(f"\n  PARTIAL DISENTANGLEMENT: different tokens control amplitude vs arrival.")
            print(f"  Token space shows some physical separation.")
        else:
            print(f"\n  ENTANGLED: one token ({best_amp_token}) dominates all physical dimensions.")
            print(f"  Token space is NOT physically disentangled.")

    # Save figures
    output_dir = Path("outputs/disentanglement_probe")
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_figures(all_results, token_indices, output_dir)

    return all_results


def _save_figures(all_results, token_indices, output_dir):
    """Generate perturbation response figures."""
    n_samples = len(all_results)
    n_tokens = len(token_indices)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_tokens))

    # Figure 1: Amplitude response per token
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for metric_idx, (metric_name, ylabel) in enumerate([
        ("peak", "Peak Amplitude"), ("cosine_to_base", "Cosine Similarity to Base"),
        ("spectral_centroid", "Spectral Centroid (Hz)"),
    ]):
        ax = axes[metric_idx]
        for ti_idx, ti in enumerate(token_indices):
            token_name = TOKEN_NAMES_V3[ti] if ti < len(TOKEN_NAMES_V3) else f"t{ti}"
            all_scales = []; all_vals = []
            for r in all_results:
                mets = r["tokens"].get(f"token_{ti}", {}).get("metrics", {})
                for s_str, mv in mets.items():
                    if metric_name in mv:
                        all_scales.append(float(s_str))
                        all_vals.append(mv[metric_name])
            if all_scales:
                # Average across samples at each scale
                import pandas as pd
                df = pd.DataFrame({"scale": all_scales, "value": all_vals})
                grouped = df.groupby("scale")["value"].agg(["mean", "std"])
                ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                           color=colors[ti_idx], marker="o", label=token_name, capsize=3)
        ax.set_xlabel("Perturbation Scale (×std)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Token Perturbation Response — Physical Disentanglement", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "perturbation_response.png", dpi=150)
    plt.close()

    # Figure 2: Waveform overlay for one sample (token 0 perturbation)
    if all_results:
        r = all_results[0]
        for ti in token_indices:
            token_name = TOKEN_NAMES_V3[ti] if ti < len(TOKEN_NAMES_V3) else f"token_{ti}"
            wfs = r["tokens"].get(f"token_{ti}", {}).get("waveforms", {})
            if not wfs: continue
            fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
            scales_to_plot = sorted([float(s) for s in wfs.keys()], key=abs)
            colors_ch = ["black", "tab:red", "tab:blue"]
            for ch in range(3):
                ax = axes[ch]
                for scale in scales_to_plot:
                    w = np.array(wfs[str(scale)])
                    alpha = 1.0 if scale == 0.0 else 0.3
                    lw = 2.0 if scale == 0.0 else 0.6
                    ax.plot(w[ch], alpha=alpha, lw=lw, color=plt.cm.RdBu(0.5 + scale/4.0))
                ax.set_ylabel(f"Ch {ch}")
            axes[-1].set_xlabel("Sample")
            fig.suptitle(f"Token {ti} ({token_name}) — Perturbation Response", fontsize=14)
            plt.tight_layout()
            fig.savefig(output_dir / f"waveform_token{ti}.png", dpi=150)
            plt.close()

    print(f"\nFigures saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physical disentanglement probe")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/disentanglement_probe")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_disentanglement_probe(
        args.checkpoint, device,
        n_samples=args.n_samples, drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    with open(output_dir / "results.json", "w") as f:
        saveable = []
        for r in results:
            entry = {"sample": r["sample"], "cache_idx": r["cache_idx"]}
            for k, v in r["tokens"].items():
                entry[k] = {"name": v["name"], "metrics": v["metrics"]}
            saveable.append(entry)
        json.dump(saveable, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
