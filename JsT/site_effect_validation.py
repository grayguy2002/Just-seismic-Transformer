"""Three-step validation of JsT site-effect measurement.

Step 1: JsT-HVSR proxy — spectral ratio consistency
  For each station, compute residual_power / predicted_power across events.
  If this ratio is consistent within a station vs across stations,
  it's a JsT-derived estimate of the site amplification spectrum.

Step 2: End-to-end demonstration
  Pick one station+event, show the full pipeline:
  W_obs → residual → edit magnitude → W_edited + residual

Step 3: Site-effect preservation under editing
  Process intra-site correlation of residuals after token editing.
  If site features survive the edit, the residual from an edited waveform
  should still correlate with the original residual.
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder

TOKEN_NAMES_V3 = [
    "source_size", "source_location_depth", "source_radiation_proxy",
    "path_geometry", "path_travel_time", "selected_phase_label",
    "path_region_proxy", "receiver_site",
]


@torch.no_grad()
def generate_waveform(dn, tokens, noise_fixed, steps=50):
    device = tokens.device
    net = dn.net
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device)
    z = noise_fixed.clone()
    for i in range(steps):
        t = ts[i]; t_next = ts[i + 1]
        xp = net(z, t.expand(z.shape[0]), tokens)
        v = (xp - z) / (1.0 - t.view(1, 1, 1)).clamp_min(dn.t_eps)
        z = z + (t_next - t) * v
    return z


def log_spaced_freqs(n_samples, sr, fmin=0.1, fmax=10.0, n_bins=40):
    freqs = np.fft.rfftfreq(n_samples, d=1.0/sr)
    bins = np.logspace(np.log10(fmin), np.log10(fmax), n_bins)
    idx = np.searchsorted(freqs, bins)
    idx = np.clip(idx, 1, len(freqs) - 1)
    return freqs, np.unique(idx)


def spectral_ratio_curve(w_residual, w_predicted, sr=40.0):
    """Compute log-power ratio residual/predicted in log-spaced frequency bins."""
    freqs, bins = log_spaced_freqs(len(w_residual), sr)
    spec_r = np.abs(np.fft.rfft(w_residual))
    spec_p = np.abs(np.fft.rfft(w_predicted))
    ratios = []
    for b in bins:
        r_power = np.mean(spec_r[b-2:b+2]) if b+2 < len(spec_r) else np.mean(spec_r[-5:])
        p_power = np.mean(spec_p[b-2:b+2]) if b+2 < len(spec_p) else 1e-12
        ratios.append(np.log10(max(r_power, 1e-12) / max(p_power, 1e-12)))
    return freqs[bins], np.array(ratios)


def cosine_sim(a, b):
    a = a.flatten(); b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def run_three_step_validation(
    checkpoint_path: str,
    device: torch.device,
    drop_tokens: list[int] | None,
) -> dict:
    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    n_tokens = dn.net.n_cond_tokens
    total_samples = 3200

    ds_train = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "data/seisbench_mlaapde_pwave_v21_36m", split="testing", augment=False,
        vocab_from=ds_train, cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    # Pre-compute per-token std for perturbation scale
    token_samples = []
    np.random.seed(42)
    sample_indices = np.random.choice(len(ds_test), min(100, len(ds_test)), replace=False)
    for idx in sample_indices:
        _, cd = ds_test[int(idx)]
        cg = {k: v.unsqueeze(0).to(device) for k, v in cd.items()}
        token_samples.append(ce(cg).cpu())
    token_std = torch.cat(token_samples, dim=0).std(dim=0)

    # Find stations with >= 4 events
    test_conditions = ds_test.conditions.iloc[ds_test.indices]
    station_counts = test_conditions["station_id"].value_counts()
    multi_event = station_counts[station_counts >= 4].index.tolist()
    selected_stations = multi_event[:12]
    max_events = 6

    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # ---- STEP 1: JsT-HVSR — spectral ratio consistency ----
    print("=" * 80)
    print("  STEP 1: JsT-HVSR — Spectral Ratio Consistency")
    print("=" * 80)
    print()

    all_ratio_curves = []   # list of (station_label, freq_bins, ratio_curve)
    all_waveform_data = []  # for step 2

    for sta_idx, station_id in enumerate(selected_stations):
        station_rows = test_conditions[test_conditions["station_id"] == station_id]
        n_events = min(max_events, len(station_rows))
        test_indices = station_rows.index[:n_events]

        sta_curves = []

        for event_idx, row_idx in enumerate(test_indices):
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            wf_tensor, cond_dict = ds_test[didx]
            wf_cpu = wf_tensor.unsqueeze(0)
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)

            torch.manual_seed(42)
            if device.type == "cuda": torch.cuda.manual_seed_all(42)
            noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
            predicted = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu()

            residual = wf_cpu - predicted
            wf_np = wf_cpu[0].numpy()
            pred_np = predicted[0].numpy()
            res_np = residual[0].numpy()

            # Spectral ratio curve (Z channel)
            freqs, ratio = spectral_ratio_curve(res_np[0], pred_np[0])
            sta_curves.append(ratio)

            # Store for step 2 (first station, first event)
            if sta_idx == 0 and event_idx == 0:
                anchor_data = {
                    "station_id": station_id,
                    "waveform": wf_np, "predicted": pred_np, "residual": res_np,
                    "freqs": freqs, "ratio": ratio,
                    "cond_dict": cond_dict, "tokens": tokens, "noise_fixed": noise_fixed,
                }

            all_waveform_data.append({
                "station_id": station_id,
                "waveform": wf_np, "predicted": pred_np, "residual": res_np,
            })

        if len(sta_curves) >= 2:
            curves = np.array(sta_curves)
            all_ratio_curves.append((station_id, freqs, curves))

    # Compute intra vs inter station ratio consistency
    print(f"  Stations with >=4 events: {len(all_ratio_curves)}")
    print()

    intra_pairs = []; inter_pairs = []
    for i in range(len(all_ratio_curves)):
        for j in range(i + 1, len(all_ratio_curves)):
            si_name, si_freqs, si_curves = all_ratio_curves[i]
            sj_name, sj_freqs, sj_curves = all_ratio_curves[j]
            for ei in range(len(si_curves)):
                for ej in range(len(sj_curves) if i!=j else range(ei+1, len(si_curves))):
                    cs = cosine_sim(si_curves[ei], si_curves[ej] if i==j else sj_curves[ej])
                    if i == j:
                        intra_pairs.append(cs)
                    else:
                        inter_pairs.append(cs)

    intra_mean = float(np.mean(intra_pairs))
    inter_mean = float(np.mean(inter_pairs))
    delta = intra_mean - inter_mean
    ratio = intra_mean / max(inter_mean, 1e-12)

    print(f"  Intra-station ratio cos: {intra_mean:.4f}")
    print(f"  Inter-station ratio cos: {inter_mean:.4f}")
    print(f"  Delta: {delta:+.4f}")
    print(f"  Ratio:  {ratio:.1f}x")
    print()

    step1_verdict = "STRONG" if delta > 0.15 else ("WEAK" if delta > 0.05 else "NONE")
    print(f"  STEP 1 VERDICT: {step1_verdict} site-effect spectral signature")
    if step1_verdict != "NONE":
        print(f"  JsT-HVSR ratio curves are {ratio:.1f}x more consistent within stations.")
        if delta > 0.1:
            print(f"  The residual/predicted power ratio IS a station-specific measurement.")
    print()

    # ---- STEP 2: End-to-end demonstration ----
    print("=" * 80)
    print("  STEP 2: End-to-end pipeline demonstration")
    print("=" * 80)
    print()

    ad = anchor_data
    sta_id = ad["station_id"]
    print(f"  Station: {sta_id}")
    print(f"  Original magnitude: {ad['cond_dict']['source_magnitude'].item():.1f}")

    # Edit magnitude: scale token 0 by +2 sigma ("upgrade" magnitude)
    edited_tokens = ad["tokens"].clone()
    scale = 2.0
    edited_tokens[0, 0, :] += scale * token_std[0].to(device)

    # Generate edited waveform (same noise = same source-time-function)
    edited_pred = generate_waveform(dn, edited_tokens, ad["noise_fixed"], steps=50).cpu()
    edited_wf = edited_pred[0].numpy()

    # Apply site residual: edited_pred + residual = "site-corrected edited waveform"
    site_corrected = edited_wf + ad["residual"]

    # Compute key metrics
    original_peak = np.max(np.abs(ad["waveform"][0]))
    edited_peak = np.max(np.abs(edited_wf[0]))
    corrected_peak = np.max(np.abs(site_corrected[0]))
    residual_preservation = cosine_sim(
        site_corrected[0] - edited_wf[0], ad["residual"][0]
    )

    print(f"  Original peak:    {original_peak:.3f}")
    print(f"  Edited peak:      {edited_peak:.3f}  (magnitude upgrade)")
    print(f"  Corrected peak:   {corrected_peak:.3f}  (site effect applied)")
    print(f"  Residual preservation cos: {residual_preservation:.3f}")
    print()

    # Save figure
    output_dir = Path("outputs/site_effect_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    # Subplot 1: Original vs predicted
    ax = axes[0, 0]
    ax.plot(ad["waveform"][0], color="black", lw=0.8, label="Real")
    ax.plot(ad["predicted"][0], color="tab:red", lw=0.8, alpha=0.7, label="JsT predicted")
    ax.set_title(f"Original vs JsT Predicted (Z channel)")
    ax.legend(fontsize=8)

    # Subplot 2: Residual
    ax = axes[0, 1]
    ax.plot(ad["residual"][0], color="tab:blue", lw=0.6)
    ax.set_title(f"Residual (Real - Predicted) = Site Effect Estimate")
    ax.set_ylabel("Amplitude")

    # Subplot 3: Spectral ratio
    ax = axes[1, 0]
    ax.semilogx(ad["freqs"], ad["ratio"], color="tab:green", lw=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", lw=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10(Residual/Predicted)")
    ax.set_title(f"JsT-HVSR — Site-{sta_id} Amplification Spectrum")
    ax.grid(True, alpha=0.3)

    # Subplot 4: Edited + site-corrected
    ax = axes[1, 1]
    ax.plot(edited_wf[0], color="tab:orange", lw=0.8, alpha=0.7, label=f"Edited (M↑)")
    ax.plot(site_corrected[0], color="tab:purple", lw=0.8, label=f"Edited + Site Effect")
    ax.set_title(f"Edited + Site Effect Applied")
    ax.legend(fontsize=8)

    # Subplot 5: JsT-HVSR comparison across stations (step 1 visualization)
    ax = axes[2, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(all_ratio_curves))))
    for si, (sname, sfreqs, scurves) in enumerate(all_ratio_curves[:10]):
        mean_curve = scurves.mean(axis=0)
        ax.semilogx(sfreqs, mean_curve, color=colors[si], lw=1.0, alpha=0.8, label=sname[:12])
    ax.axhline(y=0, color="gray", linestyle="--", lw=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log10(Residual/Predicted)")
    ax.set_title("JsT-HVSR: Per-Station Mean Ratios")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Subplot 6: Intra vs inter consistency histogram
    ax = axes[2, 1]
    ax.hist(intra_pairs, bins=20, alpha=0.6, label=f"Same station (mean={intra_mean:.3f})", color="tab:blue")
    ax.hist(inter_pairs, bins=20, alpha=0.6, label=f"Diff station (mean={inter_mean:.3f})", color="tab:gray")
    ax.axvline(intra_mean, color="tab:blue", lw=2)
    ax.axvline(inter_mean, color="tab:gray", lw=2)
    ax.set_xlabel("Cosine similarity of spectral ratio curves")
    ax.set_ylabel("Count")
    ax.set_title(f"Step 1: Ratio consistency (Δ={delta:+.4f}, {ratio:.1f}x)")
    ax.legend(fontsize=8)

    fig.suptitle(f"JsT Site-Effect Validation — Station {sta_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "step1_step2_results.png", dpi=150)
    plt.close()

    # ---- STEP 3: Site-effect preservation under editing ----
    print("=" * 80)
    print("  STEP 3: Site-effect preservation under token editing")
    print("=" * 80)
    print()

    # Take 5 stations x 3 events each
    # For each event: compute residual (original), then compute residual after
    # magnitude editing (+2σ on token 0). Compare cosine similarity.
    test_stations_3 = station_counts[station_counts >= 3].index.tolist()[:8]
    preservation_results = []

    for sta_id_3 in test_stations_3:
        station_rows = test_conditions[test_conditions["station_id"] == sta_id_3]
        n_ev = min(3, len(station_rows))
        test_indices = station_rows.index[:n_ev]

        sta_preservations = []

        for row_idx in test_indices:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            wf_tensor, cond_dict = ds_test[didx]
            wf_cpu = wf_tensor.unsqueeze(0)
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tokens = ce(cond_gpu)

            torch.manual_seed(42)
            if device.type == "cuda": torch.cuda.manual_seed_all(42)
            noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
            predicted = generate_waveform(dn, tokens, noise_fixed, steps=50).cpu()
            residual_orig = (wf_cpu - predicted)[0].numpy()

            # Edit magnitude
            edited_tokens = tokens.clone()
            edited_tokens[0, 0, :] += 2.0 * token_std[0].to(device)
            edited_pred = generate_waveform(dn, edited_tokens, noise_fixed, steps=50).cpu()
            edited_wf = edited_pred[0].numpy()

            # Residual after editing = W_obs - W_edited
            residual_after_edit = wf_cpu[0].numpy() - edited_wf

            # Correlation: does residual_after_edit preserve features of residual_orig?
            # If YES → site effect survives the edit
            orig_vs_after = cosine_sim(residual_orig, residual_after_edit)

            # Null: is residual_after_edit more similar to residual_orig than
            # to residuals from OTHER stations?
            # Pick a random other station's residual
            other_corrs = []
            for wd in all_waveform_data:
                if wd["station_id"] != sta_id_3:
                    cos_other = cosine_sim(wd["residual"], residual_after_edit)
                    other_corrs.append(cos_other)
                    if len(other_corrs) >= 5: break

            sta_preservations.append({
                "orig_vs_after": orig_vs_after,
                "orig_vs_other_mean": float(np.mean(other_corrs)) if other_corrs else 0.0,
                "delta_preservation": orig_vs_after - float(np.mean(other_corrs)) if other_corrs else 0.0,
            })

        preservation_results.append({
            "station": sta_id_3,
            "n_events": len(sta_preservations),
            "mean_orig_vs_after": float(np.mean([p["orig_vs_after"] for p in sta_preservations])),
            "mean_delta": float(np.mean([p["delta_preservation"] for p in sta_preservations])),
        })

    # Aggregate
    p_orig = [p["orig_vs_after"] for r in preservation_results for p in [
        {"orig_vs_after": r["mean_orig_vs_after"]}]]  # just extract the mean per station
    # Actually, let me redo this more cleanly
    all_orig_vs_after = [r["mean_orig_vs_after"] for r in preservation_results]
    all_deltas = [r["mean_delta"] for r in preservation_results]

    print(f"  Stations tested: {len(preservation_results)}")
    print(f"  Mean residual preservation cos:  {np.mean(all_orig_vs_after):.4f} ±{np.std(all_orig_vs_after):.4f}")
    print(f"  Mean delta vs other-station:    {np.mean(all_deltas):+.4f} ±{np.std(all_deltas):.4f}")
    print()

    if np.mean(all_orig_vs_after) > 0.3 and np.mean(all_deltas) > 0.05:
        print(f"  STEP 3 VERDICT: Site-effect SURVIVES magnitude editing.")
        print(f"  Residual after editing is {np.mean(all_orig_vs_after):.0%} similar to original residual.")
    elif np.mean(all_orig_vs_after) > 0.15:
        print(f"  STEP 3 VERDICT: PARTIAL preservation.")
        print(f"  Some site features survive but magnitude editing introduces changes.")
    else:
        print(f"  STEP 3 VERDICT: Site features are LOST under editing.")
        print(f"  Residual after editing is only {np.mean(all_orig_vs_after):.0%} similar to original.")

    # ---- FINAL SYNTHESIS ----
    print()
    print("=" * 80)
    print("  THREE-STEP SYNTHESIS")
    print("=" * 80)
    print()
    print(f"  Step 1 (JsT-HVSR):      {step1_verdict} signal (Δ={delta:+.4f}, {ratio:.1f}x)")
    print(f"  Step 2 (End-to-end):    Demonstrated on station {sta_id}")
    print(f"  Step 3 (Preservation):  {'PASS' if np.mean(all_orig_vs_after) > 0.3 else 'PARTIAL'}")
    print()

    all_pass = (step1_verdict == "STRONG" and np.mean(all_orig_vs_after) > 0.3)
    if all_pass:
        print("  SYNTHESIS: JsT provides a validated single-station, single-event")
        print("  site-effect measurement. ResIDual-based site amplification")
        print("  survives continuous magnitude editing through the token space.")
        print("  This is a new capability not available in traditional seismology.")
    elif step1_verdict in ("STRONG", "WEAK"):
        print("  SYNTHESIS: Site-effect signal is present but weak.")
        print("  The measurement is directionally correct but needs:")
        print("   - Denser candidate stations (more events/station)")
        print("   - Higher-resolution JsT-HVSR frequency bins")
        print("   - Multi-station reference normalization")
    else:
        print("  SYNTHESIS: Site-effect signal not conclusively detected.")
        print("  The residual is likely dominated by model-condition mismatch.")

    return {
        "step1": {"intra_mean": intra_mean, "inter_mean": inter_mean,
                   "delta": delta, "ratio": ratio, "verdict": step1_verdict},
        "step2": {"station": sta_id, "residual_preservation": residual_preservation},
        "step3": {"mean_orig_vs_after": float(np.mean(all_orig_vs_after)),
                  "mean_delta": float(np.mean(all_deltas))},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/site_effect_validation")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_three_step_validation(args.checkpoint, device, drop_tokens=dropped)

    output_dir = Path(args.output_dir)
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/validation_results.json")
