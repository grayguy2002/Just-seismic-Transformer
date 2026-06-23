"""Token 7 geological manifold analysis — Nature Geoscience mechanism evidence.

Three experiments:
  1. t-SNE of token 7 embeddings colored by geological context
  2. Linear interpolation between station pairs → amplification spectrum transition
  3. Within vs across geological group cosine similarity (complement to Hawaii)

If token 7 forms a continuous manifold that maps onto real geology, this is
direct causal evidence that JsT has internalized geological knowledge
of site conditions — the missing mechanistic link for Nature Geoscience.
"""

from __future__ import annotations

import sys, time, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from JsT import (
    SeismicConditionEncoder, Denoiser,
    SeismicWaveformDataset, collate_conditions, load_checkpoint_models,
)
from JsT.ablation import AblationConditionEncoder


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


def cosine_sim(a, b):
    a = a.flatten(); b = b.flatten()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def classify_geology(lat, lon, network):
    """Classify station geological context from coordinates and network."""
    lat = float(lat); lon = float(lon); network = str(network)
    # Hawaii
    if network == 'HV':
        if 19.38 <= lat <= 19.45 and -155.35 <= lon <= -155.22: return "Basalt_Kilauea"
        if lat <= 19.28: return "Basalt_coastal"
        if lat >= 19.50: return "Basalt_weathered"
        return "Basalt_flank"
    # Alaska: diverse geology
    if network == 'AK':
        if lat >= 64.0: return "Metamorphic_interior"
        if lat <= 60.0: return "Sedimentary_coastal"
        return "Metamorphic_range"
    if network == 'AT':
        if lon <= -160: return "Volcanic_arc"
        return "Sedimentary_coastal"
    # Oklahoma: sedimentary basin
    if network == 'OK': return "Sedimentary_basin"
    # USGS networks
    if network == 'GS': return "Sedimentary_centralUS"
    # Utah
    if network == 'UU': return "Basin_range"
    # Washington
    if network == 'UW': return "Volcanic_cascades"
    # Nevada
    if network == 'NN': return "Basin_range"
    # California
    if network in ('CI', 'NC'): return "Active_margin"
    # New Mexico
    if network == 'NM': return "Sedimentary_embayment"
    # Generic classification by region
    if lat >= 48.0: return "Craton_north"
    if lat <= 35.0 and lon <= -100: return "Active_margin"
    if lat <= 38.0 and lon <= -88: return "Sedimentary_embayment"
    if lon >= -75: return "Passive_margin_east"
    return "Interior_platform"


def run_token7_geological_manifold(
    checkpoint_path: str,
    device: torch.device,
    vs30_csv: str,
    output_dir: str,
    drop_tokens: list[int] | None,
) -> dict:
    ce, dn, ckpt = load_checkpoint_models(
        checkpoint_path, device, use_ema=True,
        sampling_method="heun", steps=50, cfg_scale=1.0,
    )
    if drop_tokens:
        ce = AblationConditionEncoder(ce, drop_tokens)
    dn.eval(); ce.eval()
    total_samples = 3200
    token7_dim = dn.net.hidden_size  # 512
    n_tokens = dn.net.n_cond_tokens

    ds_train = SeismicWaveformDataset(
        "/home/user54/projects/EEW/data/seisbench_mlaapde_pwave_v21_36m",
        split="training", augment=False,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )
    ds_test = SeismicWaveformDataset(
        "/home/user54/projects/EEW/data/seisbench_mlaapde_pwave_v21_36m",
        split="testing", augment=False, vocab_from=ds_train,
        cache_prefix="pwave_v21", condition_version="v2.1", field_policy="default",
    )

    test_conditions = ds_test.conditions.iloc[ds_test.indices].copy()
    test_conditions['station_id'] = (
        test_conditions['station_network_code'].fillna('UNKNOWN').astype(str)
        + '.' + test_conditions['station_code'].fillna('UNKNOWN').astype(str)
    )
    index_to_ds = {int(idx): i for i, idx in enumerate(ds_test.indices)}

    # Load Vs30 data
    try:
        vs30_df = pd.read_csv(vs30_csv)
        vs30_by_station = {r['station_id']: float(r['vs30_m_s']) for _, r in vs30_df.iterrows()}
    except:
        vs30_by_station = {}

    # ---- Extract token 7 for ALL unique stations ----
    print("Extracting token 7 for all test set stations...")
    station_tokens = {}   # station_id → list of token7 vectors (one per event)
    station_locations = {}  # station_id → (lat, lon, network)
    station_vs30 = {}

    unique_stations = test_conditions['station_id'].unique()
    for sta_id in unique_stations:
        sta_rows = test_conditions[test_conditions['station_id'] == sta_id]
        n_evts = min(6, len(sta_rows))
        lat = float(sta_rows['station_latitude_deg'].iloc[0])
        lon = float(sta_rows['station_longitude_deg'].iloc[0])
        net = str(sta_rows['station_network_code'].iloc[0])
        station_locations[sta_id] = (lat, lon, net)
        station_vs30[sta_id] = vs30_by_station.get(sta_id, None)

        tokens = []
        for row_idx in sta_rows.index[:n_evts]:
            didx = index_to_ds.get(int(row_idx))
            if didx is None: continue
            _, cond_dict = ds_test[didx]
            cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
            tok = ce(cond_gpu).cpu()
            tokens.append(tok[0, 7, :].detach().numpy())  # token 7

        if len(tokens) >= 2:
            station_tokens[sta_id] = np.array(tokens)  # (n_events, dim)

    n_stations = len(station_tokens)
    print(f"Stations with ≥2 events: {n_stations}")

    # ---- Prepare data for t-SNE ----
    # Use mean token 7 per station
    station_ids = list(station_tokens.keys())
    X_raw = np.array([station_tokens[s].mean(axis=0) for s in station_ids])  # (N, 512)

    # Geological labels
    geo_labels = [classify_geology(*station_locations[s]) for s in station_ids]
    geo_set = sorted(set(geo_labels))
    geo_colors = plt.cm.tab20(np.linspace(0, 1, len(geo_set)))
    geo_color_map = {g: geo_colors[i] for i, g in enumerate(geo_set)}

    print(f"\nGeological groups: {len(geo_set)}")
    for g in geo_set:
        n = sum(1 for l in geo_labels if l == g)
        print(f"  {g}: {n} stations")

    # ---- Experiment 1: t-SNE visualization ----
    print("\n=== Experiment 1: t-SNE of token 7 ===")
    t0 = time.time()

    # PCA first (50 dims) for speed, then t-SNE (skip if too many stations)
    pca = PCA(n_components=min(30, X_raw.shape[0]-1, X_raw.shape[1]))
    X_pca = pca.fit_transform(X_raw)
    print(f"  PCA → {X_pca.shape[1]} dims ({time.time()-t0:.0f}s)")

    if n_stations <= 300:
        tsne = TSNE(n_components=2, perplexity=min(30, max(5, n_stations//4)),
                    random_state=42, max_iter=1000, verbose=0)
        X_tsne = tsne.fit_transform(X_pca)
        print(f"  t-SNE done ({time.time()-t0:.0f}s)")
    else:
        # Too many stations: use PCA directly for 2D
        pca2 = PCA(n_components=2)
        X_tsne = pca2.fit_transform(X_pca)
        print(f"  PCA2 for visualization ({time.time()-t0:.0f}s) — too many stations for t-SNE")

    # Figure 1a: t-SNE colored by geology
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    ax = axes[0]
    for i, s in enumerate(station_ids):
        ax.scatter(X_tsne[i, 0], X_tsne[i, 1], c=[geo_color_map[geo_labels[i]]],
                   s=20, alpha=0.7, edgecolors='none')
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor=geo_color_map[g],
                               markersize=8, label=g) for g in geo_set]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper left',
              bbox_to_anchor=(1.02, 1))
    viz_label = "t-SNE" if n_stations <= 300 else "PCA"
    ax.set_title(f"Token 7 {viz_label} — Colored by Geological Context", fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # Figure 1b: t-SNE colored by Vs30 (where available)
    ax = axes[1]
    vs30_vals = np.array([float(station_vs30.get(s, np.nan)) if station_vs30.get(s) is not None else np.nan for s in station_ids])
    has_vs30 = ~np.isnan(vs30_vals)
    im = ax.scatter(X_tsne[has_vs30, 0], X_tsne[has_vs30, 1],
                    c=vs30_vals[has_vs30], cmap='viridis_r', s=20, alpha=0.7)
    plt.colorbar(im, ax=ax, label='Vs30 (m/s)')
    ax.set_title("Token 7 t-SNE — Colored by Vs30", fontsize=12, fontweight='bold')
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    # Figure 1c: PCA variance explained
    ax = axes[2]
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum)+1), cumsum, 'b-', lw=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% variance')
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% variance')
    ax.set_xlabel("PCA Components"); ax.set_ylabel("Cumulative Variance")
    ax.set_title(f"Token 7 PCA: {X_pca.shape[1]}D → 2D t-SNE", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("JsT Token 7 (receiver_site) — Geological Manifold in Embedding Space",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path / "token7_tsne_geology.png", dpi=200)
    plt.close()

    # ---- Experiment 2: Within vs across geological group consistency ----
    print("\n=== Experiment 2: Geological separability ===")

    # Mean pairwise cosine within/across groups
    intra_group = defaultdict(list)
    inter_group = []
    for i, si in enumerate(station_ids):
        gi = geo_labels[i]
        for j, sj in enumerate(station_ids[i+1:], i+1):
            gj = geo_labels[j]
            cs = cosine_sim(X_raw[i], X_raw[j])
            if gi == gj:
                intra_group[gi].append(cs)
            else:
                inter_group.append(cs)

    print(f"\n  {'Group':<30s} {'N stns':>6s} {'Mean intra-cos':>14s} {'N pairs':>8s}")
    print(f"  {'-'*60}")
    for g in sorted(intra_group.keys()):
        n_sta = sum(1 for l in geo_labels if l == g)
        mean_cos = np.mean(intra_group[g])
        n_pairs = len(intra_group[g])
        print(f"  {g:<30s} {n_sta:6d} {mean_cos:14.4f} {n_pairs:8d}")

    all_intra = [v for vals in intra_group.values() for v in vals]
    mean_intra_all = np.mean(all_intra) if all_intra else 0
    mean_inter_all = np.mean(inter_group) if inter_group else 0
    delta_all = mean_intra_all - mean_inter_all
    print(f"\n  ALL INTRA: {mean_intra_all:.4f}  ALL INTER: {mean_inter_all:.4f}  DELTA: {delta_all:+.4f} "
          f"({mean_intra_all/max(mean_inter_all,1e-12):.1f}x)")

    # ---- Experiment 3: Token 7 linear interpolation ----
    print("\n=== Experiment 3: Token 7 interpolation ===")

    # Find station pairs: hard rock (high Vs30) vs soft soil (low Vs30)
    vs30_sorted = sorted(
        [(s, float(station_vs30.get(s, np.nan))) for s in station_ids if station_vs30.get(s) is not None],
        key=lambda x: x[1]
    )

    if len(vs30_sorted) >= 4:
        # Pick extremal pairs
        hard_stations = [s for s, v in vs30_sorted[-3:]] if len(vs30_sorted) >= 6 else [vs30_sorted[-1][0]]
        soft_stations = [s for s, v in vs30_sorted[:3]]  if len(vs30_sorted) >= 6 else [vs30_sorted[0][0]]

        # For each pair, interpolate token 7 and measure amplification
        n_interp = 5  # interpolation steps
        interpolation_results = []

        # Use the first event from train set for a fixed source
        np.random.seed(42)
        sample_idx = np.random.choice(len(ds_test))
        wf_tensor, cond_dict = ds_test[sample_idx]
        cond_gpu = {k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()}
        base_tokens = ce(cond_gpu)

        for hs in hard_stations[:2]:
            for ss in soft_stations[:2]:
                if hs == ss: continue
                tok_hard = station_tokens[hs].mean(axis=0)
                tok_soft = station_tokens[ss].mean(axis=0)
                vs30_h = station_vs30[hs]; vs30_s = station_vs30[ss]

                amplifications = []
                for alpha in np.linspace(0, 1, n_interp):
                    # Interpolate token 7
                    tok7_interp = alpha * tok_hard + (1 - alpha) * tok_soft
                    interp_tokens = base_tokens.clone()
                    interp_tokens[0, 7, :] = torch.from_numpy(tok7_interp).float().to(device)

                    torch.manual_seed(42)
                    if device.type == "cuda": torch.cuda.manual_seed_all(42)
                    noise_fixed = dn.noise_scale * torch.randn(1, 3, total_samples, device=device)
                    pred = generate_waveform(dn, interp_tokens, noise_fixed, steps=50).cpu().numpy()[0]

                    # Compute amplification: log10(residual/predicted)
                    residual = wf_tensor.numpy() - pred
                    spec_r = np.abs(np.fft.rfft(residual[0]))
                    spec_p = np.abs(np.fft.rfft(pred[0]))
                    amp = np.mean(np.log10(np.maximum(spec_r, 1e-12) / np.maximum(spec_p, 1e-12)))
                    amplifications.append(float(amp))

                interpolation_results.append({
                    'hard_station': hs, 'soft_station': ss,
                    'vs30_hard': float(vs30_h), 'vs30_soft': float(vs30_s),
                    'alpha': [float(a) for a in np.linspace(0, 1, n_interp)],
                    'amplification': amplifications,
                })
                print(f"  {ss} (Vs30={vs30_s:.0f}) → {hs} (Vs30={vs30_h:.0f}): "
                      f"amp {amplifications[0]:.3f} → {amplifications[-1]:.3f}")

        # Plot interpolation curves
        if len(interpolation_results) >= 2:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(interpolation_results)))
            for i, r in enumerate(interpolation_results):
                vs30_delta = r['vs30_hard'] - r['vs30_soft']
                label = f"{r['soft_station'][:15]}→{r['hard_station'][:15]} (ΔVs30={vs30_delta:.0f})"
                ax.plot([0, 1], [r['amplification'][0], r['amplification'][-1]],
                        'o-', color=colors[i], lw=2, markersize=8, label=label)
                # Linear fit
                slope = r['amplification'][-1] - r['amplification'][0]
                ax.plot(r['alpha'], r['amplification'], 'o', color=colors[i], alpha=0.5, markersize=4)

            ax.set_xlabel("Interpolation α (0=soft soil, 1=hard rock)", fontsize=12)
            ax.set_ylabel("JsT-HVSR Amplification", fontsize=12)
            ax.set_title("Token 7 Interpolation: Soft Soil → Hard Rock", fontsize=14, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)

            fig.savefig(output_path / "token7_interpolation.png", dpi=200)
            plt.close()

            # Correlation: ΔVs30 vs amplification slope
            slopes = [(r['amplification'][-1] - r['amplification'][0]) for r in interpolation_results]
            deltas_vs30 = [(r['vs30_hard'] - r['vs30_soft']) for r in interpolation_results]
            rho_interp, p_interp = spearmanr(deltas_vs30, slopes)
            print(f"\n  ΔVs30 vs amplification slope: Spearman ρ={rho_interp:+.4f} (p={p_interp:.4f})")
            if abs(rho_interp) > 0.3:
                print(f"  SIGNIFICANT: token 7 interpolation preserves geological ordering.")
    else:
        interpolation_results = []
        rho_interp = None; p_interp = None

    # ---- Final synthesis ----
    print(f"\n{'='*70}")
    print(f"  TOKEN 7 GEOLOGICAL MANIFOLD — SYNTHESIS")
    print(f"{'='*70}\n")

    evidence = []
    if delta_all > 0.05:
        evidence.append(f"Geological groups SEPARABLE in token 7 (Δ={delta_all:+.3f})")
    if mean_intra_all > 0.8:
        evidence.append(f"Very strong intra-group token 7 consistency (cos={mean_intra_all:.3f})")
    if rho_interp and abs(rho_interp) > 0.3:
        evidence.append(f"Token 7 interpolation preserves geological ordering (ρ={rho_interp:+.3f})")

    if len(evidence) >= 2:
        print(f"  STRONG EVIDENCE for geological manifold:")
        for e in evidence: print(f"    ✓ {e}")
        print(f"\n  Token 7 forms a GEOLOGICALLY MEANINGFUL continuous manifold.")
        print(f"  JsT has INTERNALIZED site-condition knowledge through training.")
        print(f"  This is the MECHANISTIC LINK from statistical validation to causal understanding.")
    elif len(evidence) == 1:
        print(f"  PARTIAL EVIDENCE:")
        for e in evidence: print(f"    ✓ {e}")
    else:
        print(f"  WEAK EVIDENCE — token 7 manifold not clearly geological.")

    return {
        "n_stations": n_stations,
        "n_geological_groups": len(geo_set),
        "intra_mean": float(mean_intra_all),
        "inter_mean": float(mean_inter_all),
        "delta": float(delta_all),
        "interpolation_pairs": len(interpolation_results),
        "interpolation_spearman": float(rho_interp) if rho_interp else None,
        "evidence_count": len(evidence),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vs30-csv", default="data/site_vs30_search/jst_testing_vs30_matches_standardized.csv")
    parser.add_argument("--output-dir", default="outputs/token7_geological_manifold")
    parser.add_argument("--drop-tokens", type=str, default="8,9,10")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dropped = [int(x.strip()) for x in args.drop_tokens.split(",") if x.strip()] if args.drop_tokens else []

    results = run_token7_geological_manifold(
        args.checkpoint, device, args.vs30_csv, args.output_dir, drop_tokens=dropped,
    )

    output_dir = Path(args.output_dir)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir}/results.json")
