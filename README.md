# JsT — Just seismic Transformer

Conditional generative seismic waveform model for single-event site-effect measurement.

## Structure

```
code/
├── JsT/                  # Core model library
│   ├── model.py          # Flow-matching transformer architecture
│   ├── condition_encoder.py  # 8-token source/path/receiver encoder
│   ├── denoiser.py       # x-prediction denoiser
│   ├── dataset.py        # MLAAPDE v2.1 waveform dataset
│   ├── checkpoint.py     # Checkpoint loading + EMA
│   ├── ablation.py       # Token ablation wrapper
│   ├── train.py          # Training loop
│   ├── rope_1d.py        # 1-D rotary position encoding
│   └── ...               # Experiment scripts
├── figures/              # Figure rendering and experiments
│   ├── nature_geo_style.py   # Shared Nature Geoscience visual style
│   ├── fig1_compute.py / fig1_render.py    # Fig 1: Phenomenon
│   ├── fig2_compute.py / fig2_render.py    # Fig 2: Validation
│   ├── fig3_compute.py / fig3_render.py    # Fig 3: Mechanism
│   ├── expA_vs30_controls.py               # Vs30 robustness controls
│   ├── expB_cross_method.py                # Cross-method agreement
│   ├── expC_compute.py                     # Token 7 disentanglement
│   ├── expE_kiknet_method_ceiling.py       # KiK-net standard HVSR benchmark
│   ├── expF_kiknet_jst_inference.py        # JsT × KiK-net cross-domain test
│   ├── expH_dense_single_event_postprocess.py  # Dense KiK-net post-processing
│   ├── expI_kiknet_validation_controls.py      # Metadata/spatial/event-fixed
│   ├── expJ_kiknet_token_perturbation.py       # Token-donor experiment
│   ├── expK_disjoint_audit.py                  # Train/test disjoint audit
│   └── ...
├── scripts/              # Data pipeline
│   ├── build_kiknet_measured_vs30_manifest.py      # KiK-net station manifest
│   ├── build_kiknet_measured_vs30_hvsr_validation.py  # GFZ HVSR matching
│   ├── build_kiknet_designsafe_flatfile_subset.py  # DesignSafe flatfile
│   ├── build_kiknet_mlaapde_compatible_cache.py    # JsT-compatible cache
│   ├── build_kiknet_dense_single_event_cache.py    # Dense KiK-net cache
│   ├── build_seisbench_mlaapde_training_cache.py   # Training cache builder
│   ├── search_nied_kiknet_records.py               # NIED search
│   ├── download_nied_kiknet_zips.py                # NIED download
│   └── ...
└── checkpoint/
    └── checkpoint-last.pth   # V4 final (800 epochs, 851 MB)
```

## Dependencies

```
torch numpy scipy pandas scikit-learn matplotlib obspy cartopy
```

## Usage

### Inference with V4 checkpoint (site-effect measurement)

```python
import torch
from JsT import load_checkpoint_models, SeismicWaveformDataset
from JsT.ablation import AblationConditionEncoder

device = torch.device("cuda")
ce, dn, ckpt = load_checkpoint_models(
    "code/checkpoint/checkpoint-last.pth", device, use_ema=True,
    sampling_method="heun", steps=50, cfg_scale=1.0,
)
ce = AblationConditionEncoder(ce, [8, 9, 10])  # remove identity tokens
dn.eval(); ce.eval()

# Load your v2.1-format conditions.csv + waveform .npy cache
ds = SeismicWaveformDataset("path/to/cache", split="testing", ...)
wf_tensor, cond_dict = ds[0]

# Generate predicted waveform
tokens = ce({k: v.unsqueeze(0).to(device) for k, v in cond_dict.items()})
# ... run ODE integration (see fig1_compute.py for full example)
# residual = observed - predicted  →  JsT-HVSR
```

### Figure rendering (local, from pre-computed caches)

```bash
python3 manuscript/figures/fig1_render.py   # Phenomenon
python3 manuscript/figures/fig2_render.py   # Validation
python3 manuscript/figures/fig3_render.py   # Mechanism
```

## Checkpoint

`checkpoint/checkpoint-last.pth` — V4 (8-token) final, 800 epochs, 5.3M encoder parameters.

Trained on MLAAPDE v2.1 36-month P-wave cache (56,047 source–station pairs).  
Flow-matching with x-prediction + v-loss. CFG dropout 0.4 (epoch 200+).  
Batch size 1024, AdamW, cosine annealing.

## License

[To be determined]
