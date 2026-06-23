"""Upload JsT V4 checkpoint to Hugging Face Hub.

Usage:
  python3 scripts/upload_to_hf.py --token hf_YOUR_TOKEN
  # or set HF_TOKEN env var

Creates: https://huggingface.co/gary2002/jst-v4
"""

import argparse, os, sys
from pathlib import Path

CKPT = Path(__file__).resolve().parent.parent / "checkpoint" / "checkpoint-last.pth"
REPO = "gary2002/jst-v4"

MODEL_CARD = """---
license: cc-by-4.0
pipeline_tag: feature-extraction
tags:
  - seismology
  - site-effects
  - generative-model
  - flow-matching
  - Vs30
  - HVSR
---

# JsT V4 — Just seismic Transformer

An 8-token conditional flow-matching generative model for three-component
P-wave seismograms. Turns one earthquake record into a site-effect measurement.

## Model

- **Architecture**: 8-token condition encoder (source×3, path×4, receiver×1) +
  8-layer flow-matching transformer (512 dim, 8 heads), adaLN-Zero modulation,
  1-D rotary position encoding
- **Params**: 5.3M (encoder) + transformer denoiser
- **Training**: 800 epochs, batch 1024, AdamW, cosine annealing
- **Data**: MLAAPDE v2.1 36-month P-wave cache (56,047 source–station pairs)
- **Inference**: 50-step Heun ODE integration, CFG scale 1.0

## Usage

```python
import torch
from JsT import load_checkpoint_models, AblationConditionEncoder

device = torch.device("cuda")
ce, dn, ckpt = load_checkpoint_models(
    "checkpoint-last.pth", device, use_ema=True,
    sampling_method="heun", steps=50, cfg_scale=1.0,
)
ce = AblationConditionEncoder(ce, [8, 9, 10])  # remove identity tokens
dn.eval(); ce.eval()
```

Full code: [Just-seismic-Transformer](https://github.com/grayguy2002/Just-seismic-Transformer)

## Reference

Preprint forthcoming. When using this checkpoint, please cite the corresponding
paper and the JsT code repository.

## License

This checkpoint is released under **CC BY 4.0**.
The accompanying code is released under **MIT**.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: set HF_TOKEN env var or pass --token")
        print("Get yours at: https://huggingface.co/settings/tokens")
        return 1

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    # Create repo
    print(f"Creating {args.repo}...")
    create_repo(args.repo, token=args.token, private=args.private, exist_ok=True)

    # Upload model card
    print("Uploading model card...")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
    )

    # Upload checkpoint (chunked resume-friendly upload)
    print(f"Uploading checkpoint ({Path(args.ckpt).stat().st_size // 1024 // 1024} MB)...")
    api.upload_file(
        path_or_fileobj=args.ckpt,
        path_in_repo="checkpoint-last.pth",
        repo_id=args.repo,
        repo_type="model",
    )

    print(f"\nDone: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    raise SystemExit(main())
