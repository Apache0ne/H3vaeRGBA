# Same 4000-step Alpha Diffusion Sidecar ↔ two VAE decoders A/B

**Status (2026-10-12):** The bad fox masks came from the **already-connected**
4000-step RGB-conditioned alpha diffusion sidecar **and the trained dual-path
VAE LoRA working together**. The existing generation script literally calls:

```python
z_generated = sample(model, context, schedule, steps=30, seed=seed)
z_alpha = unnormalized_latents(z_generated, codec.vae)
predicted = codec.decode_alpha(z_rgb, z_alpha)
```

So treating the poor visual output as "sidecar alone, without our successful
dual-path VAE" was incorrect.

This A/B explicitly tests the key question: **does the trained VAE LoRA improve
alpha decoding from the sidecar's imperfect generated latents?**

## Fair paired design

Run `tools/compare_sidecar_dual_ab.py` on the same H3-generated 124-frame
RGB video and all nine never-validated source-domain holdout clips.

Each 17-frame window obtains RGB VAE latents once and samples **one z_alpha**
with the frozen, already-trained 4K-step sidecar. Then:

| Branch | Same alpha latents? | Decoder and prediction |
|---|---|---|
| A. Original H3 grayscale decoder | Yes | Frozen original RGB decoder, LoRA disabled; mean of denormalized RGB outputs interpreted as alpha |
| B. Trained dual-path VAE LoRA | Yes | Trained 48->24 latent merger, decoder LoRA and alpha head |
| C. Oracle reference (holdout only) | No | Ground-truth alpha encoded through the same dual VAE. Measures *potential ceiling* of the decoder when correct alpha latents are available |

**Important:** A is NOT the older *widened 4-in/4-out RGBA VAE* experiment.
That model uses a shared 24-channel RGBA latent space, whereas this 4000-step
sidecar predicts a separate alpha 24-channel stream. The old widened
checkpoint cannot validly consume sidecar alpha latents in the same space.
A widened-vs-dual generation comparison would require an additional sidecar
trained against the widened checkpoint.

## Fresh Colab cell — reuse weights and original H3 MP4; NO 33B denoising

```python
import json, subprocess, sys
from pathlib import Path
from IPython.display import display, Image

R = Path("/content/H3vaeRGBA")
ROOT = Path("/content/h3_rgba_end_to_end")
OUT = ROOT / "same_sidecar_dual_ab"

subprocess.run(["git","-C",str(R),"pull","--ff-only"],check=True)

h3_report=json.loads(
    (ROOT/"full_h3/h3_a100_full_t2va_report.json").read_text())
RGB = Path(h3_report["generated_rgb_audio_video"])
assert RGB.is_file(),RGB

adapter = ROOT/"dual_vae/checkpoints/dual_lora_best.safetensors"
if not adapter.is_file():
    adapter=Path("/content/drive/MyDrive/H3_DualLoRA_Results/dual_lora_best.safetensors")
assert adapter.is_file(),adapter

sidecar=ROOT/"alpha_diffusion/checkpoints/sidecar_best.pt"
if not sidecar.is_file():
    sidecar=Path("/content/drive/MyDrive/H3_AlphaDiffusion/checkpoints/sidecar_best.pt")
assert sidecar.is_file(),sidecar

cmd=[
    sys.executable,"-u",str(R/"tools/compare_sidecar_dual_ab.py"),
    "--vae-lora",str(adapter),
    "--alpha-diffusion",str(sidecar),
    "--rgb-mp4",str(RGB),
    "--output-dir",str(OUT),
    "--sampling-steps","30","--seed","42",
]
manifest=ROOT/"paired_training/manifest.json"
if manifest.is_file():
    cmd+=["--manifest",str(manifest),"--heldout-limit","9"]
else:
    print("WARNING: no paired-source manifest; real H3 visual A/B only")
subprocess.run(cmd,cwd=R,check=True)

summary=json.loads((OUT/"ab_summary.json").read_text())
print(json.dumps(summary.get("unused_holdout",{}),indent=2))
display(Image(filename=str(OUT/"same_sidecar_two_decoders.png")))
```

**Outputs:**

- `same_sidecar_two_decoders.png`: same 5 H3 source frames, original-gray
  alpha vs trained dual-LoRA alpha, both over checkerboard.
- `original_gray_generated_alpha_128.npz` and
  `dual_lora_generated_alpha_128.npz`: matched masks for quantitative
  per-frame analysis.
- `heldout_same_latents.csv`: nine paired real GT clips, with IoU, MAE and
  1-pixel-boundary F1 for baseline, trained-dual, and GT oracle.
- `ab_summary.json`: mean holdout metrics and generated H3 mask coverage
  and temporal alpha-difference metrics.

Interpretation:

- **Oracle high, A and B both low:** sidecar is failing to produce
  appropriate alpha latents from RGB, not just a decoder architecture fault.
- **B substantially better than A:** dual-path VAE LoRA works as trained;
  failure is primarily alpha latent generation/generalization.
- **B worse than A despite high oracle:** latent distribution mismatch with
  trained merger/decoder; investigate sidecar calibration and fine-tuning
  on denoiser-produced latent noise at inference.
- **Oracle low:** even the decoder may struggle on those GT clips.

Since the fox MP4 contains no GT alpha, its A/B is *visual only*. Heldout
paired source mattes give numerical ground-truth accuracy, but they still
may differ from the generated fox distribution.
