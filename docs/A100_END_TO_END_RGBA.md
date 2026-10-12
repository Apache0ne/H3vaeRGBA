# Experimental end-to-end RGB + separately generated alpha video on A100 80GB

**This is a two-diffusion-model design**. It does **not** apply a LoRA to
H3's 33B DiT, and it is **not** yet validated to generate accurate alpha
for arbitrary real H3 video. Use the actual A100 GPU integration run and
judge images before claiming production-quality transparent video.

The previous "full H3 inference" script only generated RGB+audio. It
never used the trained dual-alpha VAE LoRA. The new orchestrator
`tools/launch_h3_rgba_a100.py` attempts a complete sequence:

1. Download official MiniMax-H3 original text-to-video+audio components
   (only T2VA subset, ~144 decimal GB), run **real 33B H3 diffusion**
   via A100-80GB INT8/4-bit phased loader. Produce original RGB+audio MP4.
2. Restore the validated `dual_lora_best.safetensors` from Drive.
   If missing, download Wan-Alpha paired FG/alpha clips + official VAE,
   train rank-4 decoder-LoRA for 500 optimizer steps again.
3. Train a **new, separate RGB-conditioned 3D alpha latent DDPM**
   with a frozen original H3 VAE (precompute encoded RGB and matte latent
   pairs for supervised train/validation). Save `sidecar_best.pt`.
   It uses true DDPM corruption with random timesteps, and sampling starts
   from independent Gaussian noise using 30-step DDIM.
4. Read generated RGB MP4 (never read ground-truth matte on inference).
   For each overlapping 17-frame window:
   RGB frames -> H3 frozen RGB encoder -> normalized `z_rgb`;
   sidecar DDIM -> independently sampled 24-channel `z_alpha`;
   inverse H3 latent normalization -> trained decoder-LoRA alpha reconstruction.
5. Blend overlapping windows, upsample masks from 128px, export actual
   **RGBA PNG sequence** and **ProRes4444 alpha MOV**, muxing source H3 audio.
   Verify MOV pixel format is `yuva444p10le`.
6. Save report and final MOV to Drive, with checkpoint hashes and whether
   alpha really came from DDIM noise.

### Single Colab cell (A100 80GB)

```python
import sys, subprocess
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")

REPO = Path("/content/H3vaeRGBA")
def run(args):
    print(">>>", *map(str,args),flush=True)
    subprocess.run(list(map(str,args)),check=True)

if not REPO.exists():
    run(["git","clone","https://github.com/Apache0ne/H3vaeRGBA.git",REPO])
else:
    run(["git","-C",REPO,"pull","--ff-only"])

run([sys.executable,"-u",REPO/"tools/launch_h3_rgba_a100.py",
     "--run-root","/content/h3_rgba_end_to_end",
     "--drive-root","/content/drive/MyDrive",
     "--height","544","--width","960","--seconds","5",
     "--h3-steps","30","--alpha-steps","4000",
     "--seed","42"])
```

To make the script stop after preflight, add `--dry-run`; this is **not**
an end-to-end training or generation test.

### Key limits / why a successful MOV is NOT enough

- **Checkpoint prerequisites:** Previous 500-step VAE adapter should be at
  `/content/drive/MyDrive/H3_DualLoRA_Results/dual_lora_best.safetensors`
  or in complete Drive snapshots `H3_DualLoRA_Backups/`. Source on GitHub
  is not a substitute for trained adapter weights.
- **Full 33B H3 quantized inference** remains untested by this repo on
  an A100 with ~53GB of system RAM. Only CPU code tests exist. Phase-managed
  80GB-tier external runner documented a ~59.7GB VRAM peak; actual
  host RAM peaks and A100 execution must be measured.
- **Alpha sidecar is a second small diffusion model**, NOT H3's joint
  text-conditioned 33B DiT LoRA. It operates after H3 produces RGB.
  If user needs joint native H3 diffusion RGBA generation, that is still
  future work requiring RGB+alpha joint latents and matched DiT training.
- **Alpha accuracy:** Training dataset is only ~391 clips and the sidecar
  has not been evaluated on real H3 output against independent alpha GT.
  A high IoU for VAE reconstruction (0.955/0.795) does not transfer
  automatically to alpha prediction from RGB. DDPM validation here compares
  estimated alpha against the frozen VAE's teacher reconstruction on 3 clips.
- **Compositing ambiguity:** H3-generated RGB frames can contain a real
  rendered background. A transparent alpha mask is not enough to recover
  the unseen foreground's original RGB at semitransparent edges. The
  script optionally divides by alpha assuming a **uniform black**
  background; that is an approximation and can produce color fringes.
  Use strong prompts for isolated subject on black background.
- **Spatial quality:** The alpha diffusion model is trained at
  128×128 with 17-frame windows, then its matte is upsampled to H3 video
  dimensions, potentially losing fine fur/hair/glass. Separate larger
  resolution training/validation is required to improve this.
- **Audio:** H3 generates audio natively. We preserve its audio by muxing
  that stream with ProRes4444 (not a transparent MP4). Video players differ
  in support for displaying alpha and audio in ProRes MOV.
- **Temporality:** H3 original RGB denoising emits 124+ frames; the
  17-frame alpha VAE was never validated with full 37-latent decoding.
  We use overlapping fixed 17-frame windows to avoid passing incompatible
  124-frame streams directly to the alpha VAE; chunk seams must be examined.

### Tests

```bash
pytest -q tests/test_alpha_sidecar_cpu.py
pytest -q tests/test_a100_full_pipeline_cpu.py
```

Tests cover DDPM forward/backward, seeded DDIM sampling, 24ch H3 VAE latent
normalization, full 124-frame temporal coverage with overlaps, RGBA PNG
alpha, black-background uncompositing and A100 preflight. Tests do **not**
run the full 33B generator, load the ~10.4GB pretrained VAE in the GPU, or
verify actual generated alpha quality.

The full user-facing milestone is a generated MOV that retains alpha,
whose masks also score well against truly unseen paired video mattes.
Until the latter condition is established, classify this as an
**experimental end-to-end integration**.
