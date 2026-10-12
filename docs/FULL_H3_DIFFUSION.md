# Full MiniMax-H3 diffusion ↔ RGBA contract (October 2026)

**Status: FULL H3 ALPHA GENERATION = NOT YET IMPLEMENTED.**
**Status: video VAE alpha RECONSTRUCTION = TESTED.**

This project was successful at training a **VAE decoder LoRA**: ~95.53%
mean alpha IoU on 3 repeatedly validated clips, ~79.52% on nine
unused-heldout video clips. Those experiments fed the ground-truth alpha
video into the VAE encoder. That is a genuine and valuable VAE experiment
but is not diffusion video generation.

## Dedicated single-A100-80GB T2VA mode (53GB host RAM)

The standard official BF16 `ComponentsManager.enable_auto_cpu_offload`
recipe is **NOT** appropriate for a 53GB-host-RAM runtime because the
weights exceed available host RAM. The previous `h3_full_diffusion_probe.py`
resource guard was correct **for that BF16/CPU-streaming recipe**, but
incorrect to apply as a universal ban on A100 80GB H3 execution.

For this hardware, a credible alternate execution mode uses:

- A single **A100 80GB (Ampere, sm_80)** with standard PyTorch SDPA
  (do **NOT** enable Hopper FlashAttention3/Sage sm_120 backends).
- **NF4 4-bit Qwen3-VL text conditioner** (about 21GB GPU-resident).
- **TorchAO INT8 H3 video/audio diffusion transformer** (about 34GB).
- Original video/audio VAEs, phase-managed (parked when not needed).
- Phase-specific GPU loading, so host RAM does NOT need to hold 62GB+62GB
  BF16 models simultaneously.
- The independent `animede/Diffusers_minimax-h3` implementation documents
  ~59.7GB peak GPU use in the **80GB-tier** (not independently measured
  by this project on an A100/53GB Colab yet).

Repository references:
https://github.com/animede/Diffusers_minimax-h3/blob/main/README.en.md
https://github.com/animede/Diffusers_minimax-h3/blob/main/core/runner.py

### Reproducible A100 Colab commands

Use a fresh Colab with **A100 80GB**. The external runner has been tested
with **Python 3.12, torch 2.9.0 cu128, diffusers f37ab93, transformers
5.14.1, torchao 0.17.0**. Installing these in a fresh virtualenv
avoids breaking the pinned VAE-only Diffusers environment.

```bash
pip install uv
uv python install 3.12
uv venv --python 3.12 /content/h3_a100_venv
uv pip install --python /content/h3_a100_venv/bin/python \
  'torch==2.9.0' --index-url https://download.pytorch.org/whl/cu128

uv pip install --python /content/h3_a100_venv/bin/python \
  -r requirements-a100-h3.txt

git clone https://github.com/animede/Diffusers_minimax-h3.git \
  /content/Diffusers_minimax-h3

/content/h3_a100_venv/bin/python tools/h3_a100_full_t2va.py \
  --preflight-only

# Only download T2VA's required ~144GB, NEVER the full ~499GB H3 repo.
/content/h3_a100_venv/bin/python tools/h3_a100_full_t2va.py \
  --download-only

# Actually generate real 5-second H3 video + audio via the pretrained 33B
# denoising transformer (not just VAE reconstruction).
/content/h3_a100_venv/bin/python tools/h3_a100_full_t2va.py \
  --generate --height 544 --width 960 --seconds 5 --steps 30
```

This wrapper executes the external runner's `MiniMaxH3Runner.generate()`,
checks that the resulting MP4 decodes at least 124 frames and contains
both video and audio streams, records actual peak GPU VRAM, and **marks
alpha generation as false**. A real generated RGB video is not the
missing separately generated alpha latents. No 500-step decoder-LoRA
training weights are loaded in this first end-to-end baseline.

The upstream 80GB-tier result was not verified on an A100 with ~53GB
CPU RAM in this project; if the actual load runs out of memory,
capture the error and resource stats before modifying placement. Do not
mislabel a synthetic model or a VAE reconstruction as full-H3 success.


## Official full H3 diffusion architecture and resource budget

Official Hugging Face docs:
https://huggingface.co/docs/diffusers/api/pipelines/minimax_h3

- Use `ModularPipeline`, workflow `t2va`, **not**
  `DiffusionPipeline.from_pretrained(...)`.
- Pretrained H3 **video+audio diffusion transformer:** 61.7 decimal GB BF16.
- Pretrained **Qwen3-VL text conditioner:** 62.1 decimal GB BF16.
- Video VAE: 10.416 GB checkpoint; audio VAE + schedulers are shared.
- The official 24–32GB consumer GPU loading recipe requires **torchao int8**
  with 1-block-at-a-time group CPU offload for the transformer and leaf
  offload of the Qwen3-VL conditioner.
- Official estimate: roughly **75 GB system RAM** holding int8 model
  weights. Colab L4 22GB VRAM / ~53GB system RAM from the user's previous
  runtime is insufficient for that documented recipe. CUDA VRAM and CPU
  RAM are NOT interchangeable.
- Original H3 requires 24fps, a minimum ~5 seconds (124 frames), 17*n+5
  frames, and dimensions multiples of 32. Unlike the 17-frame VAE
  reconstruction tests, a 17-frame original DiT generation is unsupported.
  A tiny 192x320 canvas is only for integration debugging, not native
  quality or a faithful H3 benchmark.
- `num_inference_steps=3` includes the terminal sigma=0 and performs
  2 actual denoiser evaluations. Use larger values for genuine image quality.

## Script: actual original H3 RGB/audio generation

`tools/h3_full_diffusion_probe.py` uses the published
ModularPipeline + INT8 group offload recipe. It can:

1. Check available **host RAM**, disk and CUDA without downloading.
2. Load the *actual released* H3 transformer, conditioner, video VAE and
   audio VAE in a high-memory environment.
3. Run text-conditioned H3 denoising, decode video and audio, save MP4.
4. Request the **denoised video latent tensor**, confirm it is
   `[1,24,37,12,20]` for 124 frames at 192x320, and save a run report.
5. Mark `supports_true_rgba_generation: false`. It will NEVER invent an
   alpha mask or pretend a VAE oracle reconstruction is generated video.

```bash
# Safe preliminary check. No H3 transformer or text weights are loaded.
python tools/h3_full_diffusion_probe.py --preflight-only \
  --output-dir /content/h3_real_diffusion_probe
```

To inspect the **actual published H3 t2va ModularPipeline definition**
without downloading the 61.7GB transformer or 62.1GB conditioner weights,
first install a current Modular Diffusers build (e.g. a **separate**
full-H3 environment), then run:

```bash
python tools/h3_full_diffusion_probe.py --workflow-only \\
  --output-dir /content/h3_real_diffusion_probe
```

This checks `ModularPipeline.from_pretrained(..., workflow="t2va")`,
saves `h3_t2va_workflow_doc.txt`, and never calls `load_components`.
On a low-memory runtime it can validate workflow wiring **but not execute
the pretrained denoiser**. Results are labeled accordingly.

For the **true model call**, use an environment with at least ~85 decimal GB
**available** host RAM and ~155 GB free disk, plus BF16 CUDA. Run this in
a separate fresh environment from the VAE training Python environment:

```bash
pip install -r requirements-full-h3.txt
python tools/h3_full_diffusion_probe.py \
  --output-dir /content/h3_real_diffusion_probe \
  --height 192 --width 320 --num-frames 124 \
  --num-inference-steps 3 \
  --prompt "A pastel planet slowly spinning with thin rings, studio lighting"
```

The published code requires int8-compatible `torchao`. **Do not run
`tools/colab_compat.py --repair`**, because that VAE-only compatibility
script conditionally uninstalls torchao. The official weights download is
over 130 decimal GB before compression/storage overhead; budget disk and
internet time accordingly. The probe refuses a low-memory attempt unless
you deliberately pass `--force-resource-override` (risk of OOM, do not
recommend on 53GB host RAM). It will not automatically reroute to a fake
tiny model.

## Why the current VAE LoRA cannot make prompt→transparent video

```text
OFFICIAL original H3 t2va denoiser
  prompt -> Qwen3-VL conditioning
         -> 33B joint video/audio transformer
         -> z_video: [B,24,F,H/16,W/16] + audio latents
                    |
                    v
       standard pretrained RGB video VAE -> RGB video + sound

CURRENT trained dual alpha VAE LoRA (encoder ORACLE experiment)
       z_RGB:   [B,24,F,H/16,W/16]
       z_ALPHA: [B,24,F,H/16,W/16]    <-- encoded from provided GT alpha video
                |
        48->24 residual merger
                |
         alpha-decoder LoRA -> matte
```

The original diffusion model has **no independent z_ALPHA prediction**.
The alpha VAE adapter's two inputs cannot both be obtained from
the original generated z_video, and repeating z_video, filling zeros, or
encoding generated RGB as an alpha stream would be an invalid
generation test.

The strict testable interface is `h3rgba/diffusion_contract.py`:
`validate_two_generated_streams` rejects missing or oracle/GT alpha
streams. `denormalize_h3_video_latents` applies official
`vae.config.latents_std`/`latents_mean` to the normalized denoised
H3 latents BEFORE feeding raw VAE latent coordinates into custom decoding.

Additionally, original H3 generation is >=124 frames whereas the VAE
training used 17-frame `_encode_clip/_decode_clip`. Real full-length
alpha decoding must adapt temporal chunking to H3's official VAE decode
implementation; merely passing 37 latent frames to the old 17-frame
`decode_alpha` helper is unvalidated.

## What actually needs training next

To **generate transparency** for arbitrary text prompts, a diffusion-side
alpha model needs to predict `z_ALPHA` conditioned on prompt and
video evolution. Possible future approaches:

- **Dual denoiser approach:** train an alpha latent denoiser (H3
  transformer LoRA or separate, smaller conditional latent denoiser)
  to predict alpha latents from noisy alpha, RGB latents and conditioning
  tokens. Keep the existing RGB DiT and video VAE frozen.
- **48-channel joint approach:** widen H3 input/output latent projections
  to handle both 24-channel RGB and 24-channel alpha latents, copying
  pretrained RGB weights and initializing new alpha blocks; LoRA-finetune
  appropriate attention/MLP layers and train the new output projections.
  This changes transformer shapes and requires custom H3 modular
  denoising blocks, proper 24+24-channel sigma/noise handling and a
  matched scheduler. Not a drop-in VAE LoRA.
- **RGB-conditioned alpha estimator:** a smaller trainable segmentation
  model generating alpha from final RGB frames could be a practical
  baseline. It is not H3 alpha diffusion and must be labeled honestly.

All choices require **paired text-video-alpha training data** that matches
the diffusion conditioning objectives and a feasible compute/memory plan.
A 53GB-RAM, 22GB-VRAM single Colab L4 cannot be assumed to run 33B H3
diffusion fine-tuning. Tiny synthetic CPU tests establish contracts only.

## Automated tests

```bash
pytest -q tests/test_full_h3_pipeline_cpu.py
```

These verify 124-frame geometry, 24-channel latents, normalization,
host RAM/disk guard, and strict rejection of missing alpha diffusion
output. They **do not** execute the pretrained 33B transformer.
Running real generation is a separate manual high-memory test.
