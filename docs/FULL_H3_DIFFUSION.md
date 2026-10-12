# Full MiniMax-H3 diffusion ↔ RGBA contract (October 2026)

**Status: FULL H3 ALPHA GENERATION = NOT YET IMPLEMENTED.**
**Status: video VAE alpha RECONSTRUCTION = TESTED.**

This project was successful at training a **VAE decoder LoRA**: ~95.53%
mean alpha IoU on 3 repeatedly validated clips, ~79.52% on nine
unused-heldout video clips. Those experiments fed the ground-truth alpha
video into the VAE encoder. That is a genuine and valuable VAE experiment
but is not diffusion video generation.

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
