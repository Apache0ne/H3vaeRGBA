# MiniMax-H3 RGBA research

## Current native H3 Wan-style joint RGB+alpha training path (experimental)

**Entry point:** [One-cell A100 training and validation](docs/RUN_NATIVE_H3_WAN_RGBA_A100.md) or `tools/launch_h3_wan_native_a100.py`.
This is the **single 24-channel joint-latent** approach: a new trained H3 RGB+alpha
VAE bridge followed by a **real H3 33B diffusion LoRA** on fused latents, then
native H3 generation and same-latent RGB+alpha decoding. It does **not** use
the separate 4,000-step alpha diffusion sidecar.

The launcher runs real Qwen3-VL caption conditioning, hash-verified latent
caches, optimizer checkpoints with matching resume signatures, a two-step GPU
smoke before long training, RGBA PNG + ProRes4444/audio output, and an
independent nine-clip held-out VAE evaluation. The H3 DiT velocity convention
is data-ward and its video-noise distribution uses H3's shift=12 schedule.

**Verification:** CPU regression testing exists. Full A100 joint VAE training,
33B backpropagation, native generation and alpha fidelity are **not yet
verified** by the repository's CPU tests. Do not interpret a MOV export or
held-out *VAE reconstruction* score as a measured novel-video alpha score.

The older stages below are retained as research history and do not replace
this current two-stage experiment.

### Completing a full Colab run without GPU OOM or losing progress

The user-measured 2-step BF16 dual-path training used **6.81 GiB peak reserved VRAM** on a **22.5 GiB** GPU, so continuing at the SAME 128x128/rank-4 configuration does **not** require 8-bit weights, model offload, or decreasing quality. 128px yields strong step-zero alpha IoU **0.86064** / MAE **0.04246**; step two IoU **0.86072**, so use best-validation LoRA rather than assuming more steps improve it.

The trainer supports optional `--backup-dir /content/drive/MyDrive/H3_DualLoRA_backups` after mounting Google Drive through Colab. At every saved step it writes a new `step_0000XX/` snapshot containing the latest adapter, optimizer and metrics, then creates a `COMPLETE.json` marker **last**, retaining three newest complete snapshots. Recover by copying files from the newest COMPLETE snapshot back to the original `--output-dir`; restore dataset separately if a Colab runtime reset removed it.

New optional `--max-runtime-minutes 150` performs a graceful validation/checkpoint/backup when that training-time budget is reached; rerun the same command with `--resume` until the requested total steps are reached. Both options are excluded from the model-optimizer settings signature and can be changed safely on resume.

Never launch two trainers into the **same** output directory concurrently. The GPU memory screenshot showing idle 0.2 GiB is a momentary utilization measurement, not the peak allocation from training.


## Stage W — actual decoder-LoRA training inspired by Wan-Alpha and AlphaVAE (October 2026)

**This replaces the unsuccessful original RGB->RGBA shared-latent VAE training experiment, without overwriting its checkpoints.** Measured original vs widened VAE RGB MAE 0.080935 vs 0.116372 with alpha IoU 0.448. RGB-preservation training reduced RGB MAE but left alpha IoU only 0.465 at 100 steps. More shared-latent weighting is not enough.

### What the cited projects actually do

- [Wan-Alpha VAE training code](https://github.com/WeChatCV/Wan-Alpha/blob/main/Wan-Alpha-VAE-train/train_vae.py): pretrained VAE copies for foreground and alpha, **decoder convolution LoRAs**, alpha and foreground encoders, trainable latent mapping, compositing (soft and hard random backgrounds), Sobel, VGG, and T2V diffusion loss.
- [AlphaVAE training code](https://github.com/o0o0o00o0/AlphaVAE/blob/main/train/train_vae.py): blends RGBA over backgrounds to encode, reconstruction loss defined on background compositing, optional perceptual/KL terms. Full VAE fine-tuning, not merely alpha head.
- [FLUX.2 Klein Alpha](https://github.com/TR-MZ/flux2-klein-alpha): separate finetuned RGBA VAE and DiT extraction/removal LoRAs; latter train the 4B/9B image generation transformer, **not** the VAE.
- H3 is different: [its VAE has a 36-layer 2048-wide transformer decoder](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/vae/config.json), not Wan's convolutional decoder. We adapt the architectural **principles** but use H3 decoder **attention/MLP Linear LoRA**.

### Stage W architecture

```text
Ground truth RGBA video  [B,4,17,128,128]
       |                              |
       | foreground RGB (3ch)         | alpha copied to RGB grayscale (3ch)
       v                              v
FROZEN official H3 encoder       SAME FROZEN official H3 encoder
       | 24ch RGB latents             | 24ch alpha latents
       +--------------+---------------+
                      |
             zero-init residual latent merger 48->24
                      |
          pretrained H3 36-layer decoder
             with trainable rank-4 LoRA
                      |
          trainable grayscale output head 3->1
                      |
              predicted alpha [B,1,17,H,W]

RGB reconstruct: RGB latents -> ORIGINAL H3 decoder with LoRA DISABLED
                              -> RGB, unchanged after any alpha training
Combine the original RGB and learned alpha only for previews.
```

The merger is zero-initialized, so at step 0 the alpha path is an original RGB video VAE encoding/decoding a grayscale alpha video. Decoder LoRA starts as identity. All original encoder/decoder parameters remain frozen; LoRA, merger and head are the only trainable tensors. The model loads the original 10.416 GB checkpoint **once**, and it compares in-memory pretrained weights against the actual three safetensors shards.

**Limitations:** This is a supervised **video VAE alpha reconstruction LoRA**, not an H3 33B DiT LoRA. Alpha encoding presently requires a supplied alpha stream. Without subsequently training H3's video diffusion model on a compatible combined RGB/alpha latent distribution, **the resulting VAE does not give text-to-transparent-video generation**. Also, 128x128 yields only an 8x8 spatial latent grid; evaluate detail loss before raising resolution.

### Full 22 GiB Colab command (existing dataset + pretrained VAE cache)

```bash
git pull --ff-only
python tools/colab_compat.py --repair
python train_wan_dual_lora.py --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/wan_dual_h3 \
  --skip-download --resolution 128 --rank 4 \
  --steps 2 --val-every 1 --save-every 1
# Examine step-zero original alpha encoding baseline, step 1, step 2,
# alpha_iou, alpha_mae, rgb_visible_mae_original_h3 and peak GPU VRAM.

python train_wan_dual_lora.py --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/wan_dual_h3 \
  --skip-download --resolution 128 --rank 4 \
  --resume --steps 500 --val-every 25 --save-every 50
```

`--steps` is the **total** step count, including resumed steps; keep other loss/hyperparameter flags unchanged. Output:
`checkpoints/dual_lora_last.safetensors`, `checkpoints/optimizer_last.pt`,
`metrics.jsonl`, `validation.jsonl`, `previews/step_000000/contact_sheet.png`,
`previews/step_000002/contact_sheet.png`, etc.
Check saved checkpoint against multiple held-out examples and soft-alpha edges, not just train loss. Stage W has been CPU-tested using small instantiated H3 models; real H3 on 22 GiB must be validated using 2 steps first.


## Stage 3: preserve the pretrained RGB path while learning alpha

Observed ablation on a held-out planet clip:

| Metric | True alpha supplied | Added alpha input neutralized |
|---|---:|---:|
| Visible RGB MAE (lower better) | 0.116372 | 0.080935 |
| Alpha IoU (higher better) | 0.448150 | 0.152357 |
| Alpha MAE (lower better) | 0.433365 | 0.502688 |

Original H3 RGB visible MAE was 0.080974. The alpha information is reaching latents (absolute mean delta 0.742271, relative delta 0.292007), but corrupts the RGB reconstruction. **Do not repeat the previous loss alone.**

New optional training flags in `run_vae_22gb.py`:

- `--rgb-truth-weight`: weighted visible-pixel RGB error against source frames, normalized by visible pixel area.
- `--rgb-teacher-weight`: RGB deviation from a **no-grad frozen-original-H3 teacher**, computed using the same widened model with the extra alpha input zeroed and all decoder LoRAs temporarily bypassed.
- `--latent-shift-weight`: relative latent L1 penalty, intended as a **small stabilizer**, not a hard requirement to ignore alpha.

The original visual VAE is still loaded and verified. Stage 3 adds no extra model copy, but the teacher requires another no-gradient encoder and decoder forward per training/validation sample. All pretrained RGB weight slices stay gradient-frozen. The original Stage-1/Stage-2 runs are unchanged.

**Start with a 2-step GPU memory/gradient experiment in a NEW output directory**, initialized from the verified Stage-1 100-step alpha checkpoint. This is a controlled experiment and **not yet validated on a real 22 GiB GPU**:

```bash
python run_vae_22gb.py --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/vae22_rgb_preservation \
  --local-files-only \
  --init-alpha /content/h3rgba_runs/vae22_fresh_verified/checkpoints/alpha_last.safetensors \
  --decoder-lora-rank 4 --resolution 128 --batch-size 1 \
  --alpha-loss-weight 5 --edge-loss-weight 4 \
  --rgb-truth-weight 2 --rgb-teacher-weight 3 --latent-shift-weight 0.2 \
  --lr 5e-5 --steps 2 --val-every 1 --save-every 1
```

Inspect `peak_alloc_gib`, `visible_rgb_mae_measured`, `rgb_teacher_mae`,
`relative_latent_shift`, `alpha_iou` and the contact sheet.
To keep the baseline RGB MAE near ~0.081 while retaining alpha IoU near ~0.448
would already be a meaningful improvement on this one clip. Improvement on one
sample is not evidence of generalization: use the entire held-out set and
multiple types of transparency before moving to DiT LoRA.

If successful, continue using `--resume --steps 100` and the *same* penalty
flags, output directory, and LoRA rank. The runner explicitly rejects changing
these objective settings when resuming an optimizer checkpoint.


## Diagnose original H3 VAE vs transparent VAE before more GPU training

A 100-step clean alpha-only rerun reproduced blurred RGB and alpha masks at 128×128. This is **not evidence that the pretrained checkpoint was skipped**: the new tensor verifier confirms actual pretrained weights. At 128px, H3's 16× spatial compression makes only an 8×8 latent grid. A visible blurred ring can therefore originate in the **original RGB VAE** as well as in the new alpha branch.

Use the new read-only comparator, which loads the official 10.4 GB pretrained VAE only once, verifies the weights, reconstructs a held-out 17-frame RGB clip with the original 3-channel architecture, then widens **that same VAE** and applies the saved alpha adapter to reconstruct exactly the same clip:

```bash
git pull --ff-only
python diagnose_vae.py \
  --workdir /content/h3rgba_data \
  --alpha-checkpoint /content/h3rgba_runs/vae22_fresh_verified/checkpoints/alpha_last.safetensors \
  --resolution 128 --output-dir /content/h3rgba_runs/vae_diagnosis
```

The output `sample_00_comparison.png` now has **six** columns: *target on black*, *original pretrained H3 RGB*, *adapted RGBA on black*, *adapted RGBA on white*, *adapted alpha matte*, and *adapted matte with alpha input zeroed*. `metrics.json` reports original/adapted visible RGB MAE, alpha MAE/IoU, `alpha_input_latent_mae_delta`, `alpha_zeroed_iou` and `rgb_with_vs_without_alpha_input_mae`. If zeroing the alpha channel hardly changes alpha IoU, the model may be guessing the mask from RGB instead of conveying new alpha information through the latent bottleneck. If the ablation destroys mask quality, the alpha signal is encoded but the decoder has trouble recovering a sharp silhouette. This test never updates weights. No training, optimizer, or DiT is used. Test at 128 first; 192 or 256 can be investigated only after measuring VRAM.

**Do not train longer blindly.** If original H3 RGB looks badly blurred too, focus on native resolution, cropping and latent detail before adding VAE LoRA steps. If original RGB is sharp but adapted RGB is blurry, inspect the alpha encoder channel and latent shift before increasing decoder capacity.


## Proven pretrained VAE loading and optional clean Stage-1 rerun

**Do not mistake cache-file-listing filters for model loading failures.** The official VAE has three shards totaling 10.416 GB and 703 indexed tensors. The training runner already uses `AutoencoderKLMiniMaxH3.from_pretrained(..., subfolder="vae")`. The new code now additionally compares loaded encoder/decoder **tensor values** against the original safetensors BEFORE widening the model. This ensures the original pretrained checkpoint is not being skipped.

```bash
git pull --ff-only
python tools/colab_compat.py --repair
# Load and verify the real pretrained weights, WITHOUT training
python run_vae_22gb.py --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/check_pretrained \
  --local-files-only --verify-weights-only

# Optional clean Stage-1 comparison, NOT needed to fix the false cache reading.
# Use new directory. This does NOT load prior alpha adapters or LoRA weights.
python run_vae_22gb.py --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/vae22_verified_restart \
  --local-files-only --decoder-lora-rank 0 \
  --resolution 128 --batch-size 1 --steps 2 --val-every 1 --save-every 1
python run_vae_22gb.py --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/vae22_verified_restart \
  --local-files-only --decoder-lora-rank 0 --resume \
  --resolution 128 --batch-size 1 --steps 100 --val-every 25 --save-every 25
```

Look for `OFFICIAL_H3_PRETRAINED_WEIGHTS_MATCH_IN_MEMORY` and `OFFICIAL_H3_VAE_WEIGHTS_VERIFIED_IN_MEMORY`; inspect `pretrained_weight_verification.json` in each output directory. If tensor comparison fails, no RGBA training starts. `--local-files-only` ensures that cached H3 weights are used without re-downloading. An existing training checkpoint cannot be overwritten accidentally: use a new output directory or `--resume`.


## Improved alpha-edge training (optional decoder LoRA)

The initial alpha-slice-only stage preserves RGB but may produce soft masks (e.g. ringed planets becoming gray blobs). New **Stage 2** keeps the original 24-channel latent interface and trains rank-4 LoRA on the visual decoder's q/k/v/o and MLP layers, plus the original alpha input/output slices. No 33B DiT is loaded.

Start by reusing your existing Stage-1 adapter; **do not resume its optimizer**, because the Stage-2 optimizer has extra LoRA parameters. Use a **new output directory**:

```bash
python run_vae_22gb.py \
  --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/vae22_lora4 \
  --init-alpha /content/h3rgba_runs/vae22/checkpoints/alpha_last.safetensors \
  --decoder-lora-rank 4 \
  --resolution 128 --batch-size 1 --grad-accum 1 \
  --alpha-loss-weight 5 --edge-loss-weight 4 \
  --lr 5e-5 --steps 2 --val-every 1 --save-every 1
```

Check GPU peak allocation in the output and inspect `previews/step_000002/contact_sheet.png` before continuing. If VRAM fits and the gradients are finite, continue with the **same output directory**:

```bash
python run_vae_22gb.py \
  --workdir /content/h3rgba_data \
  --output-dir /content/h3rgba_runs/vae22_lora4 \
  --decoder-lora-rank 4 --resolution 128 --batch-size 1 --grad-accum 1 \
  --alpha-loss-weight 5 --edge-loss-weight 4 \
  --lr 5e-5 --resume --steps 300 --val-every 25 --save-every 25
```

Stage 2 saves `checkpoints/alpha_last.safetensors` and `checkpoints/decoder_lora_last.safetensors`, plus resumable optimizer state and validation `alpha_iou`/`alpha_mae`. LoRA checkpoints are **VAE decoder** adapters, NOT text-to-video DiT LoRAs. Watch alpha IoU and real composite quality; simple decreasing loss is insufficient.

**Memory caveat:** Stage 2 with the full pretrained H3 VAE hasn't been executed on a real 22 GiB GPU. Gradient checkpointing and rank-4 are conservative, not proof that it will fit. No GPU memory claim is established by CPU tests. If OOM, stop, lower resolution to 96 in a new output directory or use more capable hardware.

## Colab import fix: incompatible optional TorchAO

If Python reports `ImportError: cannot import name 'FqnToConfig' from 'torchao.quantization'`, run this **before** importing Diffusers:

```bash
python tools/colab_compat.py --repair
```

This checks TorchAO using a separate subprocess, uninstalls it **only when broken**, and verifies `from diffusers import AutoencoderKLMiniMaxH3`. Our BF16 VAE workload does not require TorchAO. If the current notebook kernel has already imported either package, restart the kernel first (or launch the trainer as a new Python process). This is an optional-library dependency mismatch, not a H3 architecture failure.

## Single-GPU 22 GiB VAE workflow

`run_vae_22gb.py` runs **only the H3 visual VAE**: paired Wan-Alpha videos -> audit -> BF16/FP32 mixed reconstruction training -> alpha-only safetensors -> RGBA PNG previews + validation metrics. It never downloads or trains the H3 33B diffusion transformer.

**Google Colab GPU:** select an L4 or another BF16-capable GPU with about 22 GiB VRAM.

```bash
git clone https://github.com/Apache0ne/H3vaeRGBA.git
cd H3vaeRGBA
pip install -r requirements.txt
python tools/colab_compat.py --repair
python run_vae_22gb.py --resolution 128 --steps 100 --val-every 25 --save-every 25 \
  --workdir /content/h3rgba_data --output-dir /content/h3rgba_runs/vae22
```

The script automatically downloads ~1.08 GB of paired compressed videos and loads ~10.4 GB of VAE checkpoint weights into the HF cache (if not already cached). It uses 17-frame clips at 128x128, batch=1, H3's FP32 encoder and BF16 decoder autocast, gradient checkpointing and AdamW on the added alpha weights only. The script has CPU checks but a complete full-checkpoint GPU training pass is **not yet validated**. Even at 128px 22 GiB memory is a target, not a guarantee; stop or drop to 96px if the GPU reports OOM.

For a smaller end-to-end GPU smoke test, use `--steps 2 --val-every 1 --save-every 1`. Check the `previews/step_*/contact_sheet.png` image and `validation.jsonl` against `alpha_step_*.safetensors`. Alpha quality cannot be assumed from a decreasing total loss alone.

If the notebook disconnects, save `--output-dir` on mounted Google Drive and rerun with `--resume --steps 500` (same directory and resolution). `--eval-only` loads `checkpoints/alpha_last.safetensors` and writes held-out visual predictions.

**Limitation:** Training only the widened 4-channel encoder input and alpha output projection may underfit alpha transparency. Further VAE adapters and H3 DiT LoRA fine-tuning are separate development stages; no promise of transparent generative output is made at this stage.


**Status:** Stage-1 RGBA-VAE implementation and offline structural tests. **Not GPU-validated or production-ready.** Do not rent an A100 based on these files alone.

Design: widen the causal-video encoder first convolution from RGB (3 channels) to RGBA (4), and widen the ViT decoder output projection from 3 to 4 channels per spatial/temporal patch. The existing 24-channel latent interface stays unchanged. Existing RGB weights are preserved; new alpha weights are zero-initialized. This is a checkpoint-compatible *initialization*, not a trained transparent-video model.

**Training order:**
1. Paired foreground and alpha videos; validate alignment and frame counts.
2. Train RGB-to-RGBA H3 VAE on 17-frame clips (VAE-native chunk length) with BF16 autocast and gradient checkpointing. Preserve RGB using frozen original projection rows and loss to original reconstruction.
3. Evaluate alpha MAE, edges, temporal differences and RGB composites. Train selective VAE adapters if the narrow-only version underfits.
4. Only after satisfactory VAE round-trips, train H3 DiT LoRA on the RGBA VAE's *normalized 24-channel latents* using the original H3 audio/video training infrastructure. The DiT training code expects frames **17n+5** (22, 39, ...). Its original RGB VAE encode path must be explicitly replaced: swapping the VAE file alone is insufficient.
5. Add RGBA decoding and alpha-preserving export; ordinary MP4 will discard alpha.

Sources:
- https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/vae
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py
- https://github.com/WeChatCV/Wan-Alpha
- https://github.com/TR-MZ/flux2-klein-alpha
- https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/minimax_h3/model_training

### Install (CPU preflight, no model download)
```bash
git clone https://github.com/Apache0ne/H3vaeRGBA.git
cd H3vaeRGBA
python -m pip install torch pytest
python -m pytest tests -q
```

### Dataset audit
```bash
python -m pip install opencv-python-headless
python tools/dataset_audit.py --foreground /path/data_fgr --alpha /path/data_pha --output manifest.json
```
For automatic download (~1.08 GB compressed), safe extraction and audit (all CPU):
```bash
python -m pip install -r requirements.txt
python tools/prepare_data.py --workdir ./wan_alpha_data
python train_vae.py --manifest ./wan_alpha_data/manifest.json --dry-run
```
Use paired data from https://huggingface.co/datasets/htdong/Wan-Alpha-dataset ; inspect archive samples and licenses before reuse.

### Experimental VAE train (GPU required, *not yet validated on A100*)
This runner uses the upstream **Diffusers** H3 VAE. It intentionally starts with small frames and 17-frame clips. Install a Diffusers revision containing `AutoencoderKLMiniMaxH3` and HF auth if necessary.
```bash
python -m pip install -r requirements.txt
python train_vae.py --manifest manifest.json --output outputs/vae_trial --resolution 128 --max-steps 10
```
**Do not launch the above training until all shape/precision/memory checks succeed in a disposable GPU environment.** It downloads a large VAE. `--dry-run` audits the manifest without loading the checkpoint.

### FP32 / BF16 policy
BF16 GPU autocast; trainable master weights retained as loaded by Diffusers. Encoder/norm modules retained FP32 if upstream pins them. Loss calculated in FP32. No blanket `.half()` or `.bfloat16()` cast over all VAE modules.

### Limitations
- Stage-1 trainer exercises the VAE-only RGB/alpha reconstruction path; diffusion/LoRA generation training is **not implemented** yet.
- A 24-channel latent tensor shape match is necessary but not sufficient for H3 diffusion compatibility: data distribution shift requires testing and DiT LoRA.
- The 17-frame VAE training samples are not interchangeable with H3's 22-frame diffusion training samples.
- RGB teacher baseline and comprehensive real-checkpoint round-trip tests still need implementation.
- Source/model/dataset usage is subject to their respective licenses. No model weights or datasets are committed here.

## Data and LoRA compatibility warning
The Wan-Alpha paired foreground/matte archive is a starting VAE dataset, **not a verified text-captioned H3 diffusion training corpus**. Text-to-transparent-video LoRA training needs prompt-aligned clips, high-quality alpha, varied transparency types, and a held-out test set. The submitted DiffSynth-Studio archive includes H3 LoRA training scripts and BF16 mode, but its training cache pipeline currently uses an RGB VAE. It must be adapted to cache the new RGBA 24-channel latents and respect H3 17n+5 frame lengths. The 33B DiT's BF16 weights alone need roughly 66 GB; an A100 80GB likely needs frozen-weight quantization/offload and cached text/audio conditions for LoRA training. BF16 *compute* is compatible with quantized frozen weight storage. This phase is NOT yet implemented or validated.
