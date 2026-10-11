# MiniMax-H3 RGBA research

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
