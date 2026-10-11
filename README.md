# MiniMax-H3 RGBA research

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
Use the paired foreground/alpha video data in https://huggingface.co/datasets/htdong/Wan-Alpha-dataset . Audit licenses before reuse.

### Experimental VAE train (GPU required, *not yet validated on A100*)
This runner uses the upstream **Diffusers** H3 VAE. It intentionally starts with small frames and 17-frame clips. Install a Diffusers revision containing `AutoencoderKLMiniMaxH3` and `peft`/HF auth if necessary.
```bash
python -m pip install -U 'diffusers[torch]' transformers accelerate huggingface_hub safetensors opencv-python-headless numpy
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
