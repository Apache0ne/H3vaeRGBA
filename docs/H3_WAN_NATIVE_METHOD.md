# Native Wan-Alpha method port for MiniMax-H3 — 2026-10-12

**Status: Experimental implementation. CPU regression tests only so far; full A100 training and RGBA quality NOT verified.**

The previous 4,000-step alpha sidecar was not the Wan-Alpha method. It generated an independent alpha latent after H3 had already produced RGB; nine-clip sidecar alpha IoU was only 0.2045.

Upstream source:
- https://github.com/WeChatCV/Wan-Alpha/blob/main/Wan-Alpha-VAE-train/train_vae.py
- https://github.com/WeChatCV/Wan-Alpha/blob/main/Wan-Alpha-train/examples/wanvideo/model_training/train_gauss_ellipse.py
- https://github.com/WeChatCV/Wan-Alpha/blob/main/readme.md

Wan-Alpha has TWO trainable parts: a VAE mapping paired RGB and alpha encodings to **one shared native latent**, then a DiT LoRA trained on that joint distribution. This H3 adaptation implements both stages using original H3 24-channel latents, avoiding any separate alpha sampling sidecar.

## Stage 1 — joint RGB+alpha H3 VAE

Files: h3rgba/wan_joint_h3.py, train_h3_wan_joint.py.

From paired source videos, the frozen original 3-channel H3 VAE produces two 24-channel latents (foreground and alpha). A trainable fuser produces one 24-channel joint latent. The original H3 decoder reconstructs RGB from it; the existing high-quality 500-step dual VAE decoder LoRA, aided by a learned alpha latent bridge, reconstructs alpha **from that same single latent**. Both are trained using reconstruction, alpha, edges, black/white/gray compositing, temporal, and latent-prior losses. The native 24-channel H3 interface remains unchanged.

The dual-path baseline was 0.9553 IoU on three repeatedly validated clips and 0.7952 on nine unused clips, but those metrics came from **two input streams**. They must not be quoted as joint-latent results. New validation is mandatory.

Unlike the Wan VAE script, H3 stage 1 uses a normalized-latent prior surrogate rather than backpropagating the frozen 33B diffusion loss during VAE training, and omits VGG perceptual loss. This is a structural/mathematical port, not an identical computation.

## Stage 2 — real H3 transformer LoRA on the new joint latent distribution

Files: tools/import_wan_alpha_captions.py, tools/cache_h3_wan_joint_targets.py, tools/cache_h3_wan_text_embeddings.py, h3rgba/wan_h3_flow.py, train_h3_wan_dit_lora.py.

The official Wan-Alpha T2V training CSV already contains real Chinese-language prompts for the source video clips. A tool matches them using exact video basenames, refuses missing captions, and preserves nine never-validated holdouts.

The trained joint encoder exports original H3-compatible normalized fused 24-channel latents. The actual frozen Qwen3-VL-32B conditioner supplies 5120-dimensional embeddings for the prompts. The int8 H3 33B transformer receives its official video/text rotary packed layout and trains attention/MLP LoRA.

MiniMax-H3's **data-ward** rectified flow target is:

  z_t = t*z_joint + (1-t)*epsilon; prediction target = z_joint - epsilon.

The audio row pathway has no supervision in the 17-frame WanAlpha matting clips, so the stage-2 port trains video-only loss. The native H3 audio pathway and 124-frame performance require a real A100 test. Quantized weight backward compatibility is also unverified.

## Native H3 RGBA inference (no sidecar)

Files: tools/generate_h3_wan_joint.py, tools/launch_h3_wan_native_a100.py.

The full H3 runner loads the trained 33B transformer LoRA and generates a **single 24-channel joint latent**. The script captures the denoised normalized latent *before* the stock RGB decoder. A new subprocess loads the pretrained H3 VAE and trained joint bridge and decodes that same native latent to RGB and alpha; it exports RGBA PNG frames and ProRes4444 MOV with original H3 audio. No GT alpha or separate alpha latent network is used.

## Colab preflight, before large A100 training

Mount Drive. Use the existing paired manifest and old validated VAE LoRA. Then:

    git -C /content/H3vaeRGBA pull --ff-only
    python -u /content/H3vaeRGBA/tools/launch_h3_wan_native_a100.py --preflight-only

Preflight downloads only the small official Wan T2V CSV, audits caption coverage, and loads no 33B weights. If captions are missing, it names the clips rather than inventing prompts.

## Full experimental run

Only after preflight succeeds:

    python -u /content/H3vaeRGBA/tools/launch_h3_wan_native_a100.py --vae-steps 1000 --dit-steps 1000 --dit-rank 8 --height 544 --width 960 --seconds 5

The launcher uses 2-step GPU smoke tests before continuing each full training stage. It prints phase markers and uses 30-second live subprocess heartbeat logging. Drive outputs:

- MyDrive/H3_WanJoint_Native/joint_vae/ — new shared VAE bridge and dual decoder weights plus resumable optimizer checkpoint.
- MyDrive/H3_WanJoint_Native/joint_dit/ — real H3 DiT LoRA checkpoints.
- MyDrive/H3_WanJoint_Native/generated/wan_joint_rgba.mov — final alpha-capable video if all GPU stages succeed.

**Limitations:** These stages have NOT been trained end-to-end on the user's A100. Neither final mask accuracy nor full H3 diffusion LoRA weight-loading has been confirmed. Do not spend the full 1,000-step 33B budget until stage-1 single-latent alpha and RGB are visibly good, including previously unused holdout mattes. A successful MOV export alone is not proof of correct transparency.
