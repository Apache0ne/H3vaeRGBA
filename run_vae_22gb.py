#!/usr/bin/env python3
"""22 GiB GPU: end-to-end *VAE* stage for MiniMax-H3 RGB -> RGBA.

Downloads paired videos, audits them, loads ONLY H3's visual VAE, expands
its 3-channel input/output without changing 24 latent channels, runs BF16
reconstruction training, logs holdout validation, saves reusable alpha-only
safetensors and RGBA PNG frame previews.

NO H3 33B diffusion transformer, audio VAE or text-to-video LoRA is loaded.
Starting default: batch 1, 17 frames, 128x128, alpha-slices-only training.

This has passed CPU structural tests; full pretrained BF16 training requires
a real CUDA GPU and must be judged by the saved PNGs and metrics.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--workdir", default="/content/h3rgba_data",
                   help="Dataset root including manifest and extracted pairs")
    p.add_argument("--manifest", default=None, help="Optional pre-audited paired-video manifest")
    p.add_argument("--skip-download", action="store_true", help="Expect already downloaded and audited data")
    p.add_argument("--output-dir", default="/content/h3rgba_runs/vae22")
    p.add_argument("--hf-model", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--decoder-lora-rank", type=int, default=0,
                   help="Rank 4 or 8 for decoder attention/MLP LoRA; 0 = original alpha-only stage")
    p.add_argument("--init-alpha", default=None,
                   help="Initialize from prior Stage-1 alpha_last.safetensors (fresh optimizer, new output dir)")
    p.add_argument("--alpha-loss-weight", type=float, default=3.,
                   help="Alpha pixel loss weight, original default 3")
    p.add_argument("--edge-loss-weight", type=float, default=1.,
                   help="Alpha Sobel-like spatial edge loss weight, original default 1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-samples", type=int, default=3)
    p.add_argument("--val-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", action="store_true", help="Restore last alpha checkpoint and optimizer state")
    p.add_argument("--eval-only", action="store_true", help="Load last alpha weights and export validation only")
    p.add_argument("--preflight-only", action="store_true", help="Audit and decode data, no HF model weights or GPU")
    p.add_argument("--verify-weights-only", action="store_true",
                   help="Load official pretrained VAE and compare its tensors to checkpoint; do not train")
    p.add_argument("--local-files-only", action="store_true",
                   help="Require pretrained H3 checkpoint in the existing local cache; no model download")
    p.add_argument("--preview-frames", type=int, default=17,
                   help="Number of RGBA output frames to save per validation (up to 17)")
    return p.parse_args()


def log(message):
    print(message, flush=True)


def prepare_manifest(args):
    if args.manifest:
        manifest = Path(args.manifest).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        return manifest
    manifest = Path(args.workdir).expanduser().resolve() / "manifest.json"
    if not manifest.is_file():
        if args.skip_download:
            raise FileNotFoundError(f"No manifest: {manifest}. Remove --skip-download to prepare data.")
        script = Path(__file__).resolve().parent / "tools" / "prepare_data.py"
        subprocess.run([sys.executable, str(script), "--workdir", args.workdir], check=True)
    return manifest


def training_split(records, seed):
    if len(records) < 2:
        raise ValueError("Need at least two verified video pairs to hold out validation data.")
    entries = list(records)
    random.Random(seed).shuffle(entries)
    nval = max(1, min(len(entries) // 5, 12))
    return entries[nval:], entries[:nval]


def save_frames(gt, predicted, path, max_frames=17):
    """Write unpremultiplied straight-alpha PNGs and a labeled contact sheet."""
    import numpy as np
    from PIL import Image, ImageDraw
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    gt = gt.detach().float().cpu().clamp(0, 1)
    predicted = predicted.detach().float().cpu().clamp(0, 1)
    T = gt.shape[1]
    count = max(1, min(max_frames, T))
    selected = sorted(set(int(round(i * (T - 1) / max(count - 1, 1))) for i in range(count)))
    H, W = int(gt.shape[-2]), int(gt.shape[-1])
    contact = Image.new("RGB", (4 * W, len(selected) * 2 * H), (22, 22, 22))
    drawing = ImageDraw.Draw(contact)

    def frame_rgba(video, t):
        return Image.fromarray((video[:, t].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), "RGBA")

    for i, t in enumerate(selected):
        pair = []
        for name, video in (("target", gt), ("pred", predicted)):
            im = frame_rgba(video, t)
            im.save(path / f"{name}_{t:03d}.png")
            alpha = im.getchannel("A").convert("RGB")
            black = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            white = Image.new("RGBA", (W, H), (255, 255, 255, 255))
            black.alpha_composite(im)
            white.alpha_composite(im)
            pair.append((black.convert("RGB"), white.convert("RGB"), alpha))
        for j, group in enumerate(pair):
            y = (2 * i + j) * H
            for k, tile in enumerate(group):
                contact.paste(tile, (k * W, y))
            drawing.text((3 * W + 5, y + 5), f"{('GT', 'PRED')[j]} frame {t}", fill="white")
    contact.save(path / "contact_sheet.png")
    return str(path / "contact_sheet.png")


def validate(vae, loader, device, preview_dir=None, preview_frames=17,
             alpha_weight=3., edge_weight=1.):
    import torch
    from h3rgba.vae import rgba_to_vae_input, vae_output_to_rgba, rgba_reconstruction_loss
    vae.eval()
    stats = []
    for index, rgba in enumerate(loader):
        rgba = rgba.to(device=device, dtype=torch.float32, non_blocking=True)
        with torch.no_grad():
            # H3 explicitly pins the video encoder and quant_conv in FP32.
            with torch.autocast("cuda", enabled=False):
                moments = vae._encode_clip(rgba_to_vae_input(rgba).float())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = moments.chunk(2, dim=1)[0]
                recon = vae._decode_clip(z)[:, :, vae.frame_pre_padding:]
                if recon.shape != rgba.shape:
                    raise RuntimeError(f"Validation shape mismatch {recon.shape} / {rgba.shape}")
                pred = vae_output_to_rgba(recon.float())
        loss, parts = rgba_reconstruction_loss(pred, rgba)
        loss = loss + (alpha_weight - 3.) * parts["alpha"] + (edge_weight - 1.) * parts["edges"]
        entry = {"loss": float(loss), **{name: float(value) for name, value in parts.items()}}
        # Compare thresholded alpha silhouettes as well as alpha MAE.
        mask_p = pred[:, 3:4] > 0.5
        mask_t = rgba[:, 3:4] > 0.5
        intersection = (mask_p & mask_t).sum().item()
        union = (mask_p | mask_t).sum().item()
        entry["alpha_iou"] = float(intersection / union) if union else 1.0
        # Record an easily interpretable held-out alpha pixel MAE.
        entry["alpha_mae"] = float((pred[:, 3:4].clamp(0, 1) - rgba[:, 3:4]).abs().mean())
        stats.append(entry)
        if preview_dir is not None and index == 0:
            sheet = save_frames(rgba[0], pred[0], preview_dir, preview_frames)
            log(f"PREVIEW {sheet}")
    vae.train()
    return {k: round(sum(item[k] for item in stats) / len(stats), 6) for k in stats[0]}


def save_checkpoint(vae, optimizer, output_dir, step, metrics, args):
    import torch
    from train_vae import save_adapter
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_adapter(vae, checkpoint_dir, step, metrics)
    src = checkpoint_dir / f"alpha_step_{step:06d}.safetensors"
    shutil.copyfile(src, checkpoint_dir / "alpha_last.safetensors")
    if args.decoder_lora_rank:
        from safetensors.torch import save_file
        from h3rgba.decoder_lora import decoder_lora_state
        lora_path = checkpoint_dir / f"decoder_lora_step_{step:06d}.safetensors"
        save_file(decoder_lora_state(vae), str(lora_path))
        shutil.copyfile(lora_path, checkpoint_dir / "decoder_lora_last.safetensors")
    # Trusted local checkpoint only. The optimizer contains only the new
    # alpha parameters; frozen multi-billion parameter weights aren't saved.
    tmp = checkpoint_dir / "optimizer_last.pt.tmp"
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "seed": args.seed, "resolution": args.resolution,
                "decoder_lora_rank": args.decoder_lora_rank}, tmp)
    os.replace(tmp, checkpoint_dir / "optimizer_last.pt")
    log(f"CHECKPOINT {src}")


def main():
    args = arguments()
    if args.batch_size != 1:
        raise ValueError("22 GiB reference setup requires --batch-size 1")
    if args.resolution < 64 or args.resolution % 32:
        raise ValueError("Use resolution 64, 96, 128, 160, 192, ... (multiple of 32)")
    if args.steps < 0 or args.grad_accum < 1 or args.val_every < 1 or args.save_every < 1:
        raise ValueError("Invalid step or accumulation setting")
    if args.val_samples < 1:
        raise ValueError("Set --val-samples >= 1")
    if args.decoder_lora_rank < 0 or args.decoder_lora_rank > 32:
        raise ValueError("LoRA rank must be between 0 and 32")
    if args.alpha_loss_weight <= 0 or args.edge_loss_weight < 0:
        raise ValueError("Invalid alpha/edge loss weights")
    if args.init_alpha and (args.resume or args.eval_only):
        raise ValueError("Use --init-alpha only for a fresh run, not with --resume or --eval-only")
    if args.verify_weights_only and (args.preflight_only or args.resume or args.eval_only):
        raise ValueError("--verify-weights-only cannot be combined with preflight/resume/eval")

    manifest_path = prepare_manifest(args)
    from train_vae import PairedVideoDataset
    report = json.loads(manifest_path.read_text())
    rejected = report.get("rejected", [])
    records = report.get("clips", [])
    train_records, val_records = training_split(records, args.seed)
    log("DATA " + json.dumps({
        "valid_pairs": len(records), "train": len(train_records), "validation": len(val_records),
        "rejected": len(rejected), "unmatched": report.get("stats", {}).get("unmatched"),
        "resolution": args.resolution, "frames": 17, "precision": "BF16 mixed",
        "batch_size": args.batch_size, "gradient_accumulation": args.grad_accum,
    }))
    trainset = PairedVideoDataset(train_records, resolution=args.resolution)
    valset = PairedVideoDataset(val_records[:args.val_samples],
                                resolution=args.resolution, deterministic=True)
    example = trainset[0]
    log(f"DECODED_FIRST_TRAIN {tuple(example.shape)} RGB [{example[:3].min():.3f},{example[:3].max():.3f}] "
        f"ALPHA [{example[3:].min():.3f},{example[3:].max():.3f}]")
    if args.preflight_only:
        example_val = valset[0]
        log(f"DECODED_FIRST_VAL {tuple(example_val.shape)}")
        log("PREFLIGHT_OK: no H3 model weights downloaded; no GPU needed")
        return

    import torch
    from torch.utils.data import DataLoader
    from diffusers import AutoencoderKLMiniMaxH3
    from h3rgba.vae import widen_h3_vae, configure_stage1, rgba_to_vae_input, vae_output_to_rgba, rgba_reconstruction_loss
    from h3rgba.checkpoint import load_stage1_alpha

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. For free CPU dataset checks use --preflight-only.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU lacks native BF16 support; training won't silently switch to FP16.")
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    gpu_bytes = torch.cuda.get_device_properties(0).total_memory
    log(f"GPU {torch.cuda.get_device_name(0)} | VRAM {gpu_bytes / 2**30:.2f} GiB")
    if gpu_bytes < 18 * 2**30:
        log("WARNING: less than 18 GiB VRAM; start with --resolution 64.")

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    output = Path(args.output_dir).expanduser().resolve()
    if not (args.resume or args.eval_only or args.verify_weights_only):
        existing = [output / "metrics.jsonl", output / "validation.jsonl",
                    output / "checkpoints" / "optimizer_last.pt"]
        if any(p.exists() for p in existing):
            raise FileExistsError(
                f"Existing training run at {output}. Use a NEW --output-dir for a "
                "clean restart or --resume for continuation. Old checkpoints are untouched."
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Do not load the H3 33B DiT or the audio model. The visual VAE checkpoint
    # uses FP32 on disk and is sharded. BF16 loading keeps selected H3 modules
    # in FP32 as requested by upstream Diffusers.
    log(f"LOADING H3 VISUAL VAE from {args.hf_model}/vae")
    started = time.monotonic()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.hf_model, subfolder="vae", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True,
        local_files_only=args.local_files_only,
    )
    log(f"VAE_LOADED in {round(time.monotonic()-started, 1)} sec")
    if args.hf_model == "MiniMaxAI/MiniMax-H3":
        # Prove the official 3-shard checkpoint is present locally. This
        # check is read-only and never initiates another model download.
        from huggingface_hub import hf_hub_download
        from tools.verify_loaded_vae import verify_loaded_vae
        config_path = Path(hf_hub_download(
            args.hf_model, "vae/config.json", local_files_only=True
        ))
        snapshot = config_path.parent.parent
        result = verify_loaded_vae(vae, snapshot)
        (output / "pretrained_weight_verification.json").write_text(
            json.dumps(result, indent=2))
        log("OFFICIAL_H3_VAE_WEIGHTS_VERIFIED_IN_MEMORY")
    elif args.verify_weights_only:
        raise RuntimeError("--verify-weights-only supports only MiniMaxAI/MiniMax-H3")
    if args.verify_weights_only:
        log("PRETRAINED_WEIGHTS_ONLY_OK: checkpoint matches loaded encoder/decoder tensors, no training")
        return
    vae = widen_h3_vae(vae)
    vae.enable_gradient_checkpointing()
    # Start with no spatial splitting at 128x128, matching the original
    # non-overlapping path; use tiling again when increasing resolution.
    if args.resolution <= 128:
        vae.disable_tiling()
    vae.to(device)
    # Critical: the new trainable alpha parameters should retain FP32 master
    # weights under BF16 autocast. Never cast the entire VAE to BF16 again.
    vae.encoder.float()
    vae.quant_conv.float()
    vae.decoder.proj_out.float()
    trainable = configure_stage1(vae)
    if args.init_alpha:
        load_stage1_alpha(vae, args.init_alpha)
        log(f"INITIALIZED_ALPHA_FROM {args.init_alpha}")
    if args.decoder_lora_rank:
        # LoRA inside the residual attention/MLP path needs a nonzero H3
        # residual gate. Tiny random models initialize these gates to zero.
        # The pretrained checkpoint should have trained nonzero gates.
        with torch.no_grad():
            gate_max = max(float(g.abs().max()) for block in vae.decoder.transformer_blocks
                           for g in (block.scale1, block.scale2))
        log(f"H3_DECODER_GATE_MAX {gate_max:.6f}")
        if gate_max < 1e-6:
            raise RuntimeError("Decoder residual gates all zero: attention/MLP LoRA "
                               "would receive zero gradients. Verify pretrained H3 weights.")
        from h3rgba.decoder_lora import add_decoder_lora, decoder_lora_parameters
        count = add_decoder_lora(vae, rank=args.decoder_lora_rank)
        trainable += decoder_lora_parameters(vae)
        log(f"DECODER_LORA rank={args.decoder_lora_rank} adapted_linear_layers={count}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    train_count = sum(p.numel() for p in trainable)
    all_count = sum(p.numel() for p in vae.parameters())
    log(f"PARAMETERS full={all_count:,}; optimizer_params={train_count:,}; "
        "RGB output rows masked; initial alpha output zero.")
    ckpt_dir = output / "checkpoints"
    step = 0
    if args.resume or args.eval_only:
        path = ckpt_dir / "alpha_last.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"No adapter checkpoint found at {path}")
        load_stage1_alpha(vae, path)
        if args.decoder_lora_rank:
            from h3rgba.decoder_lora import load_decoder_lora_state
            lora_last = ckpt_dir / "decoder_lora_last.safetensors"
            if not lora_last.exists():
                raise FileNotFoundError(f"LoRA rank is enabled but checkpoint is missing: {lora_last}")
            load_decoder_lora_state(vae, lora_last)
        if args.resume:
            state_path = ckpt_dir / "optimizer_last.pt"
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            if state["resolution"] != args.resolution:
                raise ValueError("Resume resolution differs from the saved run")
            if state.get("decoder_lora_rank", 0) != args.decoder_lora_rank:
                raise ValueError("LoRA rank differs from the saved run; use --init-alpha with a new output directory")
            optimizer.load_state_dict(state["optimizer"])
            step = int(state["step"])
            log(f"RESUMED_AT_STEP {step}")

    train_loader = DataLoader(trainset, batch_size=1, num_workers=0,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=1, num_workers=0,
                            shuffle=False, pin_memory=True)

    torch.cuda.reset_peak_memory_stats()
    if args.eval_only:
        results = validate(vae, val_loader, device,
                           output / "previews" / "eval_only", args.preview_frames,
                           alpha_weight=args.alpha_loss_weight, edge_weight=args.edge_loss_weight)
        log("EVAL " + json.dumps(results))
        return

    vae.train()
    metrics_file = output / "metrics.jsonl"
    log("TRAINING_START")
    optimizer.zero_grad(set_to_none=True)
    micro = 0
    epoch = 0
    try:
        while step < args.steps:
            epoch += 1
            for batch in train_loader:
                x = batch.to(device, dtype=torch.float32, non_blocking=True)
                # Keep the encoder FP32 as in the official H3 configuration.
                with torch.autocast(device_type="cuda", enabled=False):
                    moments = vae._encode_clip(rgba_to_vae_input(x).float())
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Use posterior MODE (mu), not noisy stochastic samples
                    # for first-stage reconstruction training.
                    z = moments.chunk(2, dim=1)[0]
                    decoded = vae._decode_clip(z)[:, :, vae.frame_pre_padding:]
                    if decoded.shape != x.shape:
                        raise RuntimeError(f"Shape mismatch {decoded.shape} / {x.shape}")
                    pred = vae_output_to_rgba(decoded.float())
                loss, parts = rgba_reconstruction_loss(pred, x)
                loss = (loss + (args.alpha_loss_weight - 3.) * parts["alpha"] +
                        (args.edge_loss_weight - 1.) * parts["edges"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at step {step}: {float(loss)}")
                (loss / args.grad_accum).backward()
                micro += 1
                if micro % args.grad_accum:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                metrics = {k: round(float(v.detach()), 6) for k, v in parts.items()}
                result = dict(step=step, epoch=epoch, loss=round(float(loss.detach()), 6),
                              grad_norm=round(float(norm), 6), components=metrics,
                              peak_alloc_gib=round(torch.cuda.max_memory_allocated()/2**30, 2),
                              peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30, 2))
                log("STEP " + json.dumps(result))
                with metrics_file.open("a") as f:
                    f.write(json.dumps(result) + "\n")
                del loss, pred, decoded, moments, z, x
                if step == 1 or step % args.val_every == 0 or step >= args.steps:
                    vals = validate(vae, val_loader, device,
                                    output / "previews" / f"step_{step:06d}",
                                    args.preview_frames,
                                    alpha_weight=args.alpha_loss_weight,
                                    edge_weight=args.edge_loss_weight)
                    log("VALIDATION " + json.dumps({"step": step, **vals}))
                    with (output / "validation.jsonl").open("a") as f:
                        f.write(json.dumps({"step": step, **vals}) + "\n")
                if step % args.save_every == 0 or step >= args.steps:
                    save_checkpoint(vae, optimizer, output, step, metrics, args)
                if step >= args.steps:
                    break
    except torch.cuda.OutOfMemoryError:
        log(f"CUDA_OOM at step {step}; peak reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB. "
            "Restart from a saved checkpoint at 96 or 64px; do NOT increase batch size.")
        raise

    log(f"FINISHED steps={step} last_adapter={ckpt_dir/'alpha_last.safetensors'} "
        f"validation={output/'validation.jsonl'} previews={output/'previews'}")
    log("IMPORTANT: VAE only. Transparent text-to-video generation needs a separately trained H3 DiT LoRA.")


if __name__ == "__main__":
    main()
