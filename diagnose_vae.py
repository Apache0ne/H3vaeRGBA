"""Pretrained RGB H3 VAE baseline vs RGBA-adapted VAE, on the SAME video.

This diagnostic separates the original model's reconstruction limit at 128px
from alpha-adaptation failures. It never trains anything and never loads DiT.
"""
import argparse
import json
from pathlib import Path

import torch

from h3rgba.vae import (
    image_net_stats, rgba_to_vae_input, vae_output_to_rgba, widen_h3_vae,
)
from h3rgba.checkpoint import load_stage1_alpha


def rgb_to_h3(rgb):
    mean, std = image_net_stats(rgb)
    return (rgb - mean) / std


def h3_to_rgb(rgb):
    mean, std = image_net_stats(rgb)
    return rgb * std + mean


@torch.no_grad()
def roundtrip_clip(vae, pixels):
    """Official 17-frame private clip path, as in Stage-1 VAE trainer."""
    if pixels.shape[2] != 17:
        raise ValueError("Expected 17 pixel frames for H3 clip diagnostic")
    if pixels.device.type == "cuda":
        # Match the training runner: encoder FP32, decoder BF16 autocast.
        with torch.autocast("cuda", enabled=False):
            moments = vae._encode_clip(pixels.float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = moments.chunk(2, 1)[0]
            decoded = vae._decode_clip(z)
    else:
        moments = vae._encode_clip(pixels.float())
        z = moments.chunk(2, 1)[0]
        decoded = vae._decode_clip(z)
    out = decoded[:, :, vae.frame_pre_padding:].float()
    if out.shape[2:] != pixels.shape[2:]:
        raise RuntimeError(f"Decoded {tuple(out.shape)} vs input {tuple(pixels.shape)}")
    return out


def encode_alpha_ablation(vae, rgba):
    """Compare true encoded alpha vs physically ZERO alpha-channel activation.

    A zero *encoded input channel* (not alpha=0 in pixel space) exactly removes
    the added encoder conv's contribution. Both passes use the same pretrained
    RGB and decoder weights; neither pass changes any model parameter.
    """
    if rgba.shape[1] != 4 or rgba.shape[2] != 17:
        raise ValueError("Expected [B,4,17,H,W] rgba video")
    x = rgba_to_vae_input(rgba)
    no_alpha = x.clone()
    no_alpha[:, 3].zero_()
    with torch.no_grad():
        if rgba.device.type == "cuda":
            with torch.autocast("cuda", enabled=False):
                z_full = vae._encode_clip(x.float()).chunk(2, dim=1)[0]
                z_off = vae._encode_clip(no_alpha.float()).chunk(2, dim=1)[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded_off = vae._decode_clip(z_off)[:, :, vae.frame_pre_padding:]
        else:
            z_full = vae._encode_clip(x.float()).chunk(2, dim=1)[0]
            z_off = vae._encode_clip(no_alpha.float()).chunk(2, dim=1)[0]
            decoded_off = vae._decode_clip(z_off)[:, :, vae.frame_pre_padding:]
        rgba_off = vae_output_to_rgba(decoded_off.float()).clamp(0, 1)
    latent_delta = float((z_full - z_off).abs().mean())
    latent_rms = float(z_off.float().square().mean().sqrt())
    alpha_off = rgba_off[:, 3:4]
    gt_alpha = rgba[:, 3:4]
    intersection = ((alpha_off > .5) & (gt_alpha > .5)).sum()
    union = ((alpha_off > .5) | (gt_alpha > .5)).sum()
    return rgba_off, {
        "alpha_input_latent_mae_delta": round(latent_delta, 6),
        "alpha_input_latent_relative_delta": round(latent_delta / (latent_rms + 1e-8), 6),
        "alpha_zeroed_mae": round(float((alpha_off - gt_alpha).abs().mean()), 6),
        "alpha_zeroed_iou": round(float(intersection / union) if union > 0 else 1.0, 6),
    }


def comparison_metrics(target, rgb_original, rgb_rgba, alpha_rgba):
    """Return independent RGB L1 scores and alpha MAE/IoU; values in [0,1]."""
    a = target[:, 3:4]
    fgmask = a.clamp(0, 1)
    rgb = target[:, :3]
    def rgb_l1(p):
        return float(((p - rgb).abs() * fgmask).sum() / (3 * fgmask.sum() + 1e-6))
    pred_alpha = alpha_rgba.clamp(0, 1)
    mask_a = pred_alpha > 0.5
    mask_b = a > 0.5
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return {
        "rgb_original_visible_mae": round(rgb_l1(rgb_original), 6),
        "rgb_rgba_visible_mae": round(rgb_l1(rgb_rgba), 6),
        "rgb_original_vs_rgba_mae": round(float((rgb_original - rgb_rgba).abs().mean()), 6),
        "alpha_mae": round(float((pred_alpha - a).abs().mean()), 6),
        "alpha_iou": round(float(intersection / union) if union > 0 else 1.0, 6),
    }


def save_comparison(target, rgb_original, rgba_model, output,
                    frame_numbers=(0, 8, 16), rgba_without_alpha=None):
    from PIL import Image, ImageDraw
    import numpy as np
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = target.detach().float().cpu().clamp(0, 1)
    orig = rgb_original.detach().float().cpu().clamp(0, 1)
    pred = rgba_model.detach().float().cpu().clamp(0, 1)
    H, W = t.shape[-2:]
    columns_count = 6 if rgba_without_alpha is not None else 5
    off = rgba_without_alpha.detach().float().cpu().clamp(0, 1) if rgba_without_alpha is not None else None
    sheet = Image.new("RGB", (columns_count * W, len(frame_numbers) * H), (12, 12, 12))
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(frame_numbers):
        gt = t[:, index].permute(1, 2, 0).numpy()
        raw = orig[:, index].permute(1, 2, 0).numpy()
        rgba = pred[:, index].permute(1, 2, 0).numpy()
        white = np.ones_like(raw)
        black = np.zeros_like(raw)
        columns = [
            gt[:, :, :3] * gt[:, :, 3:] + (1 - gt[:, :, 3:]) * black,
            raw,
            rgba[:, :, :3] * rgba[:, :, 3:] + (1 - rgba[:, :, 3:]) * black,
            rgba[:, :, :3] * rgba[:, :, 3:] + (1 - rgba[:, :, 3:]) * white,
            np.repeat(rgba[:, :, 3:], 3, axis=2),
        ]
        if off is not None:
            off_array = off[:, index].permute(1, 2, 0).numpy()
            columns.append(np.repeat(off_array[:, :, 3:], 3, axis=2))
        for col, arr in enumerate(columns):
            image = Image.fromarray(np.uint8(np.round(arr.clip(0, 1) * 255)), "RGB")
            sheet.paste(image, (col * W, row * H))
    # Save a legend separately so pixels are not obscured.
    legend = ["GT on black", "Original RGB H3 VAE", "RGBA on black", "RGBA on white", "RGBA alpha mask"]
    if off is not None:
        legend.append("Alpha mask with input alpha channel zeroed")
    sheet.save(out)
    out.with_suffix(".json").write_text(json.dumps({"columns": legend, "frames": list(frame_numbers)}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", default="/content/h3rgba_data")
    p.add_argument("--manifest")
    p.add_argument("--output-dir", default="/content/h3rgba_runs/vae_diagnosis")
    p.add_argument("--alpha-checkpoint", default="/content/h3rgba_runs/vae22_fresh_verified/checkpoints/alpha_last.safetensors")
    p.add_argument("--decoder-lora-checkpoint", help="Optional additional pretrained Stage-2 VAE decoder LoRA")
    p.add_argument("--decoder-lora-rank", type=int, default=4)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=1)
    a = p.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires BF16-capable CUDA GPU for model diagnostics")
    from run_vae_22gb import training_split
    from train_vae import PairedVideoDataset
    from torch.utils.data import DataLoader
    from diffusers import AutoencoderKLMiniMaxH3
    from tools.verify_loaded_vae import verify_loaded_vae
    from huggingface_hub import hf_hub_download

    data = Path(a.manifest) if a.manifest else Path(a.workdir) / "manifest.json"
    _, validation = training_split(json.loads(data.read_text())["clips"], a.seed)
    entries = validation[:a.samples]
    if not entries:
        raise RuntimeError("No validation pairs")
    dataset = PairedVideoDataset(entries, resolution=a.resolution, deterministic=True)
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print("LOADING_OFFICIAL_RGB_H3_VAE", flush=True)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        "MiniMaxAI/MiniMax-H3", subfolder="vae", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True
    )
    config_path = Path(hf_hub_download("MiniMaxAI/MiniMax-H3", "vae/config.json",
                                       local_files_only=True))
    verify_loaded_vae(vae, config_path.parent.parent)
    vae.to("cuda")
    vae.encoder.float()
    vae.quant_conv.float()
    vae.decoder.proj_out.float()
    if a.resolution <= 128:
        vae.disable_tiling()
    vae.eval()

    summaries = []
    for n, rgba in enumerate(DataLoader(dataset, batch_size=1, num_workers=0)):
        # Establish original RGB pretrained reconstruction BEFORE channel widening.
        rgba = rgba.to("cuda", dtype=torch.float32)
        with torch.no_grad():
            original = h3_to_rgb(roundtrip_clip(vae, rgb_to_h3(rgba[:, :3]))).clamp(0, 1)
        # Widen in-place. No model duplication and no previous optimizer state.
        if n == 0:
            vae = widen_h3_vae(vae)
            vae.decoder.proj_out.float()
            vae.eval()
            adapter = Path(a.alpha_checkpoint)
            if adapter.is_file():
                load_stage1_alpha(vae, adapter)
                print(f"USING_ALPHA_CHECKPOINT={adapter}", flush=True)
            else:
                raise FileNotFoundError(f"Missing alpha checkpoint: {adapter}")
            if a.decoder_lora_checkpoint:
                from h3rgba.decoder_lora import add_decoder_lora, load_decoder_lora_state
                count = add_decoder_lora(vae, rank=a.decoder_lora_rank)
                load_decoder_lora_state(vae, a.decoder_lora_checkpoint)
                print(f"LOADED_DECODER_LORA_LAYERS={count}", flush=True)
        with torch.no_grad():
            predicted = vae_output_to_rgba(roundtrip_clip(vae, rgba_to_vae_input(rgba))).clamp(0, 1)
        # original is only calculated once for single sample; limit CLI to 1
        metrics = comparison_metrics(rgba, original, predicted[:, :3], predicted[:, 3:])
        off, ablation = encode_alpha_ablation(vae, rgba)
        metrics.update(ablation)
        metrics["rgb_without_alpha_input_visible_mae"] = round(
            float(((off[:, :3] - rgba[:, :3]).abs() * rgba[:, 3:4]).sum() /
                  (3 * rgba[:, 3:4].sum() + 1e-6)), 6)
        metrics["rgb_with_vs_without_alpha_input_mae"] = round(
            float((predicted[:, :3] - off[:, :3]).abs().mean()), 6)
        metrics["sample"] = n
        summaries.append(metrics)
        comparison_path = save_comparison(rgba[0], original[0], predicted[0],
                                         output / f"sample_{n:02d}_comparison.png",
                                         rgba_without_alpha=off[0])
        print("DIAGNOSIS " + json.dumps(metrics), flush=True)
        print(f"COMPARE_IMAGE {comparison_path}", flush=True)
        break
    (output / "metrics.json").write_text(json.dumps(summaries, indent=2))
    print(f"DIAGNOSIS_COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
