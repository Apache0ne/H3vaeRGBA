"""RGB-preserving alpha training for the 24-channel MiniMax-H3 video VAE.

A paired video run showed that the original pretrained RGB reconstruction MAE
returned to baseline when the extra alpha encoder input was neutralized.
This objective trains the RGBA branch against:
- actual visible RGB pixels,
- a stopped-gradient, ORIGINAL-pretrained RGB teacher from the SAME model
  with its alpha input channel zeroed and decoder LoRA disabled,
- a gentle regularizer on the relative change of 24-channel latents.

No second copy of the multi-billion-parameter VAE is instantiated.
"""
from contextlib import nullcontext

import torch

from h3rgba.vae import rgba_to_vae_input, image_net_stats
from h3rgba.decoder_lora import temporarily_disable_decoder_lora


def visible_rgb_mae(rgb_pred, rgb_ref, target_alpha):
    """L1 normalized over visible foreground pixels, not the whole image."""
    if rgb_pred.shape != rgb_ref.shape or rgb_pred.shape[1] != 3:
        raise ValueError("RGB prediction and reference must match [B,3,T,H,W]")
    if target_alpha.shape != rgb_pred[:, :1].shape:
        raise ValueError("Alpha shape mismatch")
    weight = target_alpha.detach().float().clamp(0, 1)
    if float(weight.sum()) <= 0:
        return rgb_pred.float().sum() * 0.0
    return ((rgb_pred.float() - rgb_ref.float()).abs() * weight).sum() / (
        3.0 * weight.sum().clamp_min(1e-6)
    )


def relative_latent_shift(z, neutral):
    """Dimensionless mean absolute change relative to neutral latent RMS."""
    if z.shape != neutral.shape:
        raise ValueError("Latent comparison shapes do not match")
    denominator = neutral.detach().float().square().mean().sqrt().clamp_min(1e-5)
    return (z.float() - neutral.detach().float()).abs().mean() / denominator


@torch.no_grad()
def original_rgb_teacher(vae, rgba, *, need_rgb=True):
    """Neutral-alpha teacher from unchanged H3 RGB weights, same pixel frames.

    In the new 4-channel input the alpha channel's neutral activation is 0,
    NOT the normalized representation of transparent pixels (which is -1).
    The RGB channel weights and original decoder RGB output rows are frozen.
    The training runner already uses a FP32 encoder and BF16 autocast decoder.

    Returns z_ref [B,24,...] and RGB teacher [B,3,T,H,W] (or None).
    """
    if rgba.ndim != 5 or rgba.shape[1] != 4:
        raise ValueError("Expected RGBA pixel video")
    x = rgba_to_vae_input(rgba).detach()
    x[:, 3:4] = 0.0
    device = x.device.type
    was_training = vae.training
    try:
        vae.eval()
        with temporarily_disable_decoder_lora(vae):
            with torch.autocast("cuda", enabled=False) if device == "cuda" else nullcontext():
                moments = vae._encode_clip(x.float())
                neutral = moments.chunk(2, 1)[0].detach()
            rgb_ref = None
            if need_rgb:
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        decoded = vae._decode_clip(neutral)[:, :3, vae.frame_pre_padding:]
                else:
                    decoded = vae._decode_clip(neutral)[:, :3, vae.frame_pre_padding:]
                mean, std = image_net_stats(decoded.float())
                rgb_ref = (decoded.float() * std + mean).detach()
                if rgb_ref.shape != rgba[:, :3].shape:
                    raise RuntimeError(f"RGB teacher {rgb_ref.shape} != RGB target {rgba[:, :3].shape}")
    finally:
        vae.train(was_training)
    return neutral.detach(), rgb_ref


def preservation_penalties(pred_rgba, target_rgba, latents, neutral, rgb_teacher,
                           *, truth_weight=0.0, teacher_weight=0.0, latent_weight=0.0):
    """Return a scalar extra loss and detached-friendly component tensors."""
    if pred_rgba.shape != target_rgba.shape or pred_rgba.shape[1] != 4:
        raise ValueError("Expected matching RGBA [B,4,T,H,W] predictions and targets")
    zero = pred_rgba.float().new_zeros(())
    components = {}
    if truth_weight:
        components["visible_rgb_mae"] = visible_rgb_mae(
            pred_rgba[:, :3], target_rgba[:, :3], target_rgba[:, 3:4]
        )
    else:
        components["visible_rgb_mae"] = zero
    if teacher_weight:
        if rgb_teacher is None:
            raise ValueError("Pretrained RGB teacher is required when teacher_weight > 0")
        components["rgb_teacher_mae"] = visible_rgb_mae(
            pred_rgba[:, :3], rgb_teacher, target_rgba[:, 3:4]
        )
    else:
        components["rgb_teacher_mae"] = zero
    components["relative_latent_shift"] = (
        relative_latent_shift(latents, neutral) if latent_weight else zero
    )
    penalty = (truth_weight * components["visible_rgb_mae"]
               + teacher_weight * components["rgb_teacher_mae"]
               + latent_weight * components["relative_latent_shift"])
    return penalty, components
