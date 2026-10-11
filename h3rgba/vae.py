"""MiniMax-H3 RGB->RGBA widening. Requires the official Diffusers H3 VAE.

The pixel convention is [RGB normalized by ImageNet mean/std, alpha scaled to [-1,1]].
The standard H3 latent interface (24 channels) and per-channel latents_mean/std
are not changed. Do NOT feed 4-channel tensors to the original RGB processor.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def image_net_stats(reference):
    shape = (1, 3, 1, 1, 1)
    mean = reference.new_tensor(RGB_MEAN).view(shape)
    std = reference.new_tensor(RGB_STD).view(shape)
    return mean, std


def rgba_to_vae_input(rgba):
    """[B,4,T,H,W] values in [0,1] -> 4-channel VAE training convention."""
    if rgba.ndim != 5 or rgba.shape[1] != 4:
        raise ValueError(f"Expected [B,4,T,H,W], got {tuple(rgba.shape)}")
    mean, std = image_net_stats(rgba)
    return torch.cat(((rgba[:, :3] - mean) / std, 2 * rgba[:, 3:4] - 1), dim=1)


def vae_output_to_rgba(sample, clamp=False):
    """Inverse convention; clamp only for previews, never while computing loss."""
    if sample.ndim != 5 or sample.shape[1] != 4:
        raise ValueError(f"Expected [B,4,T,H,W], got {tuple(sample.shape)}")
    mean, std = image_net_stats(sample)
    result = torch.cat((sample[:, :3] * std + mean, (sample[:, 3:4] + 1) / 2), 1)
    return result.clamp(0, 1) if clamp else result


def widen_h3_vae(vae):
    """Widen trained 3-in/3-out H3 weights without reinitializing the RGB pathway.

    Works with the Diffusers H3 VAE and its custom causal-convolution subclass.
    Deep-copying the first convolution preserves its causal padding semantics.
    The ViT final projection is organized in contiguous channel-sized patch groups.
    """
    conv = vae.encoder.conv_in
    projection = vae.decoder.proj_out
    if conv.in_channels != 3 or projection.out_features % 3:
        raise ValueError("Expected original 3-in/3-out H3 VAE; already widened or incompatible model")
    if getattr(vae.config, "latent_channels", 24) != 24:
        raise ValueError("H3 latent channel count is not 24")
    patch_volume = projection.out_features // 3
    expected_volume = (vae.decoder.patch_size_t * vae.decoder.patch_size**2)
    if patch_volume != expected_volume:
        raise ValueError(f"ViT decoder patch layout mismatch: {patch_volume} != {expected_volume}")

    # Preserve *all* Conv3d attributes, including H3's causal reflect-padding.
    new_conv = copy.deepcopy(conv)
    new_conv.in_channels = 4
    new_conv.weight = nn.Parameter(torch.cat(
        (conv.weight.detach().clone(), torch.zeros_like(conv.weight[:, :1]).detach()), dim=1
    ))
    vae.encoder.conv_in = new_conv

    new_projection = nn.Linear(
        projection.in_features, 4 * patch_volume, bias=projection.bias is not None,
        device=projection.weight.device, dtype=projection.weight.dtype
    )
    with torch.no_grad():
        new_projection.weight.zero_()
        new_projection.weight[:3 * patch_volume].copy_(projection.weight)
        if projection.bias is not None:
            new_projection.bias.zero_()
            new_projection.bias[:3 * patch_volume].copy_(projection.bias)
    vae.decoder.proj_out = new_projection
    vae.register_to_config(in_channels=4, out_channels=4)
    return vae


def configure_stage1(vae):
    """Only added alpha slices change; original RGB slices receive no gradients.

    Use zero optimizer weight decay on these parameters, otherwise AdamW may
    modify the frozen RGB slices even when their gradients are masked.
    Call once, after widening, before creating the optimizer.
    """
    vae.requires_grad_(False)
    conv = vae.encoder.conv_in
    proj = vae.decoder.proj_out
    conv.weight.requires_grad_(True)
    proj.weight.requires_grad_(True)
    if proj.bias is not None:
        proj.bias.requires_grad_(True)
    # Hooks are intentionally transient; re-register on each process startup.
    conv.weight.register_hook(lambda g: torch.cat((torch.zeros_like(g[:, :3]), g[:, 3:]), 1))
    n = proj.out_features // 4
    proj.weight.register_hook(lambda g: torch.cat((torch.zeros_like(g[:3 * n]), g[3 * n:]), 0))
    if proj.bias is not None:
        proj.bias.register_hook(lambda g: torch.cat((torch.zeros_like(g[:3 * n]), g[3 * n:]), 0))
    return [p for p in vae.parameters() if p.requires_grad]


def rgba_reconstruction_loss(pred, target):
    """Differentiable RGB/alpha reconstruction with composition and temporal terms.

    Inputs are un-clamped RGBA in [B,4,T,H,W]. Loss runs in FP32.
    """
    pred, target = pred.float(), target.float()
    if pred.shape != target.shape or target.shape[1] != 4:
        raise ValueError(f"Prediction {tuple(pred.shape)} vs target {tuple(target.shape)}")
    a_p, a_t = pred[:, 3:4], target[:, 3:4]
    rgb_p, rgb_t = pred[:, :3], target[:, :3]
    # Foreground colors are meaningful primarily where alpha > 0.
    rgb_weight = a_t.detach().clamp(0, 1)
    rgb_l1 = ((rgb_p - rgb_t).abs() * rgb_weight).mean()
    alpha_l1 = (a_p - a_t).abs().mean()
    composite_l1 = 0
    # Contrasting composites expose alpha halos and premultiplication errors.
    for color in (0.0, 1.0):
        composited_p = rgb_p * a_p + color * (1 - a_p)
        composited_t = rgb_t * a_t + color * (1 - a_t)
        composite_l1 = composite_l1 + (composited_p - composited_t).abs().mean() * 0.5
    edge_loss = (a_p[..., 1:, :] - a_p[..., :-1, :] -
                 (a_t[..., 1:, :] - a_t[..., :-1, :])).abs().mean()
    edge_loss = edge_loss + (a_p[..., :, 1:] - a_p[..., :, :-1] -
                             (a_t[..., :, 1:] - a_t[..., :, :-1])).abs().mean()
    temporal_loss = a_p.new_zeros(())
    if a_p.shape[2] > 1:
        temporal_loss = ((a_p[:, :, 1:] - a_p[:, :, :-1]) -
                         (a_t[:, :, 1:] - a_t[:, :, :-1])).abs().mean()
    # The out-of-range term avoids zero-gradient clamp effects in training.
    range_loss = (F.relu(-a_p) + F.relu(a_p - 1)).mean()
    components = dict(
        alpha=alpha_l1, foreground=rgb_l1, composite=composite_l1,
        edges=edge_loss, temporal=temporal_loss, alpha_range=range_loss,
    )
    total = (3 * alpha_l1 + rgb_l1 + 2 * composite_l1 +
             edge_loss + 0.5 * temporal_loss + range_loss)
    return total, components


def normalize_latents(z, vae):
    mean = z.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = z.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    if z.shape[1] != 24 or mean.shape[1] != 24 or bool((std <= 0).any()):
        raise ValueError("Invalid H3 latent normalization configuration")
    return ((z.float() - mean.float()) / std.float()).to(z.dtype)
