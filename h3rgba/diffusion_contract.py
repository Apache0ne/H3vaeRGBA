"""Strict boundary between generated H3 video latents and alpha VAE latents.

H3's published t2va diffusion model outputs only RGB video latents (24ch),
plus audio latents. The dual-decoder VAE LoRA trained in this repo needs a
SECOND 24ch alpha latent stream (and RGB latents) as codec inputs. The codec
receiving ground-truth alpha latents is an ORACLE reconstruction experiment,
not generated transparency. These helpers are a fail-closed contract.

No network, GPU, diffusion models or heavy imports are needed.
"""
import torch


class AlphaLatentsNotGenerated(RuntimeError):
    pass


def h3_video_shape(*, num_frames, height, width, batch=1):
    if num_frames < 124 or (num_frames - 5) % 17 or num_frames > 362:
        raise ValueError("H3 generation uses at least 124 frames of form 17*n+5")
    if min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError("H3 height and width must be positive multiples of 32")
    return (batch, 24, (num_frames - 5) // 17 * 5 + 2, height // 16, width // 16)


def validate_two_generated_streams(z_rgb, z_alpha=None, *, alpha_source=None,
                                   frames, height, width):
    """Only true diffusion-produced alpha latents qualify for generation.

    Returns validated shape information for the future alpha-diffusion adapter.
    Does not hallucinate alpha from RGB or treat supplied GT alpha as generated.
    """
    expected = h3_video_shape(num_frames=frames, height=height, width=width)
    if not isinstance(z_rgb, torch.Tensor) or tuple(z_rgb.shape) != expected:
        raise ValueError(f"H3 video diffusion z_rgb has wrong shape, expected {expected}")
    if z_alpha is None:
        raise AlphaLatentsNotGenerated(
            "Official H3 t2va yields RGB latents only. DualAlphaCodec requires "
            "independently generated alpha latents; the VAE LoRA alone cannot "
            "provide them. Train a diffusion-side alpha latent predictor."
        )
    if not isinstance(z_alpha, torch.Tensor) or tuple(z_alpha.shape) != expected:
        raise ValueError(f"Generated z_alpha must be {expected}")
    if alpha_source != "trained_alpha_diffusion":
        raise AlphaLatentsNotGenerated(
            "Alpha latents were provided, but their source is not a trained "
            "diffusion model. GT alpha encoding is an oracle VAE test, not "
            "generative alpha. Supply alpha_source='trained_alpha_diffusion' "
            "only for actual verified diffusion outputs."
        )
    if not torch.isfinite(z_rgb).all() or not torch.isfinite(z_alpha).all():
        raise ValueError("Nonfinite generated latents")
    return {"rgb_latent_shape":list(expected),
            "alpha_latent_shape":list(expected),
            "alpha_source":"trained_alpha_diffusion",
            "generated_transparency_latents_available":True}


def denormalize_h3_video_latents(normalized_latents, *, mean, std):
    """Bridge from H3 *denoiser output* into raw VAE latent coordinates.

    Official H3 decode block first applies per-channel
    z_raw = z_normalized * VAE.latents_std + VAE.latents_mean.
    Must not decode normalized transformer latents directly via dual codec.
    """
    if normalized_latents.ndim != 5 or normalized_latents.shape[1] != 24:
        raise ValueError("Expected normalized 24-channel H3 video latents")
    mu = torch.as_tensor(mean, device=normalized_latents.device,
                         dtype=normalized_latents.dtype)
    sig = torch.as_tensor(std, device=normalized_latents.device,
                          dtype=normalized_latents.dtype)
    if mu.numel() != 24 or sig.numel() != 24 or torch.any(sig <= 0):
        raise ValueError("Expected 24 finite positive standard deviations")
    if not torch.isfinite(mu).all() or not torch.isfinite(sig).all():
        raise ValueError("Invalid VAE latent statistics")
    return normalized_latents * sig.view(1,24,1,1,1) + mu.view(1,24,1,1,1)
