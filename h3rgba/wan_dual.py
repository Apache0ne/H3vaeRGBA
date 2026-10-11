"""Wan-Alpha-inspired dual-path RGB/alpha codec for MiniMax-H3 VAE.

RGB: frozen original pretrained H3 encoder and decoder, completely unchanged.
Alpha: same frozen encoder, running separately on repeated grayscale alpha;
       a tiny residual latent merger combines RGB and alpha latent evidence;
       decoder LoRA + a trainable 3->1 pixel head reconstruct alpha.

Only one original H3 VAE weight copy is needed. A generated H3 RGB latent
alone cannot supply this codec's alpha branch: diffusion LoRA training on
jointly encoded latents is a separate stage, NOT claimed here.
"""
import torch
from torch import nn
from h3rgba.vae import image_net_stats
from h3rgba.decoder_lora import (
    add_decoder_lora, temporarily_disable_decoder_lora,
    decoder_lora_parameters, decoder_lora_state,
)


def normalize_rgb(rgb):
    if rgb.ndim != 5 or rgb.shape[1] != 3:
        raise ValueError("Expected RGB [B,3,T,H,W]")
    mean, std = image_net_stats(rgb)
    return (rgb - mean) / std


def denormalize_rgb(rgb):
    mean, std = image_net_stats(rgb)
    return rgb * std + mean


class DualAlphaCodec(nn.Module):
    """Lightweight modules around a pretrained 3-in/3-out H3 visual VAE."""

    def __init__(self, vae, *, rank=4, merger_scale=0.15):
        super().__init__()
        if vae.encoder.conv_in.in_channels != 3 or vae.decoder.out_channels != 3:
            raise ValueError("Use the ORIGINAL non-widened pretrained RGB H3 VAE")
        if not 0.0 < merger_scale <= 1.0:
            raise ValueError("merger_scale must lie in (0,1]")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.vae = vae
        self.vae.requires_grad_(False)
        self.merger_scale = merger_scale
        self.merger = nn.Conv3d(48, 24, 1, bias=True, dtype=torch.float32)
        nn.init.zeros_(self.merger.weight)
        nn.init.zeros_(self.merger.bias)
        self.alpha_head = nn.Conv3d(3, 1, 1, dtype=torch.float32)
        # Decode grayscale-like video using the pretrained H3 RGB output.
        # Correct conversion of H3's three ImageNet-normalized RGB channels:
        # A = (R + G + B) / 3 in the [0,1] pixel domain.
        with torch.no_grad():
            self.alpha_head.weight.copy_(
                torch.tensor([0.229/3, 0.224/3, 0.225/3], dtype=torch.float32).view(1,3,1,1,1)
            )
            self.alpha_head.bias.fill_((0.485 + 0.456 + 0.406) / 3.0)
        self.lora_layers = add_decoder_lora(self.vae, rank=rank)
        # Never call requires_grad_(True) on the entire model: original VAE
        # must remain frozen. LoRA parameters are already trainable.
        self.rank = rank

    def trainable_parameters(self):
        return list(self.merger.parameters()) + list(self.alpha_head.parameters()) + decoder_lora_parameters(self.vae)

    def train(self, mode=True):
        super().train(mode)
        # Frozen original VAE remains in inference mode; decoder LoRA still
        # receives gradients from alpha reconstruction.
        self.vae.eval()
        return self

    def encode_rgb(self, rgb):
        with torch.no_grad():
            with torch.autocast("cuda", enabled=False) if rgb.device.type == "cuda" else torch.no_grad():
                moments = self.vae._encode_clip(normalize_rgb(rgb).float())
                return moments.chunk(2, dim=1)[0].detach()

    def encode_pair(self, rgba, *, background=None):
        """Two frozen 3-channel encoding passes; no duplicated VAE weights."""
        if rgba.ndim != 5 or rgba.shape[1] != 4:
            raise ValueError("Expected RGBA [B,4,17,H,W]")
        alpha = rgba[:, 3:4].clamp(0, 1)
        if background is None:
            foreground = rgba[:, :3]
        else:
            foreground = rgba[:, :3] * alpha + background * (1 - alpha)
        z_rgb = self.encode_rgb(foreground)
        z_alpha = self.encode_rgb(alpha.expand(-1, 3, -1, -1, -1))
        if z_rgb.shape != z_alpha.shape or z_rgb.shape[1] != 24:
            raise RuntimeError("H3 latent shapes don't match RGB and alpha streams")
        return z_rgb, z_alpha

    def merge_latents(self, z_rgb, z_alpha):
        if z_rgb.shape != z_alpha.shape or z_rgb.shape[1] != 24:
            raise ValueError("Expected paired 24-channel H3 video latents")
        # A zero-initialized residual preserves original alpha VAE roundtrip
        # at step zero. Bounded residual guards against massive latent shifts.
        correction = self.merger(torch.cat([z_rgb, z_alpha], dim=1).float())
        return z_alpha + self.merger_scale * torch.tanh(correction)

    def decode_alpha(self, z_rgb, z_alpha):
        z = self.merge_latents(z_rgb, z_alpha)
        if z.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = self.vae._decode_clip(z)
        else:
            decoded = self.vae._decode_clip(z)
        decoded = decoded[:, :, self.vae.frame_pre_padding:]
        with torch.autocast("cuda", enabled=False) if decoded.device.type == "cuda" else torch.no_grad() if not torch.is_grad_enabled() else torch.enable_grad():
            predicted = self.alpha_head(decoded.float())
        return predicted

    @torch.no_grad()
    def decode_rgb(self, z_rgb):
        # Original 3-channel decoder, with the alpha LoRA disabled. This is
        # guaranteed not to update the pretrained RGB weights.
        with temporarily_disable_decoder_lora(self.vae):
            if z_rgb.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    decoded = self.vae._decode_clip(z_rgb)
            else:
                decoded = self.vae._decode_clip(z_rgb)
        return denormalize_rgb(decoded[:, :, self.vae.frame_pre_padding:].float())

    def rgba_forward(self, rgba, *, background=None, decode_rgb=False):
        z_rgb, z_alpha = self.encode_pair(rgba, background=background)
        alpha = self.decode_alpha(z_rgb, z_alpha)
        rgb = self.decode_rgb(z_rgb) if decode_rgb else None
        return alpha, rgb

    def adapter_state(self):
        state = {
            "merger.weight": self.merger.weight.detach().cpu().contiguous(),
            "merger.bias": self.merger.bias.detach().cpu().contiguous(),
            "alpha_head.weight": self.alpha_head.weight.detach().cpu().contiguous(),
            "alpha_head.bias": self.alpha_head.bias.detach().cpu().contiguous(),
        }
        for k, v in decoder_lora_state(self.vae).items():
            state["decoder." + k] = v
        return state

    def load_adapter(self, path):
        from safetensors.torch import load_file
        payload = load_file(str(path), device="cpu")
        target = dict(self.named_parameters())
        expected = set(self.adapter_state())
        if set(payload) != expected:
            raise ValueError(
                f"Adapter key mismatch: missing={sorted(expected-set(payload))[:5]} "
                f"unexpected={sorted(set(payload)-expected)[:5]}"
            )
        with torch.no_grad():
            for name, value in payload.items():
                # model named_parameters use vae.decoder.*; serialized
                # adapter state uses decoder.* to stay VAE-prefix free.
                real_name = "vae." + name if name.startswith("decoder.") else name
                param = target[real_name]
                if param.shape != value.shape:
                    raise ValueError(f"Shape mismatch {name}")
                param.copy_(value.to(device=param.device, dtype=param.dtype))
