import types

import pytest
import torch
from torch import nn

from h3rgba.vae import (
    widen_h3_vae, configure_stage1, rgba_to_vae_input,
    vae_output_to_rgba, rgba_reconstruction_loss, normalize_latents,
)


class ToyVAE(nn.Module):
    """Small CPU stand-in to catch weight-layout and gradient-mask regressions."""
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            latent_channels=24, in_channels=3, out_channels=3,
            latents_mean=[0.0] * 24, latents_std=[1.0] * 24,
        )
        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv3d(3, 8, kernel_size=3, padding=1)
        self.decoder = nn.Module()
        self.decoder.patch_size = 16
        self.decoder.patch_size_t = 4
        self.decoder.proj_out = nn.Linear(16, 3 * 4 * 16 * 16)

    def register_to_config(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.config, k, v)


def test_rgb_weights_exact_and_alpha_zero_initialized():
    model = ToyVAE()
    oldconv = model.encoder.conv_in.weight.detach().clone()
    oldproj = model.decoder.proj_out.weight.detach().clone()
    oldbias = model.decoder.proj_out.bias.detach().clone()
    widen_h3_vae(model)
    assert tuple(model.encoder.conv_in.weight.shape[:2]) == (8, 4)
    assert torch.equal(model.encoder.conv_in.weight[:, :3], oldconv)
    assert model.encoder.conv_in.weight[:, 3:].count_nonzero() == 0
    assert model.decoder.proj_out.out_features == 4096
    assert torch.equal(model.decoder.proj_out.weight[:3072], oldproj)
    assert torch.equal(model.decoder.proj_out.bias[:3072], oldbias)
    assert model.decoder.proj_out.weight[3072:].count_nonzero() == 0
    assert model.decoder.proj_out.bias[3072:].count_nonzero() == 0
    assert model.config.in_channels == model.config.out_channels == 4
    assert model.decoder.out_channels == 4
    with pytest.raises(ValueError):
        widen_h3_vae(model)


def test_gradients_do_not_modify_old_rgb_rows():
    model = widen_h3_vae(ToyVAE())
    params = configure_stage1(model)
    assert len(params) == 3
    x = torch.randn(1, 4, 2, 6, 6)
    h = model.encoder.conv_in(x).sum()
    q = model.decoder.proj_out(torch.randn(3, 16)).sum()
    (h + q).backward()
    assert model.encoder.conv_in.weight.grad[:, :3].abs().sum() == 0
    assert model.encoder.conv_in.weight.grad[:, 3:].abs().sum() > 0
    assert model.decoder.proj_out.weight.grad[:3072].abs().sum() == 0
    assert model.decoder.proj_out.weight.grad[3072:].abs().sum() > 0
    assert model.decoder.proj_out.bias.grad[:3072].abs().sum() == 0


def test_pixel_roundtrip_and_alpha_loss():
    src = torch.rand(2, 4, 3, 32, 32)
    x = rgba_to_vae_input(src)
    rec = vae_output_to_rgba(x)
    assert torch.allclose(rec, src, atol=1e-6)
    predicted = (src * 0.8).requires_grad_(True)
    loss, parts = rgba_reconstruction_loss(predicted, src)
    assert torch.isfinite(loss) and set(parts) == {
        "alpha", "foreground", "composite", "edges", "temporal", "alpha_range"
    }
    loss.backward()
    assert predicted.grad is not None
    with pytest.raises(ValueError):
        rgba_to_vae_input(src[:, :3])


def test_standard_latent_normalization():
    model = ToyVAE()
    model.config.latents_mean = [0.5] * 24
    model.config.latents_std = [2.] * 24
    z = torch.full((1, 24, 2, 4, 4), 2.5)
    normalized = normalize_latents(z, model)
    assert torch.allclose(normalized, torch.ones_like(normalized))
