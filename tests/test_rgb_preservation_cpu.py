"""CPU tests for original RGB H3 teacher and preservation objective."""
import torch

from h3rgba.vae import widen_h3_vae, rgba_to_vae_input, vae_output_to_rgba
from h3rgba.decoder_lora import add_decoder_lora, temporarily_disable_decoder_lora, LoRALinear
from h3rgba.preservation import (
    original_rgb_teacher, preservation_penalties,
    relative_latent_shift, visible_rgb_mae,
)


def make_tiny_h3():
    from diffusers import AutoencoderKLMiniMaxH3
    model = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1, norm_num_groups=4,
        decoder_num_layers=1, decoder_num_attention_heads=2,
        decoder_attention_head_dim=16, decoder_ffn_mult=2,
        clip_length=17, token_drop=3,
    )
    model.disable_tiling()
    return widen_h3_vae(model)


def test_preservation_gradients_and_visible_rgb_normalization():
    torch.manual_seed(5)
    gt = torch.rand(1, 4, 3, 16, 16)
    gt[:, 3:4, :, :8] = 0.
    pred = (gt * 0.8).clone().requires_grad_(True)
    z = torch.randn(1, 24, 2, 2, 2, requires_grad=True)
    neutral = z.detach().clone() * 0.8
    teacher = gt[:, :3].clone()

    weighted, metrics = preservation_penalties(
        pred, gt, z, neutral, teacher,
        truth_weight=2., teacher_weight=3., latent_weight=.2
    )
    assert torch.isfinite(weighted)
    assert metrics["visible_rgb_mae"] > 0
    assert metrics["rgb_teacher_mae"] > 0
    assert metrics["relative_latent_shift"] > 0
    weighted.backward()
    assert pred.grad is not None and pred.grad[:, :3].abs().sum() > 0
    assert z.grad is not None and z.grad.abs().sum() > 0
    assert torch.isfinite(pred.grad).all() and torch.isfinite(z.grad).all()

    # Masking normalizes by actual foreground area, not total pixels.
    t = torch.ones(1, 3, 1, 2, 2)
    p = torch.zeros_like(t)
    a = torch.tensor([[[[[0., 0.], [1., 0.]]]]])
    assert abs(float(visible_rgb_mae(p, t, a)) - 1.) < 1e-6
    assert abs(float(relative_latent_shift(z.detach(), z.detach()))) < 1e-6


def test_frozen_rgb_teacher_matches_original_when_alpha_is_neutral():
    torch.manual_seed(9)
    model = make_tiny_h3()
    model.eval()
    pixels = torch.rand(1, 4, 17, 32, 32)
    z, teacher = original_rgb_teacher(model, pixels, need_rgb=True)
    assert teacher.shape == pixels[:, :3].shape
    assert z.shape[1] == 24
    assert not z.requires_grad and not teacher.requires_grad

    raw = rgba_to_vae_input(pixels)
    raw[:, 3].zero_()
    with torch.no_grad():
        decoded = model._decode_clip(model._encode_clip(raw).chunk(2, 1)[0])
        original = vae_output_to_rgba(decoded[:, :, model.frame_pre_padding:])[:, :3]
    assert torch.allclose(teacher, original, atol=1e-6)


def test_rgb_teacher_disables_decoder_lora_and_restores_it():
    torch.manual_seed(10)
    model = make_tiny_h3()
    model.eval()
    rgba = torch.rand(1, 4, 17, 32, 32)
    base_latent, base_rgb = original_rgb_teacher(model, rgba)
    assert add_decoder_lora(model, rank=2) > 0
    modules = [m for m in model.decoder.modules() if isinstance(m, LoRALinear)]
    with torch.no_grad():
        # Deliberately perturb adapter; original teacher must bypass it.
        for module in modules:
            module.lora_up.fill_(0.0003)
    new_latent, new_rgb = original_rgb_teacher(model, rgba)
    assert torch.allclose(new_latent, base_latent)
    assert torch.allclose(new_rgb, base_rgb, atol=1e-6)
    assert all(m.enabled for m in modules)
    with temporarily_disable_decoder_lora(model):
        assert all(not m.enabled for m in modules)
    assert all(m.enabled for m in modules)
