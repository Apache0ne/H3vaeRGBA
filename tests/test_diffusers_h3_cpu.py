"""CPU integration check using the real Diffusers H3 VAE class with tiny dimensions."""
import pytest
import torch

from h3rgba.vae import widen_h3_vae, configure_stage1, rgba_to_vae_input, vae_output_to_rgba, rgba_reconstruction_loss


def test_real_h3_tiny_encoder_decoder_roundtrip_and_gradient():
    from diffusers import AutoencoderKLMiniMaxH3

    # Same original spatial/temporal geometry, but deliberately tiny channels,
    # one transformer layer and tiny head dimension for free CPU-only checking.
    model = AutoencoderKLMiniMaxH3(
        in_channels=3,
        out_channels=3,
        latent_channels=24,
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=16,
        decoder_ffn_mult=2,
        clip_length=17,
        token_drop=3,
    )
    model.disable_tiling()
    model = widen_h3_vae(model)
    params = configure_stage1(model)
    x = torch.rand(1, 4, 17, 32, 32, dtype=torch.float32)
    moments = model._encode_clip(rgba_to_vae_input(x))
    latent = moments.chunk(2, dim=1)[0]
    pred = model._decode_clip(latent)[:, :, model.frame_pre_padding:]
    assert pred.shape == x.shape
    rgba = vae_output_to_rgba(pred)
    loss, parts = rgba_reconstruction_loss(rgba, x)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.conv_in.weight.grad is not None
    assert model.decoder.proj_out.weight.grad is not None
    assert all(p.grad is not None for p in params)
    assert model.encoder.conv_in.weight.grad[:, :3].abs().sum() == 0
    assert model.decoder.proj_out.weight.grad[:3072].abs().sum() == 0

    # Diffusion training follows the public 17n+5 temporal API, not
    # the VAE-only 17-frame single-clip API.
    with torch.no_grad():
        x22 = torch.rand(1, 4, 22, 32, 32)
        z22 = model.encode(rgba_to_vae_input(x22)).latent_dist.mode()
        assert z22.shape[2] == 7
        out22 = model.decode(z22).sample
        assert out22.shape == x22.shape
