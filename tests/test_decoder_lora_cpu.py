"""CPU integration regression tests for H3 decoder LoRA, no full checkpoint download."""
import torch

from h3rgba.vae import widen_h3_vae, configure_stage1, rgba_to_vae_input, rgba_reconstruction_loss, vae_output_to_rgba
from h3rgba.decoder_lora import (
    add_decoder_lora, decoder_lora_parameters,
    decoder_lora_state, load_decoder_lora_state,
)


def tiny_h3():
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        decoder_ffn_mult=2, clip_length=17, token_drop=3,
    )
    vae.disable_tiling()
    return widen_h3_vae(vae)


def test_decoder_lora_initially_preserves_output_and_backpropagates(tmp_path):
    torch.manual_seed(2026)
    model = tiny_h3()
    alpha_params = configure_stage1(model)
    # Untrained H3 blocks have exactly ZERO residual gates, intentionally
    # preventing attention/MLP gradients. Released trained H3 weights should
    # have nonzero gates; simulate this in the tiny random CPU checkpoint.
    with torch.no_grad():
        for block in model.decoder.transformer_blocks:
            block.scale1.fill_(0.25)
            block.scale2.fill_(0.25)
    model.eval()
    x = torch.rand(1, 4, 17, 32, 32)
    with torch.no_grad():
        moments = model._encode_clip(rgba_to_vae_input(x))
        z = moments.chunk(2, 1)[0]
        original = model._decode_clip(z)
    count = add_decoder_lora(model, rank=4)
    assert count >= 4
    lora_params = decoder_lora_parameters(model)
    assert len(lora_params) == 2 * count
    with torch.no_grad():
        modified = model._decode_clip(z)
    assert torch.allclose(modified, original, atol=1e-6, rtol=1e-6)
    model.enable_gradient_checkpointing()
    reconstructed = model._decode_clip(model._encode_clip(rgba_to_vae_input(x)).chunk(2, 1)[0])
    reconstructed = reconstructed[:, :, model.frame_pre_padding:]
    loss, _ = rgba_reconstruction_loss(vae_output_to_rgba(reconstructed), x)
    loss.backward()
    assert model.decoder.proj_out.weight.grad is not None
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in lora_params)
    assert model.encoder.conv_in.weight.grad is not None
    assert model.decoder.proj_out.weight.grad[:3072].count_nonzero() == 0

    from safetensors.torch import save_file
    out = tmp_path / "test_lora.safetensors"
    state = decoder_lora_state(model)
    save_file(state, str(out))
    restored = tiny_h3()
    configure_stage1(restored)
    assert add_decoder_lora(restored, rank=4) == count
    load_decoder_lora_state(restored, out)
    new_state = decoder_lora_state(restored)
    assert state.keys() == new_state.keys()
    assert all(torch.equal(state[k], new_state[k]) for k in state)
