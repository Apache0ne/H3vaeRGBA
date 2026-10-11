"""No-GPU image and metric regression checks for RGB baseline diagnostic."""
import torch
from PIL import Image

from diagnose_vae import (comparison_metrics, save_comparison, rgb_to_h3, h3_to_rgb, encode_alpha_ablation)


def test_rgb_convention_and_rgb_baseline_metric():
    torch.manual_seed(3)
    rgb = torch.rand(1, 3, 17, 32, 32)
    assert torch.allclose(h3_to_rgb(rgb_to_h3(rgb)), rgb, atol=2e-7)
    alpha = torch.zeros(1, 1, 17, 32, 32)
    alpha[..., 8:24, 8:24] = 1
    gt = torch.cat((rgb, alpha), 1)
    orig = gt[:, :3].clone()
    pred = torch.cat((gt[:, :3], gt[:, 3:]), 1)
    metrics = comparison_metrics(gt, orig, pred[:, :3], pred[:, 3:])
    assert metrics["rgb_original_visible_mae"] == 0
    assert metrics["rgb_rgba_visible_mae"] == 0
    assert metrics["rgb_original_vs_rgba_mae"] == 0
    assert metrics["alpha_mae"] == 0
    assert metrics["alpha_iou"] == 1.


def test_comparison_png(tmp_path):
    gt = torch.zeros(4, 17, 32, 32)
    gt[0, :, 4:28, 4:28] = .8
    gt[3, :, 4:28, 4:28] = 1
    original = gt[:3].clone()
    path = save_comparison(gt, original, gt, tmp_path / "diagnose.png")
    assert path.is_file()
    assert Image.open(path).size == (5 * 32, 3 * 32)
    path6 = save_comparison(gt, original, gt, tmp_path / "ablation.png",
                            rgba_without_alpha=gt)
    assert Image.open(path6).size == (6 * 32, 3 * 32)


def test_alpha_ablation_detects_when_encoder_ignores_alpha():
    from diffusers import AutoencoderKLMiniMaxH3
    from h3rgba.vae import widen_h3_vae
    model = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        decoder_ffn_mult=2, clip_length=17, token_drop=3,
    )
    model.disable_tiling()
    model = widen_h3_vae(model)
    model.eval()
    x = torch.rand(1, 4, 17, 32, 32)
    # Zero-initialized alpha encoder channel cannot carry alpha signal.
    _, score = encode_alpha_ablation(model, x)
    assert score["alpha_input_latent_mae_delta"] == 0
    assert score["alpha_input_latent_relative_delta"] == 0
    assert 0 <= score["alpha_zeroed_iou"] <= 1
    with torch.no_grad():
        model.encoder.conv_in.weight[:, 3].fill_(0.1)
    _, changed = encode_alpha_ablation(model, x)
    assert changed["alpha_input_latent_mae_delta"] > 0


def test_widened_rgb_path_bitwise_preserved_with_alpha_perturbation_zeroed():
    from diffusers import AutoencoderKLMiniMaxH3
    from h3rgba.vae import widen_h3_vae
    torch.manual_seed(17)
    model = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        decoder_ffn_mult=2, clip_length=17, token_drop=3,
    )
    model.disable_tiling()
    model.eval()
    rgb = torch.rand(1, 3, 17, 32, 32)
    normalized = rgb_to_h3(rgb)
    with torch.no_grad():
        original = model._decode_clip(model._encode_clip(normalized).chunk(2, 1)[0])
    model = widen_h3_vae(model)
    no_alpha_signal = torch.cat((normalized, torch.zeros_like(normalized[:, :1])), dim=1)
    with torch.no_grad():
        expanded = model._decode_clip(model._encode_clip(no_alpha_signal).chunk(2, 1)[0])
    assert torch.equal(original, expanded[:, :3]), (
        "Widening must preserve every original pretrained RGB pixel when "
        "the added alpha encoder channel is zero."
    )
