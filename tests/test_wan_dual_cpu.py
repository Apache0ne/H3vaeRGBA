"""CPU regressions for Wan-inspired H3 dual-path alpha decoder LoRA."""
import json

import torch
from safetensors.torch import save_file

from h3rgba.wan_dual import DualAlphaCodec
from h3rgba.dual_loss import alpha_losses, alpha_scores


def tiny_codec():
    from diffusers import AutoencoderKLMiniMaxH3
    model = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8,8,8,8,8,8),
        layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        decoder_ffn_mult=2, clip_length=17, token_drop=3,
    )
    model.disable_tiling()
    model.enable_gradient_checkpointing()
    # Released pretrained H3 gates are trained nonzero; random tiny ones
    # initialize at zero, which would block LoRA gradients.
    with torch.no_grad():
        for block in model.decoder.transformer_blocks:
            block.scale1.fill_(.25)
            block.scale2.fill_(.25)
    return DualAlphaCodec(model, rank=2)


def test_dual_alpha_lora_backprop_preserves_rgb_and_saves_adapter(tmp_path):
    torch.manual_seed(22)
    codec = tiny_codec()
    codec.train()
    target = torch.rand(1,4,17,32,32)
    # Paint a clear alpha silhouette, with soft edges.
    target[:,3:4] = torch.zeros_like(target[:,3:4])
    target[:,3:4,:,:,8:24] = .9
    with torch.no_grad():
        z_rgb,z_alpha = codec.encode_pair(target)
        original_rgb = codec.decode_rgb(z_rgb)
    pred_alpha = codec.decode_alpha(z_rgb,z_alpha)
    assert pred_alpha.shape == (1,1,17,32,32)
    loss,metrics = alpha_losses(pred_alpha,target[:,3:4],target[:,:3])
    assert torch.isfinite(loss)
    loss.backward()
    trainable = codec.trainable_parameters()
    assert len(trainable) >= 8
    assert codec.merger.weight.grad is not None
    assert codec.alpha_head.weight.grad is not None
    assert any("lora_up" in name and p.grad is not None and p.grad.abs().sum() > 0
               for name,p in codec.named_parameters())
    assert all(p.grad is None for name,p in codec.vae.named_parameters()
               if "lora_down" not in name and "lora_up" not in name)
    optim = torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0.)
    optim.step()
    with torch.no_grad():
        new_rgb = codec.decode_rgb(z_rgb)
    assert torch.allclose(original_rgb,new_rgb,atol=1e-6), (
        "Frozen RGB decoder must not be affected by alpha LoRA optimizer step"
    )
    save = tmp_path/"dual_lora.safetensors"
    before = codec.adapter_state()
    save_file(before,str(save))
    fresh = tiny_codec()
    fresh.load_adapter(save)
    after = fresh.adapter_state()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[key],after[key]) for key in before)


def test_dual_alpha_head_starts_as_exact_rgb_mean():
    torch.manual_seed(12)
    codec=tiny_codec()
    decoded=torch.randn(1,3,5,8,8)
    from h3rgba.vae import image_net_stats
    mean,std=image_net_stats(decoded)
    expected=(decoded*std+mean).mean(dim=1,keepdim=True)
    result=codec.alpha_head(decoded)
    assert torch.allclose(result,expected,atol=1e-6)


def test_dual_loss_gradients_and_mask_iou():
    p=torch.ones(1,1,3,8,8,requires_grad=True)*.7
    gt=torch.zeros(1,1,3,8,8)
    gt[...,2:6,2:6]=1.
    rgb=torch.rand(1,3,3,8,8)
    loss,components=alpha_losses(p,gt,rgb)
    assert all(torch.isfinite(x) for x in components.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert alpha_scores(gt,gt)["alpha_iou"] == 1.
    assert 0 <= alpha_scores(p,gt)["alpha_iou"] <= 1.
