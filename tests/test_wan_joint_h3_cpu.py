"""CPU tests of Wan-Alpha's ONE native diffusion latent -> TWO decoders H3 port."""
import argparse

import pytest
import torch

from benchmark_wan_dual import tiny_model
from h3rgba.wan_joint_h3 import WanJointH3Codec,joint_rgba_loss
from h3rgba.benchmark_metrics import synthetic_rgba


def setup(seed=13,train_decoder_lora=False):
    torch.manual_seed(seed)
    dual=tiny_model()
    # Existing 500-step dual VAE LoRA state is initialized, and can be frozen
    # or fine-tuned. The joint representation is new.
    model=WanJointH3Codec(
        dual,hidden=16,shift_limit=.35,
        train_decoder_lora=train_decoder_lora)
    return model


def test_wan_single_stream_has_exactly_24_latents_for_both_decoders():
    model=setup()
    rgba=synthetic_rgba("moving_disk",32)[None]
    with torch.no_grad():
        out=model(rgba)
        z=out["joint"]
        assert z.shape[1]==24
        assert z.shape==out["rgb_teacher_latent"].shape
        rgb,alpha=model.decode_joint(z)
        assert rgb.shape==(1,3,17,32,32)
        assert alpha.shape==(1,1,17,32,32)
        assert torch.allclose(rgb,out["rgb"])
        assert torch.allclose(alpha,out["alpha"])
        norm=model.normalized_diffusion_target(z)
        assert norm.shape==z.shape
        assert torch.isfinite(norm).all()


def test_original_rgb_decoder_gradients_reach_joint_merger_not_pretrained():
    model=setup(train_decoder_lora=False)
    rgba=synthetic_rgba("moving_disk",32)[None]
    out=model(rgba)
    loss,metrics=joint_rgba_loss(model,rgba,out)
    assert torch.isfinite(loss)
    loss.backward()
    grads=list(model.fuser.parameters())+list(model.alpha_bridge.parameters())
    assert any(p.grad is not None and p.grad.isfinite().all()
               for p in grads)
    assert model.dual.vae.encoder.conv_in.weight.grad is None
    assert model.dual.vae.decoder.proj_out.weight.grad is None
    assert "composite_rgb_l1" in metrics
    assert "normalized_rgb_latent_prior" in metrics


def test_shared_joint_inference_requires_NO_alpha_video():
    model=setup()
    rgba=synthetic_rgba("moving_disk",32)[None]
    with torch.no_grad():
        zjoint,_,_=model.encode_joint(rgba)
        # Neither decode input nor signature accepts ground-truth alpha.
        rgb,alpha=model.decode_joint(zjoint)
        assert rgb.shape[2]==alpha.shape[2]==17
        with pytest.raises(ValueError):
            model.decode_joint_alpha(torch.zeros(1,48,7,2,2))


def test_fuser_limits_shift_and_initially_preserves_rgb():
    model=setup()
    zrgb=torch.randn(1,24,7,2,2)
    zalpha=torch.randn_like(zrgb)
    with torch.no_grad():
        fused=model.fuse(zrgb,zalpha)
        assert torch.allclose(fused,zrgb,atol=1e-7)
        model.fuser.out.bias.fill_(2.)
        modified=model.fuse(zrgb,zalpha)
        assert (modified-zrgb).abs().max()<=model.shift_limit+1e-5
        assert not torch.allclose(modified,zrgb)


def test_joint_bridge_safetensors_roundtrip(tmp_path):
    from safetensors.torch import save_file
    one=setup()
    with torch.no_grad():
        one.fuser.out.bias.fill_(.23)
        one.alpha_bridge.out.bias.fill_(-.13)
    checkpoint=tmp_path/"bridge.safetensors"
    save_file(one.bridge_state(),str(checkpoint))
    other=setup()
    other.load_bridge(checkpoint)
    for k,v in one.bridge_state().items():
        assert torch.equal(v,other.bridge_state()[k])


def test_joint_loss_detects_wrong_video_shapes():
    model=setup()
    rgba=synthetic_rgba("moving_disk",32)[None]
    with torch.no_grad():
        out=model(rgba)
    with pytest.raises(ValueError):
        joint_rgba_loss(model,rgba[:,:3],out)
