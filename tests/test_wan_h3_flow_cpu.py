"""CPU structural regression for real H3 packed-flow WAN-Alpha diffusion stage."""
import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from h3rgba.wan_h3_flow import (
    rectified_flow_batch,shifted_video_clean_time,h3_packed_flow_loss,
    inject_h3_transformer_lora,h3_dit_lora_state
)
from tools.cache_h3_wan_joint_targets import captions_for


class TinyH3PackedFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(patch_size=(1,2,2))
        self.video=nn.Linear(96,96)
    def forward(self,**kw):
        assert kw["hidden_states"].shape[2]==96
        assert kw["token_tags"].numel()==kw["position_ids"].shape[0]
        assert (kw["timestep"]>=0).all() and (kw["timestep"]<=1).all()
        assert kw["encoder_hidden_states"].shape[-1]==5120
        assert kw["audio_hidden_states"].shape==(1,0,32)
        return self.video(kw["hidden_states"]),torch.empty(1,0,32)


def test_rectified_flow_velocity_is_data_ward_not_noise_ward():
    clean=torch.ones(1,24,7,4,4)
    noise=torch.zeros_like(clean)
    noisy,target,t=rectified_flow_batch(clean,t=.3,noise=noise)
    assert torch.allclose(noisy,torch.full_like(clean,.3))
    assert torch.equal(target,clean)
    assert t.item()==pytest.approx(.3)
    with pytest.raises(ValueError):
        rectified_flow_batch(clean,t=1.)


def test_true_h3_rotary_packed_layout_and_velocity_backprop():
    torch.manual_seed(2)
    model=TinyH3PackedFlow()
    clean=torch.randn(1,24,7,4,4)
    embeddings=torch.randn(1,5,5120)
    tags=torch.ones(5,dtype=torch.long)
    loss,meta=h3_packed_flow_loss(
        model,clean,embeddings,tags,t=torch.tensor(.4),
        noise=torch.zeros_like(clean))
    assert torch.isfinite(loss)
    assert meta["video_rows"]==7*2*2
    assert meta["text_rows"]==5
    assert meta["audio_rows_unsupervised"]==0
    loss.backward()
    assert model.video.weight.grad is not None
    assert model.video.weight.grad.abs().sum()>0


def test_wrong_h3_embeds_rejected():
    model=TinyH3PackedFlow()
    clean=torch.randn(1,24,7,4,4)
    with pytest.raises(ValueError,match="Qwen3-VL"):
        h3_packed_flow_loss(model,clean,torch.randn(1,5,64),
                            torch.ones(5,dtype=torch.long))


def test_dit_lora_targets_only_attention_and_feedforward_layers():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.to_q=nn.Linear(8,8)
            self.to_k=nn.Linear(8,8)
            self.to_v=nn.Linear(8,8)
            self.to_out=nn.ModuleList([nn.Linear(8,8)])
    class FF(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.ModuleList([nn.Linear(8,16),nn.ReLU(),nn.Linear(16,8)])
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn=Attention()
            self.ff=FF()
    model=nn.Module()
    model.transformer_blocks=nn.ModuleList([Block(),Block()])
    count=inject_h3_transformer_lora(model,rank=2)
    assert count==12
    state=h3_dit_lora_state(model)
    assert len(state)==24
    assert all("lora_" in key for key in state)
    assert all(p.requires_grad==("lora_" in name)
               for name,p in model.named_parameters())


def test_genuine_caption_ids_required_without_guesses():
    items=[{"id":"sample_001"},{"id":"sample_002"}]
    assert captions_for(items,{"sample_001":"A fox with a transparent background",
                               "sample_002":"A flower on a transparent background"})==[
        "A fox with a transparent background","A flower on a transparent background"]
    with pytest.raises(ValueError,match="Missing REAL captions"):
        captions_for(items,{"sample_001":"unknown object"})


def test_h3_native_video_schedule_uses_shift12_and_data_ward_time():
    # Native inference maps u=0.5 to sigma=6/6.5, NOT t=0.5.
    assert float(shifted_video_clean_time(0.5)) == pytest.approx(1-6/6.5)
    assert float(shifted_video_clean_time(0.1)) > float(shifted_video_clean_time(0.9))
    with pytest.raises(ValueError):
        shifted_video_clean_time(1.0)
    sample=torch.zeros(1,24,7,4,4)
    gen=torch.Generator().manual_seed(1234)
    noisy,target,t=rectified_flow_batch(sample,generator=gen)
    assert 0 < float(t) < 1
    # Exact noise conservation for a clean-zero sample: target=-epsilon.
    assert torch.allclose(noisy, -(1-t)*target,atol=1e-6)
