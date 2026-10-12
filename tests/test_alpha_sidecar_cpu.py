"""CPU regression tests for the newly trained H3 alpha-diffusion sidecar.

No pretrained 33B H3 model is loaded here; separate A100 inference is required.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from h3rgba.alpha_latent_diffusion import (
    ConditionalAlphaDenoiser, cosine_alphas, noise_loss, sample,
    normalized_latents, unnormalized_latents
)
from generate_h3_rgba_end_to_end import (
    window_starts, blend_weights, letterbox_rgb, recover_mask,
    clean_foreground, encode_rgba_pngs
)


def test_denoiser_conditional_noise_learning_backprop():
    torch.manual_seed(33)
    denoiser=ConditionalAlphaDenoiser(hidden=16,blocks=2,time_dim=32)
    z0=torch.randn(2,24,7,4,4)
    cond=torch.randn_like(z0)
    schedule=cosine_alphas(100)
    loss,parts=noise_loss(denoiser,z0,cond,schedule)
    assert torch.isfinite(loss)
    loss.backward()
    assert denoiser.output[-1].weight.grad is not None
    assert denoiser.output[-1].weight.grad.abs().sum()>0
    assert 0<=parts["t_mean"]<100
    assert torch.isfinite(denoiser(z0,cond,torch.tensor([1,90]))).all()


def test_ddim_sampling_depends_on_condition_and_seed():
    torch.manual_seed(1)
    model=ConditionalAlphaDenoiser(hidden=16,blocks=2,time_dim=32).eval()
    cond=torch.randn(1,24,7,4,4)
    sched=cosine_alphas(40)
    z1=sample(model,cond,sched,steps=8,seed=12)
    z2=sample(model,cond,sched,steps=8,seed=12)
    z3=sample(model,cond,sched,steps=8,seed=13)
    assert z1.shape==cond.shape
    assert torch.equal(z1,z2)
    assert not torch.equal(z1,z3)
    assert torch.isfinite(z1).all()
    # The sidecar must be able to learn conditional dependence; random
    # initial final-layer zero init is not itself evidence of trained quality.
    with torch.no_grad():
        model.output[-1].weight.fill_(.001)
    one=sample(model,cond,sched,steps=6,seed=12)
    other=sample(model,torch.zeros_like(cond),sched,steps=6,seed=12)
    assert not torch.equal(one,other)


def test_latent_statistics_roundtrip():
    class Config:
        latents_mean=[.3]*24
        latents_std=[1.7]*24
    class Fake:
        config=Config()
    z=torch.randn(1,24,7,3,4)
    normalized=normalized_latents(z,Fake())
    restored=unnormalized_latents(normalized,Fake())
    assert torch.allclose(restored,z,atol=1e-6)


def test_generate_windows_cover_full_h3_124_frames_without_holes():
    starts=window_starts(124,stride=12)
    assert starts[0]==0
    assert starts[-1]==107
    assert len(starts)<20
    w=blend_weights()
    assert np.all(w>0)
    coverage=np.zeros(124,np.float32)
    for start in starts:
        coverage[start:start+17]+=w
    assert coverage.min()>0
    assert window_starts(17)==[0]
    with pytest.raises(ValueError):
        window_starts(16)


def test_letterbox_inverse_and_png_alpha(tmp_path):
    src=np.full((17,64,96,3),100,dtype=np.uint8)
    square,geometry=letterbox_rgb(src,64)
    assert square.shape==(17,64,64,3)
    _,_,rh,rw=geometry
    assert (rh,rw)==(43,64)
    alpha_square=np.zeros((64,64),np.float32)
    t,l,hh,ww=geometry
    alpha_square[t:t+hh,l:l+ww]=.6
    original=recover_mask(alpha_square,geometry,64,96)
    assert original.shape==(64,96)
    assert abs(float(original.mean())-.6)<.04
    assert np.all(clean_foreground(src,original,"retain")==src)
    better=clean_foreground(src,original,"black")
    assert better.mean()>src.mean()
    info=encode_rgba_pngs(src,np.repeat(alpha_square[None],17,axis=0),
                          tmp_path,geometry,background="black")
    assert info["png_frames"]==17
    assert 0 < info["hard_alpha_coverage_mean"] <= 1
    png=Image.open(tmp_path/"frame_00000.png")
    assert png.mode=="RGBA"
    assert png.getchannel("A").getextrema()[1]>0
