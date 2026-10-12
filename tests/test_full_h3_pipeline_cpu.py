"""No-download, no-GPU tests of REAL H3 diffusion integration boundaries."""
import json
import subprocess
import sys

import pytest
import torch
from PIL import Image

from h3rgba.diffusion_contract import (
    AlphaLatentsNotGenerated, h3_video_shape,
    validate_two_generated_streams, denormalize_h3_video_latents,
)
from tools.h3_full_diffusion_probe import (
    resources, inspect_result, alpha_generation_contract,
)


def test_official_h3_124_frame_24_channel_layout():
    assert h3_video_shape(num_frames=124,height=192,width=320) == (
        1,24,37,12,20
    )
    assert h3_video_shape(num_frames=141,height=256,width=256) == (
        1,24,42,16,16
    )
    for frames in (17, 122, 125):
        with pytest.raises(ValueError):
            h3_video_shape(num_frames=frames,height=192,width=320)
    with pytest.raises(ValueError,match="multiples"):
        h3_video_shape(num_frames=124,height=190,width=320)


def test_real_h3_diffusion_without_alpha_rejects_fake_rgba_success():
    zr = torch.zeros(1,24,37,12,20)
    za = torch.rand_like(zr)
    with pytest.raises(AlphaLatentsNotGenerated,match="VAE LoRA alone cannot"):
        validate_two_generated_streams(
            zr,frames=124,height=192,width=320
        )
    with pytest.raises(AlphaLatentsNotGenerated,match="oracle VAE test"):
        validate_two_generated_streams(
            zr,za,alpha_source="supplied_ground_truth_alpha",
            frames=124,height=192,width=320
        )
    ok = validate_two_generated_streams(
        zr,za,alpha_source="trained_alpha_diffusion",
        frames=124,height=192,width=320
    )
    assert ok["generated_transparency_latents_available"]
    assert ok["alpha_latent_shape"] == [1,24,37,12,20]
    contract = alpha_generation_contract(
        has_supplied_ground_truth_alpha=True
    )
    assert not contract["can_claim_generated_alpha"]
    assert contract["can_evaluate_oracle_alpha_reconstruction"]


def test_h3_normalized_latent_bridge():
    z = torch.ones(1,24,37,12,20) * 2
    raw = denormalize_h3_video_latents(
        z,mean=[.5]*24,std=[3]*24
    )
    assert raw.shape == z.shape
    assert torch.allclose(raw,torch.full_like(raw,6.5))
    with pytest.raises(ValueError):
        denormalize_h3_video_latents(z,mean=[0]*24,std=[0]*24)


def test_generation_output_checks_real_video_audio_and_video_latents():
    frames=[Image.new("RGB",(320,192)) for _ in range(124)]
    state={"videos":[frames],"audio":[torch.zeros(1600)],
           "sampling_rate":32000, "latents":torch.zeros(1,24,37,12,20)}
    summary=inspect_result(state,requested_frames=124,
                           requested_width=320,requested_height=192)
    assert summary["denoised_video_latents_shape"] == [1,24,37,12,20]
    assert not summary["supports_true_rgba_generation"]
    assert summary["status"] == "REAL_H3_RGB_AUDIO_DIFFUSION_ONLY"
    with pytest.raises(RuntimeError,match="cannot verify"):
        inspect_result({k:v for k,v in state.items() if k!="latents"},
                       requested_frames=124,requested_width=320,
                       requested_height=192)
    wrong=dict(state, latents=torch.zeros(1,48,37,12,20))
    with pytest.raises(RuntimeError,match="latents shape"):
        inspect_result(wrong,requested_frames=124,
                       requested_width=320,requested_height=192)


def test_resource_guard_never_claims_l4_vram_can_replace_host_ram():
    proof=resources("/tmp",available_ram_gb=50,
                    free_disk_gb=220,cuda_available=True,bf16_available=True)
    assert not proof["can_attempt_official_int8_streamed_inference"]
    assert proof["checks"]["ram"] is False
    bigger=resources("/tmp",available_ram_gb=110,
                     free_disk_gb=220,cuda_available=True,bf16_available=True)
    assert bigger["can_attempt_official_int8_streamed_inference"]
    insufficient_disk=resources("/tmp",available_ram_gb=110,
                         free_disk_gb=100,cuda_available=True,bf16_available=True)
    assert not insufficient_disk["checks"]["disk"]


def test_preflight_cli_requires_no_model_weights_or_network(tmp_path):
    out=tmp_path/"preflight"
    p=subprocess.run([
        sys.executable,"tools/h3_full_diffusion_probe.py",
        "--preflight-only","--output-dir",str(out)
    ],text=True,capture_output=True,timeout=50)
    assert p.returncode == 0,p.stdout+"\n"+p.stderr
    report=json.loads((out/"h3_full_diffusion_report.json").read_text())
    assert report["status"]=="PREFLIGHT_ONLY_NO_WEIGHTS_LOADED"
    assert not report["loaded_real_diffusion_components"]
    assert not report["alpha_generation_contract"]["can_claim_generated_alpha"]
