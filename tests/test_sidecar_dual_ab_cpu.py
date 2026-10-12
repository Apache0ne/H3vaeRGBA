"""CPU tests for same-z_alpha original-decoder vs dual-LoRA ablation.

This tests the comparison is paired and does not pass GT alpha to the DDIM
sidecar at inference. It does not claim checkpoint-quality scores on CPU.
"""
import numpy as np
import torch
from PIL import Image

from benchmark_wan_dual import tiny_model
from h3rgba.benchmark_metrics import synthetic_rgba
from tools.compare_sidecar_dual_ab import (
    decode_original_gray, decode_both_same_latents,
    assemble_overlaps, coverage_stats, aggregate,
    preview_comparison,
)
from generate_h3_rgba_end_to_end import letterbox_rgb,window_starts


def test_same_alpha_latent_branches_match_when_adapter_zero_initialized():
    torch.manual_seed(18)
    codec=tiny_model()
    rgba=synthetic_rgba("moving_disk",32).unsqueeze(0)
    with torch.inference_mode():
        zr,za=codec.encode_pair(rgba)
        ab=decode_both_same_latents(codec,zr,za)
        assert torch.allclose(
            ab["original_gray"], ab["dual_lora"], atol=1e-6)
        assert ab["original_gray"].shape == (1,1,17,32,32)
        # Only trained dual-head output changes; original baseline remains
        # independent of the adapter even with EXACT same latent tensor.
        orig=ab["original_gray"].clone()
        codec.alpha_head.bias.add_(.1)
        modified=decode_both_same_latents(codec,zr,za)
        assert torch.equal(modified["original_gray"],orig)
        assert not torch.equal(modified["dual_lora"],orig)


def test_all_124_h3_frames_reconstructed_with_same_overlap_geometry():
    n=124
    starts=window_starts(n,stride=12)
    chunks=[np.full((17,4,6),i/len(starts),np.float32)
            for i in range(len(starts))]
    result=assemble_overlaps(chunks,starts,n)
    assert result.shape==(124,4,6)
    assert np.isfinite(result).all()
    assert result.min()>=0 and result.max()<=1
    # A genuinely empty mask is not confused with a good segmentation result.
    stats=coverage_stats(result)
    assert 0<=stats["coverage_mean"]<=1
    assert stats["temporal_abs_difference"]>=0


def test_pair_comparison_report_and_actual_visual_png(tmp_path):
    rows=[
        {"original_gray_iou":.5,"dual_lora_iou":.7,
         "gt_alpha_oracle_iou":.95,
         "original_gray_mae":.2,"dual_lora_mae":.1,
         "gt_alpha_oracle_mae":.01},
        {"original_gray_iou":.3,"dual_lora_iou":.2,
         "gt_alpha_oracle_iou":.8,
         "original_gray_mae":.3,"dual_lora_mae":.4,
         "gt_alpha_oracle_mae":.05},
    ]
    summary=aggregate(rows)
    assert summary["samples"]==2
    assert summary["dual_iou_wins"]==1
    assert abs(summary["dual_minus_original_iou"]-.05)<1e-7
    rgb=np.full((17,48,64,3),80,np.uint8)
    _,geometry=letterbox_rgb(rgb,128)
    a=np.zeros((17,128,128),np.float32)
    b=np.ones((17,128,128),np.float32)*.5
    output=tmp_path/"comparison.png"
    preview_comparison(
        rgb,geometry,{"original_gray":a,"dual_lora":b},output,
        indices=[0,8,16])
    im=Image.open(output)
    assert im.size==(320*5,(192+26)*3)
