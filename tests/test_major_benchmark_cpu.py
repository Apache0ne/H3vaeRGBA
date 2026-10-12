"""CPU unit and end-to-end tests for the independent major LoRA benchmark."""
import json
import subprocess
import sys

import pytest
import torch
from PIL import Image

from h3rgba.benchmark_metrics import (
    STRESS_CASES, binary_iou, boundary_f1, measure,
    summarize, synthetic_rgba,
)
from benchmark_wan_dual import (
    group_summary, original_alpha, one_video, paired_bootstrap, partition,
    save_previews, tiny_model,
)


def test_alpha_scoring_exact_and_wrong_threshold():
    gt = torch.zeros((1, 4, 3, 32, 32))
    gt[:, 3:4, :, 8:24, 8:24] = 1.
    good = gt[:, 3:4].clone()
    bad = torch.zeros_like(good)
    ideal = measure(good, gt)
    assert ideal["alpha_iou"] == 1.0
    assert ideal["alpha_mae"] == 0.
    assert ideal["alpha_boundary_f1_1px"] == 1.0
    assert ideal["composite_mae_black"] == 0.0
    assert ideal["alpha_temporal_delta_mae"] == 0.0
    wrong = measure(bad, gt)
    assert wrong["alpha_iou"] == 0.0
    assert wrong["alpha_boundary_f1_1px"] == 0.0
    assert wrong["alpha_mae"] > 0
    assert wrong["alpha_soft_iou"] == 0.


def test_partial_alpha_and_compositing_are_measured():
    gt = torch.ones(1, 4, 2, 32, 32)
    gt[:, 3:4] = 0.3
    pred = torch.ones(1, 1, 2, 32, 32) * 0.6
    m = measure(pred, gt)
    assert abs(m["alpha_partial_mae"] - .3) < 1e-6
    assert abs(m["alpha_mae"] - .3) < 1e-6
    assert m["alpha_gt_partial_fraction"] == 1
    assert m["alpha_iou"] == 0.0  # target alpha .3 is below threshold; pred .6 above
    assert abs(m["composite_mae_black"] - .3) < 1e-6
    # Use a foreground that does NOT match the background to expose halo.
    gt[:, :3] = .2
    m = measure(pred, gt)
    assert m["composite_mae_black"] > 0
    assert m["composite_mae_white"] > 0


def test_boundary_tolerance_and_missing_mask():
    gt = torch.zeros((1, 1, 3, 32, 32))
    gt[..., 8:24, 8:24] = 1
    shifted = torch.zeros_like(gt)
    shifted[..., 8:24, 9:25] = 1
    assert boundary_f1(gt, gt, 1) == 1
    assert boundary_f1(shifted, gt, 1) > boundary_f1(shifted, gt, 0)
    assert boundary_f1(torch.zeros_like(gt), gt, 1) == 0
    assert boundary_f1(torch.zeros_like(gt), torch.zeros_like(gt), 1) == 1


def test_procedural_suite_all_cases_finite_and_time_varying():
    assert len(STRESS_CASES) >= 9
    for case in STRESS_CASES:
        v = synthetic_rgba(case, 64)
        assert v.shape == (4, 17, 64, 64)
        assert torch.isfinite(v).all()
        assert 0 <= v.min() and v.max() <= 1
        assert torch.allclose(v[:3, 0], v[:3, -1])
    ring = synthetic_rgba("thin_ring", 64)
    assert ring[3].max() > .52
    disk = synthetic_rgba("moving_disk", 64)
    assert not torch.equal(disk[3, 0], disk[3, 8])
    assert synthetic_rgba("all_transparent", 64)[3].sum() == 0
    assert synthetic_rgba("all_opaque", 64)[3].min() == 1
    with pytest.raises(ValueError, match="Unknown"):
        synthetic_rgba("nonexistent", 64)


def test_split_matches_trainer_and_never_calls_seen_clips_unused():
    records = [{"id":str(i),"foreground":f"f_{i}","alpha":f"a_{i}"} for i in range(80)]
    train, seen, unused = partition(records, seed=42, previously_validated=3)
    assert len(train) == 68
    assert len(seen) == 3
    assert len(unused) == 9
    assert not ({x["id"] for x in train} & {x["id"] for x in seen + unused})
    assert not ({x["id"] for x in seen} & {x["id"] for x in unused})
    with pytest.raises(RuntimeError, match="Duplicate"):
        partition(records + [records[0]], seed=42)


def test_paired_bootstrap_and_macro_not_pooled_wrong():
    rows = [
        {"id":"a","baseline":{"alpha_iou":.2},"trained":{"alpha_iou":.4},
         "delta_alpha_iou":.2, "delta_alpha_mae":-.1},
        {"id":"b","baseline":{"alpha_iou":.5},"trained":{"alpha_iou":.4},
         "delta_alpha_iou":-.1, "delta_alpha_mae":-.1},
    ]
    q=paired_bootstrap(rows, "alpha_iou", rounds=100, seed=42)
    assert abs(q["mean_delta"] - .05) < 1e-7
    assert q["ci95"][0] <= q["mean_delta"] <= q["ci95"][1]
    assert group_summary([])["samples"] == 0
    actual = [
        {"alpha_iou":.2,"alpha_mae":.25,"pixels_intersection":1,"pixels_union":2},
        {"alpha_iou":1.,"alpha_mae":0.,"pixels_intersection":90,"pixels_union":90},
    ]
    assert abs(summarize(actual)["alpha_micro_iou"] - (91/92)) < 1e-8
    assert abs(summarize(actual)["alpha_iou"] - .6) < 1e-8


def test_tiny_h3_baseline_matches_initial_decoder_alpha_head():
    torch.manual_seed(102)
    codec = tiny_model()
    gt = synthetic_rgba("moving_disk", 32).unsqueeze(0)
    with torch.inference_mode():
        zr, za = codec.encode_pair(gt)
        teacher = original_alpha(codec, za)
        x = codec.decode_alpha(zr, za)
    assert torch.allclose(teacher, x.clamp(0,1), atol=1e-6)
    results = one_video(codec, gt)
    assert abs(results["delta_alpha_mae"]) < 1e-6
    assert abs(results["delta_alpha_iou"]) < 1e-6


def test_end_to_end_cpu_smoke_saves_json_csv_and_images(tmp_path):
    root = tmp_path / "benchmark"
    cmd = [
        sys.executable, "benchmark_wan_dual.py", "--cpu-smoke", "--stress-only",
        "--resolution", "32", "--stress-cases", "thin_ring,moving_disk",
        "--output-dir", str(root), "--bootstrap", "20",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    summary = json.loads((root/"summary.json").read_text())
    assert summary["mode"] == "CPU_RANDOM_INTEGRATION_ONLY"
    assert summary["groups"]["synthetic_stress"]["samples"] == 2
    assert len((root/"samples.jsonl").read_text().splitlines()) == 2
    assert (root/"samples.csv").exists()
    assert (root/"report.md").exists()
    assert len(list(root.rglob("contact_sheet.png"))) == 4
    for image in root.rglob("contact_sheet.png"):
        assert Image.open(image).size == (4 * 32, 2 * 3 * 32)
