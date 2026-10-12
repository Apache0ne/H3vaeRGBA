#!/usr/bin/env python3
"""Final, held-out GT RGB + alpha evaluation of the trained H3 joint VAE.

The reserved clips NEVER influence the optimizer or checkpoint selection.
Only run this AFTER training the complete pipeline, for reporting. A high VAE
score is not evidence that the H3 DiT can generate correct novel alpha.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from h3rgba.dual_loss import alpha_scores
from h3rgba.wan_joint_h3 import WanJointH3Codec
from tools.validate_h3_wan_cache import sha256


def aggregate_holdout(items):
    if not items:
        raise ValueError("No reserved held-out videos; evaluation is invalid")
    metrics = ("alpha_iou", "alpha_mae", "alpha_binary_accuracy",
               "rgb_visible_l1", "composite_rgb_l1", "alpha_temporal_error")
    return {k: round(sum(float(row[k]) for row in items) / len(items), 6)
            for k in metrics}


def score_clip(rgba, prediction):
    gt_rgb = rgba[:, :3].float()
    gt_alpha = rgba[:, 3:4].float().clamp(0, 1)
    rgb = prediction["rgb"].float().clamp(0, 1)
    alpha = prediction["alpha"].float().clamp(0, 1)
    result = alpha_scores(alpha, gt_alpha)
    result["rgb_visible_l1"] = float(
        ((rgb - gt_rgb).abs() * gt_alpha).mean()
    )
    comps = []
    for bg in (0.0, 0.5, 1.0):
        actual = gt_rgb * gt_alpha + bg * (1.0 - gt_alpha)
        generated = rgb * alpha + bg * (1.0 - alpha)
        comps.append(float((actual - generated).abs().mean()))
    result["composite_rgb_l1"] = sum(comps) / len(comps)
    if gt_alpha.shape[2] > 1:
        result["alpha_temporal_error"] = float(
            ((alpha[:, :, 1:] - alpha[:, :, :-1]) -
             (gt_alpha[:, :, 1:] - gt_alpha[:, :, :-1])).abs().mean()
        )
    else:
        result["alpha_temporal_error"] = 0.0
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--base-vae-lora", required=True)
    p.add_argument("--joint-dual", required=True)
    p.add_argument("--joint-bridge", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--shift-limit", type=float, default=.35)
    p.add_argument("--bridge-scale", type=float, default=1.0)
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args()

    from run_vae_22gb import training_split
    records = json.loads(Path(args.manifest).read_text())["clips"]
    _, val = training_split(records, args.seed)
    unused_holdout = val[3:]
    if not unused_holdout:
        raise RuntimeError("No unseen reserved holdout samples")
    print("H3_WAN_FINAL_HOLDOUT_PLAN", json.dumps({
        "unused_reserved_clips": len(unused_holdout),
        "selection_validation_clips": min(3, len(val)),
        "no_retraining": True,
    }), flush=True)
    if args.preflight_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to load and evaluate actual H3 VAE")

    from train_wan_dual_lora import make_model
    from train_vae import PairedVideoDataset
    from train_h3_wan_joint import save_validation_preview

    dual, _ = make_model(
        argparse.Namespace(rank=4, merger_scale=.15, resolution=args.resolution),
        torch.device("cuda"),
    )
    dual.load_adapter(args.joint_dual)
    joint = WanJointH3Codec(
        dual, hidden=args.hidden, shift_limit=args.shift_limit,
        bridge_scale=args.bridge_scale, train_decoder_lora=False,
    ).cuda().eval()

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        PairedVideoDataset(unused_holdout, resolution=args.resolution,
                           deterministic=True),
        batch_size=1, shuffle=False, num_workers=0,
    )
    rows = []
    with torch.inference_mode():
        for i, rgba in enumerate(loader):
            rgba = rgba.cuda().float()
            output = joint(rgba)
            row = {"id": unused_holdout[i]["id"], **score_clip(rgba, output)}
            rows.append(row)
            if i == 0:
                save_validation_preview(
                    rgba, output, destination, step=0,
                )
            print("H3_WAN_FINAL_HOLDOUT_CLIP", json.dumps(row), flush=True)

    report = {
        "status": "HELD_OUT_VAE_RECONSTRUCTION_EVALUATED",
        "not_a_generated_video_alpha_score": True,
        "clip_count": len(rows),
        "metrics": aggregate_holdout(rows),
        "clips": rows,
        "manifest_sha256": sha256(args.manifest),
        "source_dual_adapter_sha256": sha256(args.base_vae_lora),
        "trained_joint_dual_sha256": sha256(args.joint_dual),
        "trained_joint_bridge_sha256": sha256(args.joint_bridge),
        "no_holdout_used_for_optimizer_or_checkpoint_selection": True,
    }
    (destination / "joint_vae_holdout_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print("H3_WAN_FINAL_HOLDOUT_COMPLETE", json.dumps(report["metrics"]), flush=True)


if __name__ == "__main__":
    main()
