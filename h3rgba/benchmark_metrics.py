"""Measurement primitives and deterministic stress-video generation for H3 alpha VAEs.

All scores are on straight (non-premultiplied) RGBA [B,4,T,H,W] with pixel
values 0..1. No model imports, downloads, or dataset access in this module.
"""
import math
from collections import defaultdict

import torch
import torch.nn.functional as F


def _assert_shapes(pred, gt):
    if pred.shape != gt.shape or pred.ndim != 5 or pred.shape[1] != 1:
        raise ValueError("Expected matching alpha [B,1,T,H,W] tensors")


def binary_iou(pred, gt, threshold=0.5):
    _assert_shapes(pred, gt)
    a, b = pred > threshold, gt > threshold
    inter = (a & b).sum(dtype=torch.float64)
    union = (a | b).sum(dtype=torch.float64)
    return float((inter / union).item()) if union.item() else 1.0


def edge_mask(alpha, threshold=0.5):
    """Spatial boundary pixels of hard alpha segmentation; time is not mixed."""
    binary = (alpha > threshold).float()
    high = F.max_pool3d(binary, (1, 3, 3), stride=1, padding=(0, 1, 1))
    low = -F.max_pool3d(-binary, (1, 3, 3), stride=1, padding=(0, 1, 1))
    return (high - low) > 0


def boundary_f1(pred, target, tolerance=1):
    """Boundary F1 with an explicit spatial tolerance, not frame mixing.

    Empty/empty boundary -> 1, missing one side -> 0.
    """
    _assert_shapes(pred, target)
    a, b = edge_mask(pred), edge_mask(target)
    na, nb = int(a.sum()), int(b.sum())
    if na + nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    if tolerance:
        k = 1 + tolerance * 2
        a_near = F.max_pool3d(a.float(), (1, k, k), stride=1, padding=(0, tolerance, tolerance)) > 0
        b_near = F.max_pool3d(b.float(), (1, k, k), stride=1, padding=(0, tolerance, tolerance)) > 0
    else:
        a_near, b_near = a, b
    precision = float((a & b_near).sum()) / na
    recall = float((b & a_near).sum()) / nb
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def measure(pred_alpha, target_rgba, pred_rgb=None):
    """Per-clip metrics; recompute IoU from framewise intersection, not averaged IoUs."""
    if target_rgba.ndim != 5 or target_rgba.shape[1] != 4:
        raise ValueError("Expected RGBA [B,4,T,H,W]")
    target = target_rgba[:, 3:4].float().clamp(0, 1)
    pred = pred_alpha.float().clamp(0, 1)
    _assert_shapes(pred, target)
    error = (pred - target).abs()
    hard_pred, hard_gt = pred > 0.5, target > 0.5
    intersection = (hard_pred & hard_gt).sum()
    union = (hard_pred | hard_gt).sum()
    intersection_value = int(intersection)
    union_value = int(union)
    total_fg = int(hard_gt.sum())
    result = {
        "alpha_mae": float(error.mean()),
        "alpha_rmse": float(error.square().mean().sqrt()),
        "alpha_iou": binary_iou(pred, target),
        "alpha_dice": (2 * intersection_value / (int(hard_pred.sum()) + total_fg))
                      if (int(hard_pred.sum()) + total_fg) else 1.0,
        "alpha_binary_accuracy": float((hard_pred == hard_gt).float().mean()),
        "alpha_boundary_f1_1px": boundary_f1(pred, target, 1),
        "alpha_soft_iou": float(torch.minimum(pred, target).sum() /
                                torch.maximum(pred, target).sum().clamp_min(1e-8)),
        "alpha_temporal_delta_mae": float(
            ((pred[:, :, 1:] - pred[:, :, :-1]) -
             (target[:, :, 1:] - target[:, :, :-1])).abs().mean()
        ) if target.shape[2] > 1 else 0.0,
        "alpha_gt_opaque_fraction": float((target > .95).float().mean()),
        "alpha_gt_transparent_fraction": float((target < .05).float().mean()),
        "alpha_gt_partial_fraction": float(((target >= .05) & (target <= .95)).float().mean()),
        "pixels_intersection": intersection_value,
        "pixels_union": union_value,
        "pixels_foreground": total_fg,
        "pixels_total": int(target.numel()),
    }
    partial = (target >= .05) & (target <= .95)
    result["alpha_partial_mae"] = float(error[partial].mean()) if partial.any() else None
    fg = target_rgba[:, :3].float().clamp(0, 1)
    composites = {}
    for name, bg in [("black", 0.0), ("gray", 0.5), ("white", 1.0)]:
        p = fg * pred + bg * (1 - pred)
        t = fg * target + bg * (1 - target)
        composites[name] = float((p - t).abs().mean())
    result["composite_mae_black"] = composites["black"]
    result["composite_mae_gray"] = composites["gray"]
    result["composite_mae_white"] = composites["white"]
    result["composite_mae_average"] = sum(composites.values()) / len(composites)
    if pred_rgb is not None:
        if pred_rgb.shape != fg.shape:
            raise ValueError("Reconstructed RGB and original foreground shape mismatch")
        result["rgb_visible_mae"] = float(
            (((pred_rgb.float().clamp(0,1) - fg).abs() * target).sum() /
             (3 * target.sum()).clamp_min(1e-8))
        )
    return result


def summarize(results):
    """Macro average over clips; also give pixel-weighted IoU and worst cases."""
    if not results:
        return {"samples": 0}
    metric_names = [k for k in results[0]
                    if k.startswith(("alpha_", "composite_", "rgb_")) and
                    k not in ("alpha_gt_opaque_fraction", "alpha_gt_transparent_fraction",
                              "alpha_gt_partial_fraction")]
    means = {}
    for key in metric_names:
        values = [float(row[key]) for row in results if row.get(key) is not None]
        if values:
            means[key] = sum(values) / len(values)
    intersection = sum(row.get("pixels_intersection", 0) for row in results)
    union = sum(row.get("pixels_union", 0) for row in results)
    means["alpha_micro_iou"] = intersection / union if union else 1.0
    means["samples"] = len(results)
    means["worst_three_iou"] = [
        {"id": row.get("id", ""), "alpha_iou": row["alpha_iou"]}
        for row in sorted(results, key=lambda v: v["alpha_iou"])[:3]
    ]
    return means


def synthetic_rgba(case, resolution=128, frames=17, seed=2026):
    """Procedural *unseen* alpha stress clips. Nothing comes from training data.

    Cases deliberately span spatial frequencies, alpha mixtures and movement.
    Uses only arithmetic SDF-like masks so video, GT masks and RGB align exactly.
    """
    if resolution < 32 or resolution % 16 or frames < 2:
        raise ValueError("Resolution must be divisible by 16, >=32; frames>=2")
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, resolution),
                            torch.linspace(-1, 1, resolution), indexing="ij")
    alpha_frames = []
    for index in range(frames):
        phase = index / max(frames - 1, 1)
        cx = 0.28 * math.sin(phase * 2 * math.pi)
        cy = 0.18 * math.cos(phase * 2 * math.pi)
        dx, dy = xx - cx, yy - cy
        distance = torch.sqrt(dx**2 + dy**2 + 1e-9)
        if case == "moving_disk":
            alpha = torch.sigmoid((.43 - distance) * resolution / 2)
        elif case == "thin_ring":
            alpha = torch.sigmoid((.015 - (distance - .51).abs()) * resolution / 1.8)
        elif case == "hair_strands":
            wave = xx - .10 * torch.sin(yy * 12. + phase * 7.)
            centers = torch.sin(wave * 38.) * torch.exp(-(xx**2 + yy**2))
            alpha = torch.sigmoid((centers - .78) * resolution / 12.) * (yy.abs() < .85)
        elif case == "soft_glass":
            alpha = .10 + .48 * torch.exp(-distance.square() / .17)
            alpha = alpha * torch.sigmoid((.7 - distance) * resolution / 5)
        elif case == "feathered_edge":
            alpha = torch.sigmoid((.4 - distance) * 22.)
        elif case == "motion_blur":
            shape = (.42 - (dx.abs() + dy.abs() * .4))
            alpha = torch.sigmoid(shape * 15.) * .77
        elif case == "checker_alpha":
            alpha = (torch.sin(xx*27. + phase*3.) * torch.sin(yy*27.) > 0).float()
            alpha = F.avg_pool2d(alpha[None, None], 3, stride=1, padding=1)[0, 0]
        elif case == "all_transparent":
            alpha = torch.zeros_like(xx)
        elif case == "all_opaque":
            alpha = torch.ones_like(xx)
        elif case == "partial_flat":
            alpha = torch.full_like(xx, .35)
        else:
            raise ValueError(f"Unknown synthetic stress case: {case}")
        alpha_frames.append(alpha.clamp(0, 1))
    alpha = torch.stack(alpha_frames, dim=0)  # T,H,W
    # Foreground has independent, stationary textured RGB, suitable for
    # testing whether the original H3 RGB reconstruction stays frozen.
    rgb = torch.stack((
        (.5 + .45 * torch.sin(xx*7 + yy*4)).clamp(0, 1),
        (.5 + .4 * torch.cos(xx*5 - yy*6)).clamp(0, 1),
        (.5 + .4 * torch.sin(xx*4 - yy*7)).clamp(0, 1),
    ), dim=0)[:,None].expand(-1,frames,-1,-1)
    return torch.cat((rgb, alpha[None]), dim=0).contiguous().float()


STRESS_CASES = (
    "moving_disk", "thin_ring", "hair_strands", "soft_glass", "feathered_edge",
    "motion_blur", "checker_alpha", "all_transparent", "all_opaque", "partial_flat",
)
