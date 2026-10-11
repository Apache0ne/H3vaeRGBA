"""Wan-style transparency losses for dual-decoder H3 alpha reconstruction.

The RGB decoder is frozen. Only alpha decoder LoRA, the RGB/alpha latent
merger and the small output head receive gradients.
"""
import torch
import torch.nn.functional as F


def sobel_2d(video):
    if video.ndim != 5 or video.shape[1] != 1:
        raise ValueError("Expected scalar [B,1,T,H,W] mask")
    kernel_x = video.new_tensor([[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]).view(1, 1, 1, 3, 3) / 8.
    kernel_y = kernel_x.transpose(-1, -2).contiguous()
    # Replicate padding prevents arbitrary zero halos around video edges.
    padded = F.pad(video, (1, 1, 1, 1, 0, 0), mode="replicate")
    return F.conv3d(padded, kernel_x), F.conv3d(padded, kernel_y)


def alpha_losses(pred, gt, foreground, *, edge_weight=2., composite_weight=1.,
                 alpha_weight=2., temporal_weight=.2, hard_weight=.25):
    """Continuous alpha, Sobel, hard silhouette, temporal and random-bg loss.

    RGB foreground stays DETACHED; no LoRA can degrade H3's original RGB
    path. We use several high-contrast backgrounds, as Wan-Alpha does.
    """
    if pred.shape != gt.shape or pred.ndim != 5 or pred.shape[1] != 1:
        raise ValueError("Expected paired alpha masks [B,1,T,H,W]")
    if foreground.shape != (gt.shape[0], 3, *gt.shape[2:]):
        raise ValueError("Foreground must be [B,3,T,H,W]")
    p, t = pred.float(), gt.float().clamp(0, 1)
    fg = foreground.detach().float().clamp(0, 1)
    l1 = (p - t).abs().mean()
    gx_p, gy_p = sobel_2d(p)
    gx_t, gy_t = sobel_2d(t)
    edge = (gx_p - gx_t).abs().mean() + (gy_p - gy_t).abs().mean()
    # Smooth but decisive weight on silhouette mistakes near alpha boundaries.
    hard = F.binary_cross_entropy_with_logits(
        torch.logit(p.clamp(1e-4, 1 - 1e-4)), (t > .5).float()
    ) if hard_weight else p.new_zeros(())
    flicker = p.new_zeros(())
    if p.shape[2] > 1:
        flicker = ((p[:, :, 1:] - p[:, :, :-1]) -
                   (t[:, :, 1:] - t[:, :, :-1])).abs().mean()
    # L1 of composited RGB against same GT foreground, contrasting backgrounds.
    composites = []
    for background in (0., 1., .5):
        image_p = fg * p + background * (1 - p)
        image_t = fg * t + background * (1 - t)
        composites.append((image_p - image_t).abs().mean())
    composite = torch.stack(composites).mean()
    outside = (F.relu(-p) + F.relu(p - 1)).mean()
    total = (alpha_weight*l1 + edge_weight*edge +
             composite_weight*composite + temporal_weight*flicker +
             hard_weight*hard + .5*outside)
    metrics = {"alpha_l1":l1,"sobel":edge,"composite":composite,
               "hard_bce":hard,"temporal":flicker,"alpha_out_of_range":outside}
    return total, metrics


@torch.no_grad()
def alpha_scores(pred, gt):
    p = pred.float().clamp(0, 1)
    t = gt.float().clamp(0, 1)
    if p.shape != t.shape:
        raise ValueError("Alpha evaluation shape mismatch")
    intersection = ((p > .5) & (t > .5)).sum()
    union = ((p > .5) | (t > .5)).sum()
    return {
        "alpha_mae": float((p-t).abs().mean()),
        "alpha_iou": float(intersection / union) if union else 1.0,
        "alpha_binary_accuracy": float(((p>.5)==(t>.5)).float().mean()),
    }
