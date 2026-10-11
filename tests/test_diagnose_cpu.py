"""No-GPU image and metric regression checks for RGB baseline diagnostic."""
import torch
from PIL import Image

from diagnose_vae import comparison_metrics, save_comparison, rgb_to_h3, h3_to_rgb


def test_rgb_convention_and_rgb_baseline_metric():
    torch.manual_seed(3)
    rgb = torch.rand(1, 3, 17, 32, 32)
    assert torch.allclose(h3_to_rgb(rgb_to_h3(rgb)), rgb, atol=2e-7)
    alpha = torch.zeros(1, 1, 17, 32, 32)
    alpha[..., 8:24, 8:24] = 1
    gt = torch.cat((rgb, alpha), 1)
    orig = gt[:, :3].clone()
    pred = torch.cat((gt[:, :3], gt[:, 3:]), 1)
    metrics = comparison_metrics(gt, orig, pred[:, :3], pred[:, 3:])
    assert metrics["rgb_original_visible_mae"] == 0
    assert metrics["rgb_rgba_visible_mae"] == 0
    assert metrics["rgb_original_vs_rgba_mae"] == 0
    assert metrics["alpha_mae"] == 0
    assert metrics["alpha_iou"] == 1.


def test_comparison_png(tmp_path):
    gt = torch.zeros(4, 17, 32, 32)
    gt[0, :, 4:28, 4:28] = .8
    gt[3, :, 4:28, 4:28] = 1
    original = gt[:3].clone()
    path = save_comparison(gt, original, gt, tmp_path / "diagnose.png")
    assert path.is_file()
    assert Image.open(path).size == (5 * 32, 3 * 32)
