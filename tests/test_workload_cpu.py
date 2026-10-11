import json
import subprocess
import sys

import cv2
import numpy as np
import torch
from PIL import Image

from train_vae import PairedVideoDataset
from run_vae_22gb import save_frames
from tools.dataset_audit import build_manifest


def write_mp4(path, frames, width=64, height=32, matte=False):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8., (width, height))
    assert writer.isOpened()
    for frame_idx in range(frames):
        sample = np.zeros((height, width, 3), dtype=np.uint8)
        if matte:
            sample[:, frame_idx % 20:20 + frame_idx % 20, :] = 255
        else:
            sample[:, :, 0] = 40 + frame_idx
            sample[:, :, 1] = 150
        writer.write(sample)
    writer.release()


def test_end_to_end_cpu_data_preflight(tmp_path):
    fg = tmp_path / "data_fgr"
    alpha = tmp_path / "data_pha"
    fg.mkdir()
    alpha.mkdir()
    for index in range(2):
        write_mp4(fg / f"clip_{index}.mp4", 20)
        write_mp4(alpha / f"clip_{index}.mp4", 20, matte=True)
    manifest = tmp_path / "manifest.json"
    result = build_manifest(fg, alpha, manifest)
    assert result["stats"]["valid"] == 2
    ds = PairedVideoDataset(result["clips"], resolution=64, deterministic=True)
    sample = ds[0]
    assert sample.shape == (4, 17, 64, 64)
    assert float(sample[3, :, :12, :].abs().max()) == 0
    assert torch.isfinite(sample).all()
    command = [sys.executable, "run_vae_22gb.py", "--manifest", str(manifest),
               "--preflight-only", "--resolution", "64"]
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "PREFLIGHT_OK" in proc.stdout


def test_rgba_preview_png_has_alpha(tmp_path):
    gt = torch.rand(4, 17, 32, 32)
    pred = gt.clone()
    pred[3] *= 0.5
    sheet = save_frames(gt, pred, tmp_path / "previews", max_frames=3)
    assert Image.open(sheet).size == (128, 192)
    rgba = Image.open(tmp_path / "previews" / "pred_000.png")
    assert rgba.mode == "RGBA"
    assert rgba.getchannel("A").getextrema()[1] < 255
