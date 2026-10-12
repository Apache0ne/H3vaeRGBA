"""Validate full dual LoRA command-line CPU dataset path and best adapter logic."""
import json
import subprocess
import sys

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from tools.dataset_audit import build_manifest
from train_wan_dual_lora import maybe_save_best


def write_video(path, *, matte=False):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),8.,(64,32))
    assert writer.isOpened()
    try:
        for index in range(20):
            pixels=np.zeros((32,64,3),np.uint8)
            if matte:
                pixels[6:24, 12:44] = 255
            else:
                pixels[:]=[30+index, 70, 110]
            writer.write(pixels)
    finally:
        writer.release()


def test_full_dual_trainer_cpu_manifest_preflight(tmp_path):
    fg=tmp_path/"data_fgr"
    a=tmp_path/"data_pha"
    fg.mkdir()
    a.mkdir()
    for i in range(2):
        write_video(fg/f"clip_{i}.mp4")
        write_video(a/f"clip_{i}.mp4",matte=True)
    m=tmp_path/"manifest.json"
    build_manifest(fg,a,m)
    result=subprocess.run(
        [sys.executable,"train_wan_dual_lora.py",
         "--manifest",str(m),"--skip-download",
         "--preflight-only","--resolution","64"],
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    assert "PREFLIGHT_OK" in result.stdout
    assert "DATA_SAMPLE" in result.stdout


def test_best_safetensors_follows_validation_alpha_iou(tmp_path):
    class Adapter:
        def adapter_state(self):
            return {"dummy.weight":torch.tensor([2.0])}
    v1={"step":0,"alpha_iou":0.4,"alpha_mae":0.3}
    v2={"step":1,"alpha_iou":0.5,"alpha_mae":0.2}
    v3={"step":2,"alpha_iou":0.45,"alpha_mae":0.1}
    assert maybe_save_best(Adapter(),tmp_path,v1)
    assert maybe_save_best(Adapter(),tmp_path,v2)
    assert not maybe_save_best(Adapter(),tmp_path,v3)
    best=json.loads((tmp_path/"checkpoints"/"best_validation.json").read_text())
    assert best==v2
    saved=load_file(str(tmp_path/"checkpoints"/"dual_lora_best.safetensors"))
    assert torch.equal(saved["dummy.weight"],torch.tensor([2.0]))
