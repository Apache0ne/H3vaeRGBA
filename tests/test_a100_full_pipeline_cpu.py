"""No-GPU regression tests for A100 80GB full H3 diffusion launch."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.h3_a100_full_t2va import inspect_host, import_upstream, PATTERNS, MODEL_ID, cached_t2va_weights_present, validate_qwen_processor


def test_a100_80gb_host_ram_53gb_is_allowed_with_phased_quantized_loading():
    proof=inspect_host(gpu_gib=79.1,ram_total_gb=53.,
                       ram_available_gb=47.,disk_free_gb=181.)
    assert proof["can_proceed"],proof
    assert proof["checks"]["host_ram_at_least_40gb"]
    assert "INT8" in proof["configuration"]
    # Old official full-bf16 CPU-offload guard would reject 53GB, but
    # phase-resident NF4+INT8 loader should not automatically reject it.
    too_small_gpu=inspect_host(gpu_gib=22.4,ram_total_gb=53.,
                        ram_available_gb=47.,disk_free_gb=181.)
    assert not too_small_gpu["can_proceed"]
    assert not too_small_gpu["checks"]["gpu_80gb_class"]
    out_of_disk=inspect_host(gpu_gib=79.1,ram_total_gb=53.,
                         ram_available_gb=47.,disk_free_gb=100.)
    assert not out_of_disk["checks"]["free_disk_155gb"]


def test_only_original_t2va_checkpoint_partitions_downloaded():
    assert MODEL_ID=="MiniMaxAI/MiniMax-H3"
    assert "transformer/*" in PATTERNS
    assert "text_encoder/*" in PATTERNS
    assert "vae/*" in PATTERNS
    assert "audio_vae/*" in PATTERNS
    assert "transformer_ref/*" not in PATTERNS
    assert "Ref2VA/*" not in PATTERNS


def test_a100_cli_preflight_works_without_model_or_gpu(tmp_path):
    output=tmp_path/"report"
    p=subprocess.run([
        sys.executable,"tools/h3_a100_full_t2va.py","--preflight-only",
        "--output-dir",str(output)
    ],capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
    assert "A100_PHASED_RESOURCE_PREFLIGHT" in p.stdout
    report=json.loads((output/"h3_a100_preflight.json").read_text())
    assert "checks" in report
    assert "available_host_ram_25gb" in report["checks"]


def test_requires_actual_upstream_runner_directory(tmp_path):
    with pytest.raises(FileNotFoundError,match="Missing external runner"):
        import_upstream(tmp_path)


def test_generation_disk_check_is_not_another_155gb_after_model_download():
    # Once 144 GB of weights are already on disk, 65 GB remaining is enough
    # for a generation working directory. Pre-download requires >=155 GB.
    first = inspect_host(gpu_gib=79.1, ram_total_gb=167,
                         ram_available_gb=150, disk_free_gb=65,
                         phase="download")
    after = inspect_host(gpu_gib=79.1, ram_total_gb=167,
                         ram_available_gb=150, disk_free_gb=65,
                         phase="generate")
    assert not first["can_proceed"]
    assert after["can_proceed"]
    assert not first["checks"]["free_disk_155gb"]
    assert after["checks"]["free_disk_10gb"]
    assert after["disk_required_decimal_gb"] == 10


def test_preexisting_complete_h3_weight_cache_detected_without_redownloading(tmp_path):
    home = tmp_path/"hf"
    snapshot=home/"hub/models--MiniMaxAI--MiniMax-H3/snapshots/revision"
    assert not cached_t2va_weights_present(home)
    for name in ("transformer","text_encoder","vae","audio_vae"):
        folder=snapshot/name
        folder.mkdir(parents=True)
        (folder/"config.json").write_text("{}")
        (folder/"weights.safetensors.index.json").write_text(
            json.dumps({"weight_map":{"weight":"weights.safetensors"}})
        )
        (folder/"weights.safetensors").write_bytes(b"x"*2048)
    assert cached_t2va_weights_present(home)
    (snapshot/"transformer/weights.safetensors").unlink()
    assert not cached_t2va_weights_present(home)


def test_invalid_disk_guard_phase_is_rejected():
    with pytest.raises(ValueError):
        inspect_host(gpu_gib=79.1,ram_total_gb=167,
                     ram_available_gb=160,disk_free_gb=20,phase="unknown")


def test_qwen_processor_fails_early_on_missing_torchvision(monkeypatch):
    # No giant model download, processor data, or CUDA needed.
    # Simulate missing torchvision even when PyTorch itself is installed.
    monkeypatch.setitem(sys.modules, "torchvision", None)
    with pytest.raises(RuntimeError, match="preflight failed BEFORE loading"):
        validate_qwen_processor()


def test_full_h3_requirements_pin_cuda_compatible_torchvision():
    requirements = (Path(__file__).resolve().parents[1] /
                    "requirements-a100-h3.txt").read_text()
    launcher = (Path(__file__).resolve().parents[1] /
                "tools/launch_h3_rgba_a100.py").read_text()
    assert "torchvision==0.24.0" in requirements
    assert '"torch==2.9.0","torchvision==0.24.0"' in launcher
    assert "https://download.pytorch.org/whl/cu128" in launcher
