"""CPU-only E2E launcher selection, recovery and source-path tests."""
from pathlib import Path

from tools.launch_h3_rgba_a100 import best_vae_lora, best_sidecar


def test_checkpoint_restore_prefers_final_best_and_does_not_invent(tmp_path):
    drive=tmp_path/"Drive"
    drive.mkdir()
    local=tmp_path/"local/checkpoints/dual_lora_best.safetensors"
    assert best_vae_lora(drive,local) is None
    final=drive/"H3_DualLoRA_Results/dual_lora_best.safetensors"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"real_file_here")
    assert best_vae_lora(drive,local)==final
    local.parent.mkdir(parents=True)
    local.write_bytes(b"local_checkpoint")
    assert best_vae_lora(drive,local)==local


def test_complete_drive_snapshot_fallback(tmp_path):
    backup=tmp_path/"H3_DualLoRA_Backups/step_000500"
    nested=backup/"checkpoints"
    nested.mkdir(parents=True)
    (nested/"dual_lora_best.safetensors").write_bytes(b"best")
    local=tmp_path/"unused.safetensors"
    # An incomplete copy must never be trusted.
    assert best_vae_lora(tmp_path,local) is None
    (backup/"COMPLETE.json").write_text('{"step":500}')
    assert best_vae_lora(tmp_path,local)==nested/"dual_lora_best.safetensors"


def test_alpha_sidecar_best_restored_only_when_file_exists(tmp_path):
    local=tmp_path/"latest/sidecar_best.pt"
    assert best_sidecar(tmp_path,local) is None
    backup=tmp_path/"H3_AlphaDiffusion/checkpoints/sidecar_best.pt"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b"weights")
    assert best_sidecar(tmp_path,local)==backup
