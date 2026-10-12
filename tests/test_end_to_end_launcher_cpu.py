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


def test_heartbeat_prints_status_even_when_child_silent(tmp_path, monkeypatch, capsys):
    import sys
    import json
    import tools.launch_h3_rgba_a100 as mod

    monkeypatch.setenv("H3_E2E_LOG_PATH",str(tmp_path/"launcher.log"))
    monkeypatch.setattr(mod,"_status_snapshot",
        lambda **kw: {"phase":kw["phase"],"pid":kw["pid"],
                      "free_gb_content":60., "child_process":"sleeping"})
    mod.launch([sys.executable,"-u","-c",
                "import time; time.sleep(.45); print('done')"],
               heartbeat_seconds=.1)
    out=capsys.readouterr().out
    assert "[H3 STAGE START]" in out
    assert "[H3 HEARTBEAT]" in out
    assert "done" in out
    status=json.loads((tmp_path/"launcher_status.json").read_text())
    assert status["phase"]=="completed"
    assert status["exit_code"]==0
    assert "H3 HEARTBEAT" in (tmp_path/"launcher.log").read_text()


def test_child_errors_keep_full_nested_output_in_log(tmp_path,monkeypatch):
    import sys
    import pytest
    import tools.launch_h3_rgba_a100 as mod

    monkeypatch.setenv("H3_E2E_LOG_PATH",str(tmp_path/"launcher.log"))
    monkeypatch.setattr(mod,"_status_snapshot",
        lambda **kw: {"phase":kw["phase"],"pid":kw["pid"]})
    with pytest.raises(RuntimeError,match="CHILD_PROCESS_ERROR_TAIL"):
        mod.launch([sys.executable,"-u","-c",
                    "import sys;print('hidden-trace');sys.exit(4)"],
                   heartbeat_seconds=.1)
    assert "hidden-trace" in (tmp_path/"launcher.log").read_text()
    assert "EXIT" in (tmp_path/"launcher.log").read_text() or (
        "[H3 STAGE END]" in (tmp_path/"launcher.log").read_text()
    )


def test_reuse_preinstalled_venv_without_running_uv(tmp_path):
    from tools.launch_h3_rgba_a100 import a100_venv_ready
    assert not a100_venv_ready(tmp_path/"absent/python")
