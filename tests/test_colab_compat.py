"""No-GPU tests for Colab's optional TorchAO compatibility gate."""
import subprocess
import types

from tools import colab_compat


def test_no_torchao_is_acceptable(monkeypatch):
    monkeypatch.setattr(colab_compat, "package_version", lambda name: None)
    def forbid(*args, **kwargs):
        raise AssertionError("should not spawn an import for absent TorchAO")
    monkeypatch.setattr(colab_compat.subprocess, "run", forbid)
    assert colab_compat.check_torchao()


def test_incompatible_torchao_detected(monkeypatch, capsys):
    monkeypatch.setattr(colab_compat, "package_version", lambda name: "0.12.0")
    def fail_check(*args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "cannot import name 'FqnToConfig'")
    monkeypatch.setattr(colab_compat.subprocess, "run", fail_check)
    assert not colab_compat.check_torchao()
    assert "TORCHAO_INCOMPATIBLE" in capsys.readouterr().out


def test_compatible_torchao_not_removed(monkeypatch):
    monkeypatch.setattr(colab_compat, "package_version", lambda name: "0.17.0")
    def success(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, "TORCHAO_API_OK", "")
    monkeypatch.setattr(colab_compat.subprocess, "run", success)
    assert colab_compat.check_torchao()
