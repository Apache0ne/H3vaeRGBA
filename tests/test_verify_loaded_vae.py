"""Verify the *loaded tensors*, not merely the presence of checkpoint files."""
import json
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tools import verify_loaded_vae


class TinyOriginalVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv3d(3, 5, 3)
        self.decoder = nn.Module()
        self.decoder.proj_in = nn.Linear(7, 5)
        self.decoder.proj_out = nn.Linear(5, 3 * 4 * 16 * 16)


def make_snapshot(tmp_path, original):
    folder = tmp_path / "vae"
    folder.mkdir(parents=True)
    weight_map = {}
    weights = {}
    for name, p in original.named_parameters():
        weights[name] = p.detach().clone().cpu().float()
        weight_map[name] = "diffusion_pytorch_model-00001-of-00003.safetensors"
    save_file(weights, folder / "diffusion_pytorch_model-00001-of-00003.safetensors")
    (folder / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return tmp_path


def test_loaded_pretrained_weights_match_exactly_and_detect_random_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_loaded_vae, "check_snapshot", lambda snapshot: True)
    original = TinyOriginalVAE()
    path = make_snapshot(tmp_path, original)
    # Exact checkpoint-to-BF16 conversion must be accepted.
    loaded = TinyOriginalVAE().to(dtype=torch.bfloat16)
    loaded.load_state_dict({k: v.to(torch.bfloat16)
                            for k, v in original.state_dict().items()})
    report = verify_loaded_vae.verify_loaded_vae(loaded, path, verbose=False)
    assert report["result"] == "OFFICIAL_H3_PRETRAINED_WEIGHTS_MATCH_IN_MEMORY"
    assert "encoder.conv_in.weight" in report["verified_tensor_names"]
    assert "decoder.proj_out.weight" in report["verified_tensor_names"]

    # A random/altered in-memory tensor must fail instead of giving false confidence.
    with torch.no_grad():
        loaded.decoder.proj_out.weight[0, 0] += 1
    with pytest.raises(RuntimeError, match="OFFICIAL_H3_WEIGHT_MISMATCH"):
        verify_loaded_vae.verify_loaded_vae(loaded, path, verbose=False)
