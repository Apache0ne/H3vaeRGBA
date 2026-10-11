"""Verify in-memory original RGB H3 VAE tensors against official safetensors.

Checkpoint *presence* does not establish that from_pretrained() populated
model weights. This module compares loaded tensors exactly to their
checkpoint values after converting to the loaded module's dtype. It must run
BEFORE widening the encoder and decoder to four channels.

Uses memory-mapped safetensors, and copies one tensor at a time to CPU.
It does not download a model or modify model weights.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

from tools.verify_h3_cache import check_snapshot

# Both ends of the VAE must match the original pretrained checkpoint.
REQUIRED = ("encoder.conv_in.weight", "decoder.proj_out.weight")


def verify_loaded_vae(model, snapshot, *, verbose=True):
    """Raise RuntimeError unless selected actual model parameters exactly match
    the original pretrained files, with expected dtype conversion.
    """
    snapshot = Path(snapshot)
    if not check_snapshot(snapshot):
        raise RuntimeError("Official H3 VAE files missing/incomplete.")
    vae_dir = snapshot / "vae"
    index = json.loads((vae_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    named = dict(model.named_parameters())

    # Check several original encoder and decoder parameters, without reading
    # entire state_dict() copies or opening full-shard tensors simultaneously.
    preferred = list(REQUIRED) + [
        "encoder.conv_in.bias", "decoder.proj_in.weight",
        "decoder.proj_out.bias",
    ]
    verify_names = list(dict.fromkeys(preferred))
    for key in REQUIRED:
        if key not in named or key not in weight_map:
            raise RuntimeError(f"Required pretrained tensor not found: {key}")

    checked = []
    for key in verify_names:
        if key not in named or key not in weight_map:
            continue
        path = vae_dir / weight_map[key]
        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint shard for {key}: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as source:
            reference = source.get_tensor(key)
        loaded = named[key].detach().to("cpu")
        reference = reference.to(loaded.dtype)
        if loaded.shape != reference.shape:
            raise RuntimeError(
                f"OFFICIAL_H3_WEIGHT_MISMATCH {key}: "
                f"loaded={tuple(loaded.shape)} file={tuple(reference.shape)}"
            )
        # Byte-exact agreement following source->model dtype conversion.
        if not torch.equal(loaded, reference):
            max_err = float((loaded.float() - reference.float()).abs().max())
            raise RuntimeError(
                f"OFFICIAL_H3_WEIGHT_MISMATCH {key}: max_abs_diff={max_err}"
            )
        checked.append(key)
        if verbose:
            print(f"LOADED_WEIGHT_MATCH {key} shape={tuple(loaded.shape)} dtype={loaded.dtype}",
                  flush=True)
        del loaded, reference

    if any(key not in checked for key in REQUIRED):
        raise RuntimeError("Could not check both required encoder and decoder weights")
    report = {
        "checkpoint_snapshot": str(snapshot),
        "verified_tensor_names": checked,
        "comparison": "exact after checkpoint-to-loaded dtype conversion",
        "result": "OFFICIAL_H3_PRETRAINED_WEIGHTS_MATCH_IN_MEMORY",
    }
    if verbose:
        print(report["result"], flush=True)
    return report
