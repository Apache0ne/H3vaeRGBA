"""Load Stage-1 alpha-only adapter into a freshly widened H3 VAE."""
from pathlib import Path
import torch


def load_stage1_alpha(vae, path):
    from safetensors.torch import load_file
    if vae.encoder.conv_in.in_channels != 4:
        raise ValueError("Widen the original VAE before loading the alpha adapter")
    path = Path(path)
    weights = load_file(str(path), device="cpu")
    block = vae.decoder.proj_out.out_features // 4
    target = {
        "encoder.conv_in.weight.alpha": vae.encoder.conv_in.weight[:, 3:],
        "decoder.proj_out.weight.alpha": vae.decoder.proj_out.weight[3 * block:],
    }
    if vae.decoder.proj_out.bias is not None:
        target["decoder.proj_out.bias.alpha"] = vae.decoder.proj_out.bias[3 * block:]
    if set(weights) != set(target):
        raise ValueError(f"Adapter keys {sorted(weights)} do not match {sorted(target)}")
    with torch.no_grad():
        for name, out in target.items():
            if tuple(out.shape) != tuple(weights[name].shape):
                raise ValueError(f"Shape mismatch for {name}: {weights[name].shape} vs {out.shape}")
            out.copy_(weights[name].to(device=out.device, dtype=out.dtype))
    return vae
