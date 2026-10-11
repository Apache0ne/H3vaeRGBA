"""Small native LoRA for H3's ViT video decoder (no PEFT dependency).

Only decoder transformer linear layers are wrapped; the pretrained weights stay
frozen and bit-identical at initialization. LoRA matrices use FP32 master
weights with BF16 autocast for the rest of the VAE.
"""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=None):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRALinear requires an nn.Linear and positive rank")
        self.base = base
        self.base.requires_grad_(False)
        self.enabled = True
        self.rank = int(rank)
        self.scale = float(rank if alpha is None else alpha) / rank
        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features,
                                                  device=base.weight.device, dtype=torch.float32))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank,
                                                device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        if not self.enabled:
            return original
        # Cast the intermediate only, not the full model or base weights.
        y = F.linear(x.to(self.lora_down.dtype), self.lora_down)
        y = F.linear(y, self.lora_up) * self.scale
        return original + y.to(original.dtype)


def add_decoder_lora(vae, rank=8, alpha=None):
    """Target q/k/v/o plus decoder MLP linear layers in all H3 blocks.

    The names are checked against real H3's decoder architecture. This is
    intentionally restricted to decoder.transformer_blocks; it doesn't modify
    encoder or latent channel count.
    """
    if rank < 1:
        raise ValueError("rank must be positive")
    decoder = vae.decoder
    count = 0
    for name, module in list(decoder.named_modules()):
        if not name.startswith("transformer_blocks."):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if ".attn.to_q" not in name and ".attn.to_k" not in name and ".attn.to_v" not in name and ".attn.to_out.0" not in name and ".ff." not in name:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = decoder.get_submodule(parent_name)
        parent._modules[child_name] = LoRALinear(module, rank=rank, alpha=alpha)
        count += 1
    if count == 0:
        raise RuntimeError("No H3 decoder Linear layers matched the LoRA target paths")
    return count


def decoder_lora_parameters(vae):
    return [p for module in vae.decoder.modules() if isinstance(module, LoRALinear)
            for p in (module.lora_down, module.lora_up)]


def decoder_lora_state(vae):
    state = {}
    for name, module in vae.decoder.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_down"] = module.lora_down.detach().cpu().contiguous()
            state[f"{name}.lora_up"] = module.lora_up.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("No LoRA adapters in decoder")
    return state


def load_decoder_lora_state(vae, path):
    from safetensors.torch import load_file
    data = load_file(str(path), device="cpu")
    expected = {}
    for name, module in vae.decoder.named_modules():
        if isinstance(module, LoRALinear):
            expected[f"{name}.lora_down"] = module.lora_down
            expected[f"{name}.lora_up"] = module.lora_up
    if set(data) != set(expected):
        raise ValueError(f"LoRA keys mismatch: missing={sorted(set(expected)-set(data))[:8]}; "
                         f"unexpected={sorted(set(data)-set(expected))[:8]}")
    with torch.no_grad():
        for name, target in expected.items():
            value = data[name]
            if value.shape != target.shape:
                raise ValueError(f"Shape mismatch for {name}: {value.shape} vs {target.shape}")
            target.copy_(value.to(device=target.device, dtype=target.dtype))


@contextmanager
def temporarily_disable_decoder_lora(vae):
    """Use only the frozen original pretrained decoder linears for a teacher pass.

    Does not change any parameter, gradients or optimizer state. Restoration
    occurs even if the forward pass raises.
    """
    modules = [m for m in vae.decoder.modules() if isinstance(m, LoRALinear)]
    old = [m.enabled for m in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, flag in zip(modules, old):
            module.enabled = flag
