"""Experimental H3 RGBA VAE reconstruction trainer; BF16, single GPU.

Stage 1 only. Does NOT train H3's 33B DiT, and is not a validated
end-to-end transparent-video model. Always run --dry-run first.

Manifest created by tools/dataset_audit.py.
"""
import argparse
import json
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from h3rgba.vae import (
    widen_h3_vae, configure_stage1, rgba_to_vae_input,
    vae_output_to_rgba, rgba_reconstruction_loss,
)


class PairedVideoDataset(Dataset):
    def __init__(self, records, frames=17, resolution=128):
        if frames != 17:
            raise ValueError("The initial VAE-only runner uses 17-frame chunks.")
        if resolution < 64 or resolution % 32:
            raise ValueError("Resolution must be >=64 and divisible by 32.")
        self.records = records
        self.frames = frames
        self.resolution = resolution

    def __len__(self):
        return len(self.records)

    @staticmethod
    def read_frames(path, start, count):
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        try:
            for _ in range(count):
                good, frame = cap.read()
                if not good or frame is None:
                    raise RuntimeError(f"Could not decode {count} frames from {path}, starting {start}")
                frames.append(frame)
        finally:
            cap.release()
        return frames

    def __getitem__(self, ix):
        import cv2
        entry = self.records[ix]
        total = int(entry["frames"])
        # Same temporal window and geometry for RGB and alpha.
        start = random.randint(0, max(0, total - self.frames))
        fg_frames = self.read_frames(entry["foreground"], start, self.frames)
        a_frames = self.read_frames(entry["alpha"], start, self.frames)
        rgb, alpha = [], []
        for fg, a in zip(fg_frames, a_frames):
            if fg.shape[:2] != a.shape[:2]:
                raise RuntimeError(f"Mismatched frame dimensions: {entry['id']}")
            fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
            # Alpha video is expected to encode a gray matte. Inspect examples
            # before use; do NOT infer alpha from an arbitrary RGB foreground.
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            sz = (self.resolution, self.resolution)
            rgb.append(cv2.resize(fg, sz, interpolation=cv2.INTER_AREA))
            alpha.append(cv2.resize(a, sz, interpolation=cv2.INTER_AREA))
        result = np.concatenate(
            (np.stack(rgb, axis=0), np.stack(alpha, axis=0)[..., None]), axis=-1
        )
        return torch.from_numpy(result.copy()).permute(3, 0, 1, 2).float() / 255.0


def save_adapter(vae, dest, step, metrics):
    from safetensors.torch import save_file
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    block = vae.decoder.proj_out.out_features // 4
    # Only the newly added alpha slices need to be stored at Stage 1.
    tensors = {
        "encoder.conv_in.weight.alpha": vae.encoder.conv_in.weight[:, 3:].detach().cpu().contiguous(),
        "decoder.proj_out.weight.alpha": vae.decoder.proj_out.weight[3 * block:].detach().cpu().contiguous(),
    }
    if vae.decoder.proj_out.bias is not None:
        tensors["decoder.proj_out.bias.alpha"] = vae.decoder.proj_out.bias[3 * block:].detach().cpu().contiguous()
    save_file(tensors, str(dest / f"alpha_step_{step:06d}.safetensors"))
    (dest / "training_state.json").write_text(json.dumps(
        dict(step=step, metrics=metrics, precision="bf16", latent_channels=24,
             model="MiniMaxAI/MiniMax-H3", stage="vae-alpha-slices-only"), indent=2
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", default="outputs/vae_trial")
    p.add_argument("--model", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    doc = json.loads(pathlib.Path(args.manifest).read_text())
    clips = doc["clips"]
    if not clips:
        raise ValueError("Manifest has no valid clips")
    ds = PairedVideoDataset(clips, frames=17, resolution=args.resolution)
    summary = dict(clips=len(ds), frames=17, resolution=args.resolution, batch_size=args.batch_size,
                   max_steps=args.max_steps, precision="bf16", model=args.model)
    print("CONFIG", json.dumps(summary, indent=2))
    # Read a real data pair as part of CPU preflight.
    sample = ds[0]
    print("FIRST_SAMPLE", tuple(sample.shape), "alpha_range",
          (float(sample[3].min()), float(sample[3].max())))
    if args.dry_run:
        print("DRY_RUN_OK: no model weights downloaded; no GPU required")
        return

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required for training; use --dry-run without one")
    from diffusers import AutoencoderKLMiniMaxH3

    # Respect Diffusers' mixed-precision keep-in-fp32 rules for the H3 encoder.
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.bfloat16
    )
    vae = widen_h3_vae(vae)
    vae.enable_gradient_checkpointing()
    vae.to("cuda")
    trainable = configure_stage1(vae)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    loader = DataLoader(ds, shuffle=True, num_workers=0, batch_size=args.batch_size,
                        drop_last=False, pin_memory=True)
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    vae.train()
    step = 0
    while step < args.max_steps:
        for rgba in loader:
            rgba = rgba.to("cuda", dtype=torch.float32, non_blocking=True)
            input_tensor = rgba_to_vae_input(rgba)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                posterior = vae.encode(input_tensor).latent_dist
                z = posterior.mode()
                reconstruction = vae.decode(z).sample
            if reconstruction.shape != input_tensor.shape:
                raise RuntimeError(f"Temporal/shape mismatch: {reconstruction.shape} vs {input_tensor.shape}")
            result = vae_output_to_rgba(reconstruction.float())
            loss, losses = rgba_reconstruction_loss(result, rgba)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss)}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            step += 1
            metrics = {k: round(float(v.detach()), 6) for k, v in losses.items()}
            print(json.dumps(dict(step=step, total_loss=round(float(loss.detach()), 6),
                                  losses=metrics, max_vram_gb=round(torch.cuda.max_memory_allocated()/2**30, 2))),
                  flush=True)
            if step % args.save_every == 0 or step == args.max_steps:
                save_adapter(vae, args.output, step, metrics)
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
