"""CPU-only real MiniMax-H3 VAE configuration audit; downloads ~2 KB, NOT model weights."""
import argparse
import json
import math
from pathlib import Path


def validate(cfg):
    if cfg.get("_class_name") != "AutoencoderKLMiniMaxH3":
        raise ValueError(f"Unexpected VAE class: {cfg.get('_class_name')}")
    required = dict(in_channels=3, out_channels=3, latent_channels=24,
                    decoder_num_layers=36, clip_length=17, token_drop=3)
    for k, value in required.items():
        if cfg.get(k) != value:
            raise ValueError(f"Expected {k}={value}, got {cfg.get(k)}")
    spatial = math.prod(cfg["spatial_downsample_factors"])
    temporal = math.prod(cfg["temporal_downsample_factors"])
    if spatial != 16 or temporal != 4:
        raise ValueError(f"Unexpected compression: {spatial}x spatial, {temporal}x temporal")
    if len(cfg["latents_mean"]) != 24 or len(cfg["latents_std"]) != 24:
        raise ValueError("H3 requires 24 per-channel latent normalization entries")
    patch = spatial * spatial * temporal
    return dict(
        model_class=cfg["_class_name"], spatial_ratio=spatial,
        temporal_ratio=temporal, latent_channels=24,
        encoder_in_before=3, encoder_in_after=4,
        decoder_projection_before=3 * patch,
        decoder_projection_after=4 * patch,
        train_vae_clip_frames=17, train_dit_clip_frames=22,
        output_rows_added=patch,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local-config", help="Optional local vae/config.json, no network")
    p.add_argument("--output", default="h3_config_report.json")
    args = p.parse_args()
    if args.local_config:
        cfg_path = Path(args.local_config)
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = Path(hf_hub_download("MiniMaxAI/MiniMax-H3", "vae/config.json"))
    cfg = json.loads(cfg_path.read_text())
    report = validate(cfg)
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
