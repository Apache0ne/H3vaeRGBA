#!/usr/bin/env python3
"""Independent, repeatable full-model benchmark of trained H3 alpha-decoder LoRA.

Compares the trained adapter to the *actual frozen H3 grayscale-alpha VAE*
(step-zero equivalent) on every identical input. Reports previously seen
validation, never-validated holdout, and synthetic out-of-distribution stress
separately. This is a VAE round-trip benchmark; the alpha video is an INPUT.
This does NOT test prompt-to-transparent-video or the 33B H3 diffusion model.

Examples:
  python benchmark_wan_dual.py --run-dir /content/h3rgba_runs/h3_wan_dual_lora_v1
  python benchmark_wan_dual.py --cpu-smoke --stress-only --stress-cases thin_ring,moving_disk
"""
import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
import time
from pathlib import Path

import torch

from h3rgba.benchmark_metrics import (
    STRESS_CASES, measure, summarize, synthetic_rgba, threshold_diagnostics,
)
from h3rgba.decoder_lora import temporarily_disable_decoder_lora
from h3rgba.vae import image_net_stats


def config():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", default="/content/h3rgba_runs/h3_wan_dual_lora_v1",
                   help="Directory with trained dual alpha adapter checkpoints")
    p.add_argument("--checkpoint", default=None, help="Override best LoRA adapter path")
    p.add_argument("--manifest", default="/content/h3rgba_data/manifest.json")
    p.add_argument("--external-manifest", default=None,
                   help="Optional independent foreground+alpha pairs, never used in training")
    p.add_argument("--max-external", type=int, default=0,
                   help="0 = test all clips in external manifest")
    p.add_argument("--output-dir", default="/content/h3rgba_runs/h3_wan_major_benchmark")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--merger-scale", type=float, default=.15)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--trained-val-samples", type=int, default=3,
                   help="Number of first validation clips used during training")
    p.add_argument("--max-heldout", type=int, default=0,
                   help="0 = all clips in each validation partition")
    p.add_argument("--stress-cases", default=",".join(STRESS_CASES))
    p.add_argument("--stress-only", action="store_true",
                   help="Skip dataset manifest and test generated stress clips only")
    p.add_argument("--cpu-smoke", action="store_true",
                   help="Tiny random H3 CPU integration test; not a trained-model score")
    p.add_argument("--preflight-only", action="store_true",
                   help="Inspect held-out coverage / external overlap and exit without loading model")
    p.add_argument("--preview-limit", type=int, default=4,
                   help="Per-group preview limit; all stress cases render previews")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--strict-unused", action="store_true",
                   help="Fail if no truly unused held-out validation clips exist")
    p.add_argument("--min-unused-iou", type=float, default=None,
                   help="Optional minimum macro alpha IoU for unused holdout")
    p.add_argument("--max-unused-mae", type=float, default=None,
                   help="Optional maximum macro alpha MAE for unused holdout")
    return p.parse_args()


def sha256_file(path, *, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            data = file.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def partition(records, *, seed=42, previously_validated=3):
    """Reproduce EXACT training split, and isolate first-3 seen validation clips.

    The first min(3,Nvalidation) clips were used to select the best checkpoint
    during 500-step training. The rest were not selected/validated by the
    trainer (though still from the same dataset source/domain).
    """
    from run_vae_22gb import training_split
    train, val = training_split(records, seed)
    boundary = min(len(val), previously_validated)
    seen, unused = val[:boundary], val[boundary:]
    train_keys = {(e.get("id"), e.get("foreground"), e.get("alpha")) for e in train}
    for item in val:
        if (item.get("id"), item.get("foreground"), item.get("alpha")) in train_keys:
            raise RuntimeError(f"Train/validation record overlap: {item['id']}")
    all_ids = [e.get("id") for e in records]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("Duplicate manifest record IDs: cannot certify held-out split")
    file_keys = [(e.get("foreground"), e.get("alpha")) for e in records]
    if len(set(file_keys)) != len(file_keys):
        raise RuntimeError("Repeated foreground/alpha file pair in manifest")
    return train, seen, unused


@torch.no_grad()
def original_alpha(codec, z_alpha):
    """Step-zero grayscale alpha reconstruction from the ORIGINAL pretrained VAE.

    The actual trained alpha adapter is still present, but its LoRA layers
    are bypassed and the trainable merger/head are not called. Instead
    use the exact original head initialization: (R+G+B)/3 in pixel space.
    """
    with temporarily_disable_decoder_lora(codec.vae):
        if z_alpha.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = codec.vae._decode_clip(z_alpha)
        else:
            output = codec.vae._decode_clip(z_alpha)
    output = output[:, :3, codec.vae.frame_pre_padding:].float()
    mean, std = image_net_stats(output)
    return ((output * std + mean).mean(1, keepdim=True)).clamp(0, 1)


def one_video(codec, rgba, *, include_rgb=True):
    """Only one pair of frozen H3 encoder calls, then distinct decoder passes."""
    if rgba.ndim != 5 or rgba.shape[1] != 4 or rgba.shape[2] != 17:
        raise ValueError("Must supply RGBA [B,4,17,H,W]")
    with torch.inference_mode():
        zr, za = codec.encode_pair(rgba)
        baseline = original_alpha(codec, za)
        trained = codec.decode_alpha(zr, za).clamp(0, 1)
        rgb = codec.decode_rgb(zr) if include_rgb else None
        baseline_metrics = measure(baseline, rgba, pred_rgb=rgb)
        trained_metrics = measure(trained, rgba, pred_rgb=rgb)
        baseline_thresholds = threshold_diagnostics(baseline, rgba)
        trained_thresholds = threshold_diagnostics(trained, rgba)
    return {
        "baseline": baseline_metrics, "trained": trained_metrics,
        "baseline_thresholds": baseline_thresholds,
        "trained_thresholds": trained_thresholds,
        "delta_alpha_iou": trained_metrics["alpha_iou"] - baseline_metrics["alpha_iou"],
        "delta_alpha_mae": trained_metrics["alpha_mae"] - baseline_metrics["alpha_mae"],
        "delta_boundary_f1": trained_metrics["alpha_boundary_f1_1px"] -
                             baseline_metrics["alpha_boundary_f1_1px"],
        "pred": trained, "baseline_pred": baseline,
        "rgb": rgb,
    }


def paired_bootstrap(rows, key="alpha_iou", rounds=2000, seed=2026):
    """Bootstrap paired *clip-level* average improvement, 95% percentile CI."""
    if not rows:
        return None
    values = [r["trained"][key] - r["baseline"][key] for r in rows]
    average = sum(values) / len(values)
    if len(values) < 2 or rounds < 2:
        return {"mean_delta": average, "ci95": None, "n": len(values)}
    rng = random.Random(seed)
    trials = []
    for _ in range(rounds):
        trials.append(sum(rng.choice(values) for _ in values) / len(values))
    trials.sort()
    return {
        "mean_delta": average,
        "ci95": [trials[int(.025 * (rounds-1))], trials[int(.975 * (rounds-1))]],
        "n": len(values),
        "method": "paired clip bootstrap (not independent pixel bootstrap)",
    }


def group_summary(rows, *, bootstrap=2000):
    if not rows:
        return {"samples": 0, "baseline": None, "trained": None}
    return {
        "samples": len(rows),
        "baseline": summarize([{"id":r["id"], **r["baseline"]} for r in rows]),
        "trained": summarize([{"id":r["id"], **r["trained"]} for r in rows]),
        "threshold_calibration": {
            "mean_oracle_gain_trained": (
                sum(r["trained_thresholds"]["oracle_gain_over_half"]
                    for r in rows)/len(rows)
            ),
            "mean_oracle_gain_original": (
                sum(r["baseline_thresholds"]["oracle_gain_over_half"]
                    for r in rows)/len(rows)
            ),
            "note": "Oracle gain chooses threshold using this test clip. Diagnostic, not held-out performance.",
        },
        "delta_alpha_iou": paired_bootstrap(rows, "alpha_iou", bootstrap),
        "delta_alpha_mae": paired_bootstrap(rows, "alpha_mae", bootstrap),
        "improved_iou_samples": sum(r["delta_alpha_iou"] > 0 for r in rows),
        "worse_iou_samples": sum(r["delta_alpha_iou"] < 0 for r in rows),
    }


def safe_slug(name):
    s = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(name))
    return s[:85] or "clip"


def save_previews(rgba, rgb, pred, baseline, dest):
    """Two contact sheets: target vs trained and target vs original RGB-VAE alpha."""
    from run_vae_22gb import save_frames
    dest = Path(dest)
    target = rgba[0].detach().cpu()
    rgb = rgb.detach().cpu() if rgb is not None else target[:3][None]
    trained = torch.cat((rgb[0].clamp(0, 1), pred[0].detach().cpu()), 0)
    orig = torch.cat((rgb[0].clamp(0, 1), baseline[0].detach().cpu()), 0)
    a = save_frames(target, trained, dest / "trained", max_frames=3)
    b = save_frames(target, orig, dest / "original", max_frames=3)
    return {"trained_contact": str(a), "original_contact": str(b)}


def tiny_model():
    """Only CPU smoke integration; must NEVER be presented as released H3 quality."""
    from diffusers import AutoencoderKLMiniMaxH3
    from h3rgba.wan_dual import DualAlphaCodec
    vae = AutoencoderKLMiniMaxH3(
        latent_channels=24, block_out_channels=(8,8,8,8,8,8),
        layers_per_block=1, norm_num_groups=4, decoder_num_layers=1,
        decoder_num_attention_heads=2, decoder_attention_head_dim=16,
        decoder_ffn_mult=2, clip_length=17, token_drop=3,
    )
    vae.disable_tiling()
    with torch.no_grad():
        for block in vae.decoder.transformer_blocks:
            block.scale1.fill_(.25)
            block.scale2.fill_(.25)
    return DualAlphaCodec(vae, rank=2).eval()


def model_from_checkpoint(args, device):
    from train_wan_dual_lora import make_model
    config = argparse.Namespace(rank=args.rank, merger_scale=args.merger_scale,
                                resolution=args.resolution)
    codec, proof = make_model(config, device)
    ckpt = Path(args.checkpoint) if args.checkpoint else (
        Path(args.run_dir) / "checkpoints" / "dual_lora_best.safetensors"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(f"Trained adapter missing: {ckpt}")
    codec.load_adapter(ckpt)
    codec.eval()
    return codec, {"path":str(ckpt.resolve()), "sha256":sha256_file(ckpt),
                   "pretrained_verification":proof}


def benchmark(codec, items, output, args, device):
    from train_vae import PairedVideoDataset
    groups = {}
    report_lines = []
    for group, entries in items.items():
        rows = []
        if group in ("previously_validated", "unused_holdout", "external_dataset"):
            dataset = PairedVideoDataset(entries, resolution=args.resolution, deterministic=True)
            data = ((str(entry.get("id", idx)), dataset[idx]) for idx, entry in enumerate(entries))
        else:
            data = ((name, synthetic_rgba(name, args.resolution)) for name in entries)
        for index, (case_id, tensor) in enumerate(data):
            start = time.perf_counter()
            rgba = tensor.unsqueeze(0).to(device, dtype=torch.float32)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = one_video(codec, rgba)
            if device.type == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - t0
            row = {"group": group, "id": case_id,
                   "seconds": duration,
                   **{key: value for key, value in result.items()
                      if key not in ("pred", "baseline_pred", "rgb")}}
            rows.append(row)
            print("SAMPLE", json.dumps({
                "group": group, "id": case_id,
                "trained_iou": row["trained"]["alpha_iou"],
                "original_iou": row["baseline"]["alpha_iou"],
                "trained_mae": row["trained"]["alpha_mae"],
                "original_mae": row["baseline"]["alpha_mae"],
                "sec": round(duration, 3),
            }), flush=True)
            if group == "synthetic_stress" or index < args.preview_limit:
                target = output / "previews" / group / f"{index:03d}_{safe_slug(case_id)}"
                save_previews(rgba, result["rgb"], result["pred"],
                              result["baseline_pred"], target)
            del rgba, result
        groups[group] = rows
    return groups


def main():
    args = config()
    if args.resolution < 32 or args.resolution % 32 or args.trained_val_samples < 0:
        raise ValueError("Invalid resolution or trained validation split size")
    if args.max_heldout < 0 or args.max_external < 0 or args.bootstrap < 0:
        raise ValueError("Invalid benchmark size/bootstrap")
    cases = tuple(filter(None, (v.strip() for v in args.stress_cases.split(","))))
    if not cases or set(cases) - set(STRESS_CASES):
        raise ValueError(f"Unknown stress cases: {sorted(set(cases)-set(STRESS_CASES))}")
    if args.cpu_smoke and not args.stress_only:
        raise ValueError("CPU smoke must be --stress-only; no pretrained dataset inference")
    random.seed(args.split_seed)
    torch.manual_seed(args.split_seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(exist_ok=True, parents=True)
    items = {}
    manifest_proof = None
    if not args.stress_only:
        manifest = Path(args.manifest).resolve()
        data = json.loads(manifest.read_text())
        records = data["clips"]
        train, seen, unused = partition(records, seed=args.split_seed,
                                        previously_validated=args.trained_val_samples)
        if args.strict_unused and not unused:
            raise RuntimeError(
                f"No genuinely unused held-out clips: {len(train)} trained, "
                f"{len(seen)} already validated; provide external held-out clips"
            )
        if args.max_heldout:
            seen = seen[:args.max_heldout]
            unused = unused[:args.max_heldout]
        items["previously_validated"] = seen
        items["unused_holdout"] = unused
        manifest_proof = {
            "file":str(manifest), "sha256":sha256_file(manifest),
            "records":len(records), "training_clip_count":len(train),
            "previously_validated_clips":len(seen),
            "unused_holdout_clips":len(unused),
            "note":"Unused clips never used in trainer validation but same dataset domain; "
                   "no content-level near-duplicate audit. If no unused clips, this is NOT "
                   "independent dataset generalization evidence.",
        }
        print("SPLIT_AUDIT",json.dumps(manifest_proof),flush=True)
    if args.external_manifest:
        external_path = Path(args.external_manifest).resolve()
        external = json.loads(external_path.read_text())["clips"]
        if args.stress_only:
            raise ValueError("--external-manifest cannot be combined with --stress-only")
        training_locations = {str(Path(e["foreground"]).resolve()) for e in records} | {
            str(Path(e["alpha"]).resolve()) for e in records
        }
        collision = [e["id"] for e in external if
                     str(Path(e["foreground"]).resolve()) in training_locations or
                     str(Path(e["alpha"]).resolve()) in training_locations]
        if collision:
            raise RuntimeError("External test videos overlap training/validation file paths: "
                               + str(collision[:6]))
        if len({(e["foreground"], e["alpha"]) for e in external}) != len(external):
            raise RuntimeError("Duplicate files in external manifest")
        items["external_dataset"] = external[:args.max_external] if args.max_external else external
        manifest_proof["external"] = {"path":str(external_path),
                                      "sha256":sha256_file(external_path),
                                      "records":len(external),
                                      "tested":len(items["external_dataset"]),
                                      "file_path_disjoint":True,
                                      "near_duplicate_content_audited":False}
    items["synthetic_stress"] = cases[:2] if args.cpu_smoke else cases
    if args.preflight_only:
        report = {
            "mode": "MANIFEST_PREFLIGHT_NO_MODEL",
            "manifest": manifest_proof,
            "planned_groups": {key: len(value) for key,value in items.items()},
            "warning": "No VAE inference and no trained checkpoint was tested.",
        }
        (output / "preflight.json").write_text(json.dumps(report,indent=2))
        print("BENCHMARK_PREFLIGHT_OK", json.dumps(report), flush=True)
        return
    device = torch.device("cpu" if args.cpu_smoke else "cuda")
    if not args.cpu_smoke:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Real pretrained H3 benchmark requires BF16-capable GPU")
        codec, checkpoint_proof = model_from_checkpoint(args,device)
        torch.cuda.reset_peak_memory_stats()
    else:
        codec = tiny_model()
        checkpoint_proof = {"random_tiny_cpu_smoke":True, "trained_adapter":False}
    codec.eval()
    start = time.perf_counter()
    groups = benchmark(codec, items, output, args, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime = time.perf_counter()-start
    summaries = {name:group_summary(rows,bootstrap=args.bootstrap)
                 for name,rows in groups.items()}
    report = {
        "benchmark_version":1,
        "mode":"CPU_RANDOM_INTEGRATION_ONLY" if args.cpu_smoke else
               "REAL_PRETRAINED_H3_WITH_TRAINED_ALPHA_DECODER_LORA",
        "checkpoint":checkpoint_proof, "manifest":manifest_proof,
        "resolution":args.resolution, "frames":17,
        "seed":args.split_seed, "runtime_seconds":runtime,
        "gpu_peak_alloc_gib": round(torch.cuda.max_memory_allocated()/2**30,4)
                              if device.type=="cuda" else None,
        "groups":summaries,
        "qualification":"Input RGBA supplies ground-truth alpha. This is a VAE "
                        "roundtrip test, NOT generation or alpha-from-RGB extraction.",
    }
    (output / "summary.json").write_text(json.dumps(report,indent=2))
    with (output / "samples.jsonl").open("w") as file:
        for rows in groups.values():
            for entry in rows:
                file.write(json.dumps(entry)+"\n")
    with (output / "samples.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["group","id","model","alpha_iou","alpha_mae",
                         "boundary_f1_1px","soft_iou","partial_mae","temporal_delta_mae",
                         "composite_black","composite_white","rgb_visible_mae",
                         "oracle_threshold_testclip","oracle_iou_testclip",
                         "gt_hard_foreground_pixels","seconds"])
        for group,rows in groups.items():
            for row in rows:
                for label in ("baseline","trained"):
                    m=row[label]
                    writer.writerow([group,row["id"],label,m["alpha_iou"],m["alpha_mae"],
                                     m["alpha_boundary_f1_1px"],m["alpha_soft_iou"],
                                     m["alpha_partial_mae"],m["alpha_temporal_delta_mae"],
                                     m["composite_mae_black"],m["composite_mae_white"],
                                     m.get("rgb_visible_mae"),
                                     row[label+"_thresholds"]["oracle_threshold"],
                                     row[label+"_thresholds"]["oracle_iou"],
                                     row[label+"_thresholds"]["hard_gt_foreground_pixels"],
                                     row["seconds"]])
    markdown=["# Major H3 VAE alpha-decoder benchmark",
              "",
              f"Mode: **{report['mode']}**. Resolution: {args.resolution}.",
              "",
              "**The alpha mask is provided to the encoder**; this is not an "
              "alpha-extraction or text-to-video generation benchmark.",
              "",
              "| Group | N | Original alpha IoU | Trained alpha IoU | "
              "Original MAE | Trained MAE | Trained boundary F1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name,values in summaries.items():
        if not values["samples"]:
            markdown.append(f"| {name} | 0 | — | — | — | — | — |")
        else:
            a,b=values["baseline"],values["trained"]
            markdown.append(f"| {name} | {values['samples']} | "
                            f"{a['alpha_iou']:.4f} | {b['alpha_iou']:.4f} | "
                            f"{a['alpha_mae']:.4f} | {b['alpha_mae']:.4f} | "
                            f"{b['alpha_boundary_f1_1px']:.4f} |")
    markdown += ["", "Individual results: `samples.csv` and `samples.jsonl`.",
                 "Previews: `previews/`, showing original vs trained.",
                 "For unseen clips, also inspect the paired bootstrap interval in `summary.json`.",
                 "This experiment does not detect content-level near duplicates across files."]
    (output/"report.md").write_text("\n".join(markdown)+"\n")
    print("BENCHMARK_COMPLETE",json.dumps({
        "summary":str(output/"summary.json"),"report":str(output/"report.md"),
        "mode":report["mode"],"seconds":round(runtime,2),
        "samples":{k:len(v) for k,v in groups.items()},
    }),flush=True)
    print("\n".join(markdown),flush=True)
    u=summaries.get("unused_holdout",{})
    if not args.cpu_smoke:
        if args.min_unused_iou is not None:
            if not u.get("samples") or u["trained"]["alpha_iou"] < args.min_unused_iou:
                raise SystemExit("FAIL: unused holdout IoU below requested threshold")
        if args.max_unused_mae is not None:
            if not u.get("samples") or u["trained"]["alpha_mae"] > args.max_unused_mae:
                raise SystemExit("FAIL: unused holdout MAE above requested threshold")


if __name__ == "__main__":
    main()
