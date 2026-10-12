#!/usr/bin/env python3
"""Post-process existing H3 major benchmark results WITHOUT loading VAE/GPU.

This is intentionally a *separate* analysis stage. The .5-threshold alpha
scores are the benchmark. Per-test-clip oracle thresholds are diagnostics only.
For a less optimistic calibration experiment, choose ONE threshold using only
the three checkpoint-selection validation clips, then apply it unchanged to
the nine unused held-out clips. The 3-clip calibration set is small and has
already been used for model selection; do NOT treat it as proof of independent
cross-domain calibration.

Usage:
  python analyze_benchmark_thresholds.py \
    --results-dir /content/h3rgba_runs/h3_major_benchmark_v2
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def load_rows(path):
    data = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not data:
        raise ValueError("No benchmark samples available")
    needed = ("group", "id", "baseline", "trained", "trained_thresholds",
              "baseline_thresholds")
    for row in data:
        absent = [name for name in needed if name not in row]
        if absent:
            raise ValueError(f"Old benchmark output missing {absent}; rerun V2 benchmark")
        if "threshold_iou" not in row["trained_thresholds"]:
            raise ValueError("Missing trained threshold curve")
    return data


def calibrate(rows, model="trained", source_group="previously_validated"):
    """Choose threshold on previously validated clips only.

    The reported evaluation performance is then computed on UNUSED clips;
    NEVER select threshold using their oracle IoU.
    """
    source = [r for r in rows if r["group"] == source_group]
    if not source:
        return None
    key = model + "_thresholds"
    candidates = sorted(set.intersection(*[
        set(row[key]["threshold_iou"]) for row in source
    ]), key=float)
    if not candidates:
        raise ValueError("No common threshold grid across calibrating clips")
    means = {
        th: statistics.mean(row[key]["threshold_iou"][th] for row in source)
        for th in candidates
    }
    best = sorted(candidates, key=lambda th: (-means[th], abs(float(th)-.5),
                                              float(th)))[0]
    return {"threshold": float(best), "threshold_key": best,
            "calibration_n": len(source), "calibration_macro_iou": means[best],
            "candidate_macro_iou": means}


def analyze(rows):
    fitted = calibrate(rows, "trained")
    original_fit = calibrate(rows, "baseline")
    groups = {}
    named_rows = []
    for group in sorted({r["group"] for r in rows}):
        members = [r for r in rows if r["group"] == group]
        meaningful = [r for r in members
                      if 0 < r["trained_thresholds"]["hard_gt_foreground_pixels"]
                         < r["trained"]["pixels_total"]]
        group_scores = {
            "n": len(members),
            "nontrivial_binary_n": len(meaningful),
            "fixed_0_5_trained_iou_macro":
                statistics.mean(r["trained"]["alpha_iou"] for r in members),
            "fixed_0_5_baseline_iou_macro":
                statistics.mean(r["baseline"]["alpha_iou"] for r in members),
            "mean_test_clip_oracle_gain_trained":
                statistics.mean(r["trained_thresholds"]["oracle_gain_over_half"]
                                for r in members),
            "mean_alpha_mae": statistics.mean(r["trained"]["alpha_mae"]
                                               for r in members),
            "mean_boundary_f1": statistics.mean(
                r["trained"]["alpha_boundary_f1_1px"] for r in members),
            "mean_partial_alpha_mae": (
                statistics.mean(r["trained"]["alpha_partial_mae"]
                                for r in members
                                if r["trained"]["alpha_partial_mae"] is not None)
                if any(r["trained"]["alpha_partial_mae"] is not None
                       for r in members) else None
            ),
        }
        if meaningful:
            group_scores["nontrivial_binary_fixed_iou"] = statistics.mean(
                r["trained"]["alpha_iou"] for r in meaningful)
        else:
            group_scores["nontrivial_binary_fixed_iou"] = None
        if fitted:
            key = fitted["threshold_key"]
            group_scores["trained_iou_calibrated_from_previous_validation"] = (
                statistics.mean(r["trained_thresholds"]["threshold_iou"][key]
                                for r in members)
            )
            group_scores["calibrated_delta_vs_fixed"] = (
                group_scores["trained_iou_calibrated_from_previous_validation"] -
                group_scores["fixed_0_5_trained_iou_macro"]
            )
        if original_fit:
            key = original_fit["threshold_key"]
            group_scores["original_iou_calibrated_from_previous_validation"] = (
                statistics.mean(r["baseline_thresholds"]["threshold_iou"][key]
                                for r in members)
            )
        groups[group] = group_scores
    for row in sorted(rows, key=lambda r:r["trained"]["alpha_iou"]):
        mask_pixels = row["trained_thresholds"]["hard_gt_foreground_pixels"]
        px = row["trained"]["pixels_total"]
        info = {
            "group":row["group"], "id":row["id"],
            "gt_binary_fg_fraction":mask_pixels/px,
            "gt_partial_alpha_fraction":row["trained"]["alpha_gt_partial_fraction"],
            "fixed_alpha_iou":row["trained"]["alpha_iou"],
            "oracle_iou_on_same_clip":row["trained_thresholds"]["oracle_iou"],
            "oracle_threshold_on_same_clip":row["trained_thresholds"]["oracle_threshold"],
            "oracle_gain_diagnostic_only":
                row["trained_thresholds"]["oracle_gain_over_half"],
            "alpha_mae":row["trained"]["alpha_mae"],
            "alpha_partial_mae":row["trained"]["alpha_partial_mae"],
            "boundary_f1":row["trained"]["alpha_boundary_f1_1px"],
        }
        if fitted:
            info["calibrated_threshold"] = fitted["threshold"]
            info["calibrated_alpha_iou"] = (
                row["trained_thresholds"]["threshold_iou"][fitted["threshold_key"]]
            )
        named_rows.append(info)
    return {
        "method":"Threshold chosen only on previously validated clips. Oracle "
                 "per-test-clip IoU is NOT an unbiased performance estimate.",
        "calibration_trained":fitted,
        "calibration_original":original_fit,
        "groups":groups, "ranked_cases_worst_first":named_rows,
    }


def write_report(report, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    (output/"threshold_analysis.json").write_text(json.dumps(report,indent=2))
    with (output/"threshold_cases.csv").open("w", newline="") as file:
        rows = report["ranked_cases_worst_first"]
        fields = list(rows[0])
        writer = csv.DictWriter(file,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    train = report["calibration_trained"]
    lines = ["# H3 alpha mask threshold / difficult-case analysis",
             "",
             "**The fixed 0.50 threshold remains the headline benchmark.** "
             "Per-test-clip oracle IoU is an optimistic diagnostic, not "
             "held-out performance.",
             ""]
    if train:
        lines += [f"Global threshold chosen on only "
                  f"{train['calibration_n']} previously validated clips: "
                  f"**{train['threshold']:.2f}**.", "",
                  "This is exploratory calibration because the same few clips "
                  "were used for LoRA checkpoint selection.", ""]
    lines += ["| Group | Samples | Fixed 0.50 IoU | Calibrated IoU | "
              "Nontrivial-mask IoU | Mean oracle gain (diagnostic) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name,grp in report["groups"].items():
        cal = grp.get("trained_iou_calibrated_from_previous_validation")
        meaningful = grp["nontrivial_binary_fixed_iou"]
        lines.append(
            f"| {name} | {grp['n']} | {grp['fixed_0_5_trained_iou_macro']:.4f} | "
            f"{cal:.4f} | " if cal is not None else
            f"| {name} | {grp['n']} | {grp['fixed_0_5_trained_iou_macro']:.4f} | — | "
        )
        suffix = (f"{meaningful:.4f}" if meaningful is not None else "—") + (
            f" | {grp['mean_test_clip_oracle_gain_trained']:+.4f} |"
        )
        lines[-1] += suffix
    lines += ["", "## Worst cases", "",
              "| Clip | Group | Alpha IoU @0.5 | GT hard-foreground % | "
              "Partial alpha % | Oracle threshold* | Oracle IoU* |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for case in report["ranked_cases_worst_first"][:12]:
        lines.append(f"| {case['id']} | {case['group']} | "
                     f"{case['fixed_alpha_iou']:.4f} | "
                     f"{100*case['gt_binary_fg_fraction']:.4f}% | "
                     f"{100*case['gt_partial_alpha_fraction']:.2f}% | "
                     f"{case['oracle_threshold_on_same_clip']:.2f} | "
                     f"{case['oracle_iou_on_same_clip']:.4f} |")
    lines += ["", "*Oracle values are fitted to each evaluated clip and "
              "**must not** be substituted for the fixed-threshold benchmark.",
              "", "Files: `threshold_cases.csv`, `threshold_analysis.json`."]
    (output/"threshold_analysis.md").write_text("\n".join(lines)+"\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",required=True)
    parser.add_argument("--output-dir",default=None)
    args=parser.parse_args()
    base=Path(args.results_dir)
    out=Path(args.output_dir) if args.output_dir else base
    report=analyze(load_rows(base/"samples.jsonl"))
    text=write_report(report,out)
    print(text,flush=True)
    print("THRESHOLD_ANALYSIS_COMPLETE",out,flush=True)


if __name__ == "__main__":
    main()
