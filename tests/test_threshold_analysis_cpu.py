"""No-GPU benchmark V2 threshold analysis regression checks."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from analyze_benchmark_thresholds import analyze, calibrate, load_rows, write_report


def sample(name, group, *, fixed=.35, lower=.75, original=.2,
           foreground=100, total=1000, partial=.12):
    pred = {"alpha_iou":fixed,"alpha_mae":.02,
            "alpha_boundary_f1_1px":.8,
            "alpha_partial_mae":.15,
            "alpha_gt_partial_fraction":partial,
            "pixels_total":total}
    baseline = dict(pred,alpha_iou=original,alpha_mae=.05)
    a = {"0.20":lower-.1,"0.30":lower,"0.40":lower-.02,
         "0.50":fixed,"0.60":fixed-.1}
    b = {"0.20":original+.1,"0.30":original+.15,
         "0.40":original+.05,"0.50":original,"0.60":.05}
    return {"group":group,"id":name,"trained":pred,"baseline":baseline,
            "trained_thresholds":{
                "threshold_iou":a,
                "oracle_threshold":.3,"oracle_iou":lower,
                "oracle_gain_over_half":lower-fixed,
                "hard_gt_foreground_pixels":foreground,
            },
            "baseline_thresholds":{
                "threshold_iou":b,
                "oracle_threshold":.3,"oracle_iou":original+.15,
                "oracle_gain_over_half":.15,
                "hard_gt_foreground_pixels":foreground,
            }}


def test_calibration_uses_only_previous_validation_and_not_unseen_oracles():
    known = [
        sample("seen_a","previously_validated",lower=.8),
        sample("seen_b","previously_validated",lower=.7),
    ]
    unused = [sample("holdout","unused_holdout",fixed=.65,lower=.05)]
    # Calibration must still choose .3, even though it is bad on holdout.
    best = calibrate(known+unused)
    assert best["threshold"] == .3
    report = analyze(known+unused)
    assert report["calibration_trained"]["threshold"] == .3
    group = report["groups"]["unused_holdout"]
    assert group["fixed_0_5_trained_iou_macro"] == .65
    assert group["trained_iou_calibrated_from_previous_validation"] == .05
    # Never hide a loss with a selectively chosen test-clip oracle threshold.
    assert group["calibrated_delta_vs_fixed"] < 0


def test_nontrivial_mask_stats_exclude_transparent_full_and_flat(tmp_path):
    seen = [sample("seen","previously_validated")]
    stress = [
        sample("hair","synthetic_stress",fixed=.3,foreground=50),
        sample("empty","synthetic_stress",fixed=1.,foreground=0),
        sample("full","synthetic_stress",fixed=1.,foreground=1000),
        sample("flat","synthetic_stress",fixed=1.,foreground=0),
    ]
    result = analyze(seen+stress)
    group = result["groups"]["synthetic_stress"]
    assert group["n"] == 4
    assert group["nontrivial_binary_n"] == 1
    assert abs(group["nontrivial_binary_fixed_iou"]-.3)<1e-8
    text = write_report(result, tmp_path)
    assert "hair" in text
    assert "Oracle values" in text
    assert (tmp_path/"threshold_cases.csv").exists()
    assert (tmp_path/"threshold_analysis.json").exists()
    assert (tmp_path/"threshold_analysis.md").exists()
    assert len(json.loads((tmp_path/"threshold_analysis.json").read_text())
               ["ranked_cases_worst_first"])==5


def test_analyzer_cli_consumes_saved_v2_samples_without_gpu(tmp_path):
    output = tmp_path/"benchmark"
    output.mkdir()
    rows = [sample("known","previously_validated"),
            sample("new","unused_holdout",fixed=.52,lower=.53)]
    (output/"samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)+"\n")
    completed = subprocess.run([
        sys.executable,"analyze_benchmark_thresholds.py",
        "--results-dir",str(output),
    ],capture_output=True,text=True,timeout=30)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert "THRESHOLD_ANALYSIS_COMPLETE" in completed.stdout
    assert (output/"threshold_analysis.md").exists()
    assert "unused_holdout" in (output/"threshold_analysis.md").read_text()


def test_analyzer_does_not_silently_accept_old_benchmark(tmp_path):
    path=tmp_path/"samples.jsonl"
    path.write_text(json.dumps({"group":"unused_holdout","id":"x"})+"\n")
    with pytest.raises(ValueError,match="Old benchmark output"):
        load_rows(path)
