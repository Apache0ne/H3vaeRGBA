# Major model-validation suite

**Entrypoint:** [`benchmark_wan_dual.py`](../benchmark_wan_dual.py)

The trained 500-step VAE alpha-decoder LoRA achieved **alpha IoU 0.955329,
alpha MAE 0.017947** on the **three repeatedly validated clips**. These numbers
are real but **not independent generalization evidence**. This suite isolates
additional clips not previously used for checkpoint selection, and synthetic
out-of-distribution transparency clips.

It performs no training, does not update the existing checkpoint, and does not
download the 33B H3 diffusion transformer.

## Test taxonomy

1. **Previously validated** (`previously_validated`): the first three of the
   seeded validation split that were used to select checkpoints.
2. **Unused holdout** (`unused_holdout`): remaining items from the **same**
   seeded validation split, never used in checkpoint selection. If there are
   none, the benchmark reports zero; it does **not** falsely claim a new heldout.
3. **External paired videos** (`external_dataset`): optional second audited
   foreground/alpha manifest. Paths overlapping the training/validation
   manifest are rejected. Identical content saved under different names is
   **not** automatically detected.
4. **Synthetic stress** (`synthetic_stress`): deterministic 17-frame videos:
   moving disk, very thin ring, hair-like strands, soft glass, feathered edge,
   motion blur, checker alpha, entirely transparent, entirely opaque, and
   flat partial alpha.

Each sample uses the **same GT input alpha** in a 17-frame roundtrip through
both model variants:

- **Original pretrained H3 baseline:** alpha treated as a 3-channel grayscale
  RGB video, original frozen VAE encoder/decoder without adapter, averaged
  over RGB channels after ImageNet denormalization. **It is the exact
  step-zero equivalent** (no trained merger/head/LoRA).
- **Trained H3 dual alpha branch:** original RGB and alpha encodings,
  trainable checkpoint merger + decoder attention/MLP LoRA + alpha head.
- **RGB protection:** the original RGB decoder is run with alpha LoRA disabled.

The alpha input is ground truth. This benchmark **does not** demonstrate alpha
segmentation/extraction from an RGB-only video, nor text-to-video generation.

## Metrics

Measured separately per clip and group:

- Thresholded alpha intersection-over-union (IoU, **higher**)
- Dice, exact binary accuracy, soft/continuous IoU
- Straight alpha MAE/RMSE (**lower**) and MAE on pixels with alpha 0.05–0.95
- 1-pixel-tolerant spatial **boundary F1** for thin edges (**higher**)
- Temporal difference MAE for flicker or motion trails (**lower**)
- Composite RGB MAE over black, gray, and white backgrounds (**lower**)
- Original-RGB reconstruction visible-pixel MAE (**lower**)
- Mean clip inference seconds, GPU peak allocated memory
- Baseline-versus-trained **paired clip bootstrap 95% CI** for IoU and alpha MAE
- Macro means across clips **and** micro/pixel-pooled IoU

Interpretation: threshold IoU alone can look good while partially transparent
glass or hair mattes remain poor. Inspect *soft alpha*, *edge F1*, composites,
temporal error, and actual PNGs, not just the IoU headline.

## Colab GPU run on the existing 500-step model

```bash
git -C /content/H3vaeRGBA pull --ff-only
python /content/H3vaeRGBA/benchmark_wan_dual.py \
  --run-dir /content/h3rgba_runs/h3_wan_dual_lora_v1 \
  --manifest /content/h3rgba_data/manifest.json \
  --output-dir /content/h3rgba_runs/major_benchmark_128 \
  --resolution 128 --rank 4 --merger-scale 0.15 \
  --split-seed 42 --trained-val-samples 3 \
  --preview-limit 4 --bootstrap 2000
```

Add `--strict-unused` to **fail** if there are no never-validated clips.
If an independent dataset is available, first audit the paired MP4 videos
with `tools/dataset_audit.py`, then add
`--external-manifest /content/external_test/manifest.json`.

### Check the split and dataset first without spending GPU time

```bash
python /content/H3vaeRGBA/benchmark_wan_dual.py \
  --manifest /content/h3rgba_data/manifest.json \
  --output-dir /content/h3rgba_runs/major_benchmark_preflight \
  --preflight-only
```

This creates `preflight.json` listing exactly how many clips are previously
validated versus unused, so there is no invisible data-split ambiguity.

### Optional higher-resolution tests

Only after a successful 128px baseline, use another output directory with
`--resolution 192` or `--resolution 256`. H3 attention memory increases with
spatial tokens; **22 GiB fit at 128px does not guarantee 256px fit**. Different
resolutions should be compared as separate experiments.

### CPU code tests — no pretrained weights

```bash
pytest -q tests/test_major_benchmark_cpu.py
python benchmark_wan_dual.py --cpu-smoke --stress-only \
  --resolution 32 --output-dir /tmp/h3_major_cpu
```

`CPU_RANDOM_INTEGRATION_ONLY` scores test **benchmark correctness and I/O**,
not the quality of the trained H3 VAE. Never quote them as model performance.

## Outputs

```text
major_benchmark_128/
├── summary.json                # grouped stats, confidence intervals and provenance
├── samples.csv                 # per-clip metrics for original and trained models
├── samples.jsonl               # machine-readable detailed per-clip results
├── report.md                   # comparison table
└── previews/
    ├── previously_validated/
    ├── unused_holdout/
    ├── external_dataset/      # if supplied
    └── synthetic_stress/
        └── 000_thin_ring/
            ├── original/contact_sheet.png
            └── trained/contact_sheet.png
```

**Comparison target:** trained alpha IoU 0.955329/MAE 0.017947 previously
validated. We should only claim generalization if unused/external clips and
stress tests perform well too, especially with partial alpha and thin geometry.
