# Perturbation Robustness Evaluation PRD

## Problem Statement

The thesis proposal abstract explicitly commits to evaluating the cattle vision framework under "controlled environmental perturbations" as part of the generalization and robustness analysis layer. The existing `results/generalization/perturbation_delta.csv` file is 0 bytes — no perturbation experiments have been run. Without this data, thesis §5.4.2 ("Controlled Environmental Perturbations") cannot be written with numbers, and the abstract overstates what was evaluated. The four OOD datasets (OpenCows2020, Cows2021, CattleEyeView, Freeman Center) already have clean detection baselines in `results/detection/`, making it straightforward to measure mAP degradation under synthetic perturbations without any retraining.

## Solution

Write a single self-contained perturbation evaluation script that applies five classes of synthetic image transforms (brightness reduction, Gaussian noise, motion blur, synthetic fog, synthetic rain) at two severity levels to each of the four OOD dataset test splits, runs RF-DETR inference on the perturbed images in-memory, and writes the mAP50 delta relative to the existing clean baseline into `results/generalization/perturbation_delta.csv`. The script wraps the existing OOD eval logic without subprocess calls — transforms are applied before the image is passed to the model. Thesis §5.4.2 is then written from the populated CSV, and issue #24 (§6.4.2 pipeline error propagation) is updated to reference perturbation sensitivity as a third generalization axis.

## User Stories

1. As a thesis author, I want `results/generalization/perturbation_delta.csv` populated with mAP50 degradation measurements for all five perturbation types, so that §5.4.2 can report quantitative robustness results rather than a qualitative placeholder.
2. As a thesis author, I want clean baseline mAP50 values co-located in `perturbation_delta.csv` alongside perturbed values, so that the delta is self-contained and the table can be reproduced without cross-referencing four separate JSON files.
3. As a thesis author, I want both a low-severity and high-severity condition for each perturbation type, so that §5.4.2 can discuss sensitivity gradients rather than just presence/absence of degradation.
4. As a thesis author, I want brightness reduction included as a perturbation type, so that low-light and dusk operating conditions are represented.
5. As a thesis author, I want Gaussian noise included as a perturbation type, so that sensor noise and low-SNR camera conditions are represented.
6. As a thesis author, I want motion blur included as a perturbation type, so that camera vibration and panning artifacts are represented.
7. As a thesis author, I want synthetic fog included as a perturbation type, so that weather-induced visibility reduction is represented.
8. As a thesis author, I want synthetic rain streak included as a perturbation type, so that rain — the most practically relevant outdoor weather perturbation for cattle monitoring — is represented.
9. As a thesis author, I want all four OOD datasets (OpenCows2020, Cows2021, CattleEyeView, Freeman Center) evaluated under every perturbation condition, so that §5.4.2 can compare robustness across datasets with different domain shift magnitudes.
10. As a thesis author, I want the per-dataset mAP50 delta for each perturbation to be interpretable in relation to the clean OOD baseline, so that high-shift datasets (e.g. OpenCows2020 at 33.3%) are not conflated with low-shift datasets (e.g. Freeman at 73.0%) when assessing absolute vs. relative degradation.
11. As a thesis author, I want the perturbation script to read existing clean baseline values from the already-committed OOD eval JSONs rather than re-running clean inference, so that no GPU time is wasted re-computing known results.
12. As a thesis author, I want the perturbation script to produce one intermediate JSON per (dataset, perturbation_type, severity) condition, so that individual runs can be re-run or inspected without regenerating the full CSV.
13. As a thesis author, I want a separate aggregation step that assembles the intermediate JSONs into `perturbation_delta.csv`, so that the CSV can be regenerated if a single condition is re-run without invalidating the others.
14. As a thesis author, I want the script to print a summary table of mAP50 deltas after all conditions complete, so that the results are immediately readable without opening the CSV.
15. As a future reader of the codebase, I want the perturbation types and severity parameters documented in the script header, so that the evaluation can be reproduced or extended without reverse-engineering magic numbers.
16. As a thesis author, I want issue #24 (§6.4.2 pipeline error propagation) updated to include perturbation sensitivity as a third generalization axis alongside cross-dataset OOD results, so that the discussion section synthesizes all generalization evidence.

## Implementation Decisions

### Module: Perturbation Transform Layer

- A pure-function transform module accepts a PIL Image and returns a perturbed PIL Image. It has no side effects and no dependency on the model or COCO annotations.
- Five transform functions are implemented using `albumentations` (which covers all five types cleanly):
  - **Brightness reduction**: `albumentations.RandomBrightnessContrast` with `brightness_limit` clamped to the target factor. Low: factor=0.5 (half brightness). High: factor=0.25 (quarter brightness).
  - **Gaussian noise**: `albumentations.GaussNoise` with `var_limit`. Low: σ=25. High: σ=50.
  - **Motion blur**: `albumentations.MotionBlur` with `blur_limit`. Low: kernel=7. High: kernel=15.
  - **Fog**: `albumentations.RandomFog` with `fog_coef_lower` and `fog_coef_upper`. Low: coef=0.3. High: coef=0.6.
  - **Rain**: `albumentations.RandomRain` with `slant_range`, `drop_length`, `drop_width`. Low: slant=10, drop_length=20, drop_width=1. High: slant=20, drop_length=40, drop_width=2.
- Each severity level is a named constant (`"low"`, `"high"`) to avoid numeric ambiguity in the CSV.
- The transform layer converts PIL → numpy for albumentations → PIL for the model, keeping the model-facing interface identical to the clean eval path.

### Module: Perturbation Eval Runner

- Wraps the existing `collect_predictions` + `run_coco_eval` logic from `eval_detection_ood.py` by injecting the transform function into the image loading step.
- Accepts `--dataset_dir`, `--dataset_name`, `--perturbation`, `--severity` args; `--checkpoint` and `--threshold` default to the canonical values (checkpoint_best_total.pth, threshold=0.3) to match all prior OOD evaluations.
- Loads clean baseline mAP50, mAP, and AR100 from the corresponding OOD eval JSON in `results/detection/` rather than re-running inference.
- Writes one intermediate result JSON to `results/generalization/perturbation_runs/{dataset}_{perturbation}_{severity}.json` with fields: `dataset`, `perturbation_type`, `severity`, `n_images`, `mAP50_clean`, `mAP50_perturbed`, `delta_mAP50`, `mAP_clean`, `mAP_perturbed`, `AR100_clean`, `AR100_perturbed`.

### Module: CSV Aggregator

- A short aggregation script reads all intermediate JSONs under `results/generalization/perturbation_runs/` and writes `results/generalization/perturbation_delta.csv`.
- Output CSV schema:

```
dataset, perturbation_type, severity, n_images,
mAP50_clean, mAP50_perturbed, delta_mAP50,
mAP_clean, mAP_perturbed,
AR100_clean, AR100_perturbed
```

- Rows are sorted by `dataset`, then `perturbation_type`, then `severity`.
- The aggregator is idempotent: re-running it after a single condition is re-run picks up the updated intermediate JSON without touching the others.

### Execution Plan

- 5 perturbation types × 2 severity levels × 4 OOD datasets = **40 eval runs**.
- Each run loads the model once and iterates through the dataset. Model reload is the dominant setup cost.
- To amortize model loading, the runner accepts a `--all` flag that loops over all (perturbation, severity) combinations for a given dataset before releasing the model, requiring only 4 model loads total (one per dataset).
- All runs use the `cattletransformer` conda environment with CUDA. Transforms are CPU-side; inference is GPU-side. No new packages beyond `albumentations` are required (it is likely already installed; if not, `pip install albumentations`).
- Intermediate JSONs under `results/generalization/perturbation_runs/` are gitignored since they are large in aggregate; only `perturbation_delta.csv` is committed.

### Thesis §5.4.2 Revision

- Section title changes from "Controlled Environmental Perturbations" (placeholder) to content covering: perturbation taxonomy (5 types, 2 severities), dataset coverage (all 4 OOD datasets), methodology (in-memory transforms, no retraining, same threshold=0.3), and key findings from the populated CSV.
- Issue #24 (§6.4.2 pipeline error propagation) is updated to add perturbation sensitivity as a third generalization axis: detection layer → tracking layer → behavior layer → perturbation sensitivity across datasets.

## Testing Decisions

These are data pipeline scripts producing CSV/JSON outputs, so testing focuses on output correctness and known-value round-trips rather than unit behavioral testing.

- **What makes a good test here:** verify non-empty output, correct schema (column names, row count), and that known clean baseline values from existing OOD eval JSONs appear verbatim in the `mAP50_clean` column of `perturbation_delta.csv`. A perturbed mAP50 should always be ≤ the clean baseline for high-severity conditions (a sanity check that transforms are actually degrading the images).
- **Transform layer:** apply each transform to a single test image and assert the output shape is unchanged, the dtype is uint8, and the pixel values differ from the input (i.e. the transform is not a no-op). This can be run without the model or any COCO annotations.
- **Aggregator:** provide three synthetic intermediate JSONs with known values and assert the output CSV has exactly three rows with correct `delta_mAP50` arithmetic.
- **Known-value spot checks on the final CSV:**
  - Freeman Center clean baseline: `mAP50_clean` = 0.7298 for all Freeman rows.
  - CattleEyeView clean baseline: `mAP50_clean` = 0.470 for all CattleEyeView rows.
  - OpenCows2020 clean baseline: `mAP50_clean` = 0.3326 for all OpenCows2020 rows.
  - Cows2021 clean baseline: `mAP50_clean` = 0.2729 for all Cows2021 rows.
  - Row count: exactly 40 rows (5 types × 2 severities × 4 datasets).
- No pytest files are required. Manual verification against known values is sufficient given the thesis context.

## Out of Scope

- **CBVD-5 and CVB perturbation evaluation** — the thesis frames perturbations as part of OOD generalization; applying them to in-domain data would conflate two analysis axes. Only OOD datasets are perturbed.
- **Behavior-level perturbation evaluation** — perturbation effects are measured at the detection stage only. Propagating perturbed inputs through the full pipeline (tracking → tubelets → VideoMAE) is out of scope.
- **Depth-based or physics-based rain/fog simulation** — albumentations stochastic transforms are sufficient for a thesis-level robustness evaluation. No raymarching or depth-aware rendering.
- **More than two severity levels** — low and high are sufficient to show gradient effects. Intermediate severities add compute without adding thesis value.
- **Additional perturbation types** (JPEG compression artifacts, geometric distortions, occlusion simulation) — the five chosen types cover the principal outdoor environmental degradation axes for cattle monitoring.
- **Automated threshold tuning under perturbation** — confidence threshold remains 0.3 for all conditions, matching all prior OOD evaluations.
- **Re-training or fine-tuning the detector on perturbed data** — no model changes of any kind.
- **HuggingFace uploads** — out of scope for this PRD.

## Further Notes

- The clean baseline values for all four OOD datasets are already committed to `results/detection/` as JSON files. The perturbation runner reads these directly; clean inference does not need to be re-run at any point.
- `results/generalization/perturbation_runs/` (intermediate per-condition JSONs) should be added to `.gitignore`; only `perturbation_delta.csv` is committed. The intermediate files are individually small but 40 of them add unnecessary repository noise.
- The thesis abstract currently reads "cross-dataset testing and controlled environmental perturbations" — this PRD fulfills that commitment. No abstract revision is needed.
- Issue #24 (§6.4.2 pipeline error propagation) should be updated after `perturbation_delta.csv` is populated to incorporate the perturbation sensitivity discussion as a third axis: detection FP/FN propagation → tracking fragmentation → behavior window contamination → perturbation sensitivity stratified by domain shift magnitude.
- GPU availability: the same GPU session used for issue #22 (per-dataset in-domain detection AP) can run the perturbation eval back-to-back since the same checkpoint and eval script are used.
