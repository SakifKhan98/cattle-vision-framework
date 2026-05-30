# Thesis Data Completion PRD

## Problem Statement

Phases 0–9 of the cattle vision framework are complete, but several results files that the thesis document directly cites are either empty (0 bytes) or missing entirely. The thesis outline (Chapters IV–VI) requires per-dataset detection AP for CBVD-5 and CVB, CVB-only tracking metrics, v2 training curves for all five behavior configs, a `min_hits` hyperparameter ablation table, and a consolidated OOD detection summary. Without these, six thesis sections cannot be written with numbers. All underlying experiments have already been run; the gap is in evaluation scripts, log parsing, and results aggregation — not in re-training models.

## Solution

Run a set of targeted scripts and one new tracking sweep to fill every empty results file cited by the thesis outline. The work falls into three categories:

1. **Pure aggregation** — reshape already-computed outputs into the CSV/JSON schemas the thesis expects (Tasks 1, 2, 3).
2. **Targeted re-evaluation** — run the existing evaluation script against in-domain test splits using the already-downloaded checkpoint (Task 4).
3. **Small tracking sweep** — re-run OC-SORT with `min_hits ∈ {1, 2, 5}` on CVB (mh=3 canonical run already exists; mh=5 and mh=7 archives already exist) and evaluate to produce an ablation table (Task 5).
4. **Analytical write-up** — synthesize the completed metrics into the pipeline error propagation section of the thesis (Task 6).

Perturbation experiments (controlled rain/fog/noise) are explicitly out of scope and the corresponding thesis section (§5.4.2) will be revised to reflect cross-dataset generalization analysis only.

## User Stories

1. As a thesis author, I want `results/generalization/ood_summary.csv` populated with all four OOD dataset detection metrics, so that I can write §6.1.2 and §6.4.1 with a single consolidated reference table.
2. As a thesis author, I want CattleEyeView segmentation metrics (Mask IoU = 86.5%) included in the OOD summary, so that the segmentation cross-domain result is co-located with detection results.
3. As a thesis author, I want `results/tracking/cvb_idf1.json` populated with CVB-scoped tracking metrics, so that §6.2.1 can report IDF1/MOTA broken down by dataset.
4. As a thesis author, I want `results/tracking/cbvd5_idf1.json` to contain an explicit documented stub explaining why CBVD-5 tracking metrics cannot be computed, so that the thesis can acknowledge this limitation cleanly.
5. As a thesis author, I want v2 training log CSVs for all five behavior configs in `results/behavior/training_logs/`, so that training curve figures can be generated for both v1 and v2 models.
6. As a thesis author, I want the v2 CSV schema to match the v1 schema exactly, so that existing figure-generation notebooks load both without modification.
7. As a thesis author, I want `results/detection/cbvd5_test_ap.json` populated with CBVD-5 in-domain test AP metrics, so that §6.1.1 can report per-dataset detection performance alongside the combined 70.4% mAP@50 figure.
8. As a thesis author, I want `results/detection/cvb_test_ap.json` populated with CVB in-domain test AP metrics, so that detection performance can be compared across the two primary training datasets.
9. As a thesis author, I want a `results/tracking/minhits_ablation.csv` table with IDF1/MOTA/IDSW for `min_hits ∈ {1, 2, 3, 5}`, so that §6.2.2 can justify the choice of `min_hits=3` with quantitative evidence.
10. As a thesis author, I want per-min_hits summary JSONs stored under `results/tracking/`, so that the ablation table can be regenerated from source data if the CSV is lost.
11. As a thesis author, I want the canonical `results/tracking/tracking_summary_all.json` and `tracking_per_video_all.csv` files preserved before any ablation eval overwrites them, so that the mh=3 ground truth is not lost.
12. As a thesis author, I want §6.4.2 (Pipeline Error Propagation Analysis) written using the completed detection, tracking, and behavior metrics, so that the thesis demonstrates how errors propagate from detection FP/FN through tracking fragmentation into behavior misclassification.
13. As a thesis author, I want the error propagation analysis to use only already-computed metrics (no new experiments), so that thesis writing is not blocked on additional compute.
14. As a future reader of the codebase, I want each aggregation script to be self-contained and re-runnable, so that results CSVs can be regenerated if source JSONs are updated.

## Implementation Decisions

### Task 1 — OOD Summary CSV

- A new script `scripts/aggregate_ood_summary.py` reads the four OOD detection JSONs (`opencows2020_eval.json`, `cows2021_eval.json`, `cattleeyeview_eval.json`, `freeman_detection_eval.json`) and the CattleEyeView segmentation JSON (`cattleeyeview_maskiou.json`).
- Output schema: `dataset, n_images, mAP50, mAP50_95, AR100, mAP_s, mAP_m, mAP_l, mean_mask_iou, domain_shift_note`. The `mean_mask_iou` column is null for all datasets except CattleEyeView.
- The `domain_shift_note` column captures a one-phrase descriptor per dataset (e.g., "aerial top-down UAV", "indoor UK barn", "top-down outdoor polygon masks", "angled real ranch").
- After Task 4 completes, the in-domain CBVD-5 and CVB rows are appended to the same CSV so all detection AP results live in one place.
- The empty `results/detection/combined_ood.json` is left as-is (vestigial placeholder); `ood_summary.csv` is the canonical reference going forward.

### Task 2 — CVB and CBVD-5 Tracking JSONs

- All 447 rows in `tracking_per_video_all.csv` belong to CVB videos (confirmed by video ID pattern matching). Therefore `tracking_summary_all.json` is already a CVB-only summary.
- A new script `scripts/write_cvb_idf1.py` reads `tracking_summary_all.json` and `tracking_per_video_all.csv`, adds a `dataset` and `n_videos` field, and writes `results/tracking/cvb_idf1.json`.
- `results/tracking/cbvd5_idf1.json` is written as a documented stub with `computable: false` and a note explaining that CBVD-5 annotations do not carry persistent track IDs across frames, so MOT evaluation is not applicable. OC-SORT was run on CBVD-5 for tubelet generation only.

### Task 3 — v2 Training Log CSVs

- A new script `scripts/parse_v2_logs.py` parses five raw HiPE1 log files in `logs/` into structured CSVs under `results/behavior/training_logs/`.
- Log format per epoch: `Epoch N | lr=X | train_loss=A train_f1=B | val_loss=C val_macro_f1=D` followed by `per-class val F1: Standing=E  Lying=F  Foraging=G  Drinking=H  Ruminating=I  Grooming=J  Other=K`.
- Output CSV schema matches v1 exactly: `epoch, train_loss, val_loss, val_macro_f1, val_f1_standing, val_f1_lying, val_f1_foraging, val_f1_drinking, val_f1_ruminating, val_f1_grooming, val_f1_other, lr`. The `train_f1` field present in v2 logs is dropped to maintain schema parity with v1.
- Log-to-CSV mapping: `logs/cbvd5_v2.log` → `videomae_cbvd5_v2.csv`, `logs/cvb_v2.log` → `videomae_cvb_v2.csv`, `logs/combined_v2.log` → `videomae_combined_v2.csv`, `logs/cbvd5_to_cvb_v2.log` → `videomae_cbvd5_to_cvb_v2.csv`, `logs/cvb_to_cbvd5_v2.log` → `videomae_cvb_to_cbvd5_v2.csv`.

### Task 4 — Per-Dataset In-Domain Detection AP

- Uses the existing `src/tools/eval_detection_ood.py` script unchanged.
- Checkpoint: `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth` (already present locally — no download required).
- Evaluated against each dataset's **test** split under `data/processed/detection/{cbvd5,cvb}/test/`.
- Confidence threshold: `0.3` (matches all prior OOD evaluations for consistency).
- Requires the `cattletransformer` conda environment and a CUDA-capable GPU. If local GPU is unavailable, submit to HiPE1 using the Docker eval pattern established in Phase 8.
- Outputs populate the previously-empty `results/detection/cbvd5_test_ap.json` and `results/detection/cvb_test_ap.json` in the same JSON schema as the OOD detection files.
- Note: the combined validation mAP@50 = 70.4% was computed on the merged CBVD-5 + CVB validation set during training. The per-dataset test AP from this task will likely differ and should be reported alongside the combined figure in §6.1.1, not as a replacement.

### Task 5 — `min_hits` Ablation Sweep

- Sweep: `min_hits ∈ {1, 2, 3, 5}`. mh=3 is the canonical run (already evaluated). mh=5 and mh=7 tracks already exist in `_archive/runs/tracking_experiments/`. Only mh=1 and mh=2 require fresh tracking runs.
- Fresh runs output to `data/processed/tracking_ablation/cvb_mh{1,2}/` to avoid touching the canonical `data/processed/tracking_v2/cvb/`.
- **Critical:** Before running any ablation evaluation, copy canonical results: `tracking_summary_all.json` → `cvb_mh3_summary.json` and `tracking_per_video_all.csv` → `cvb_mh3_per_video.csv`. The eval script always overwrites the canonical filenames.
- After each eval run, immediately copy the output to a named file (`cvb_mh{N}_summary.json`) before the next run overwrites it.
- mh=7 archived tracks are not evaluated; the ablation table uses {1, 2, 3, 5} only as four data points is sufficient to show the tradeoff.
- A final aggregation script `scripts/write_minhits_ablation.py` reads the four named summary JSONs and writes `results/tracking/minhits_ablation.csv` with columns: `min_hits, idf1, mota, motp, total_id_switches, n_videos`.
- All tracking runs use the `cattletransformer` conda env. Tracking is CPU-bound (no GPU required).

### Task 6 — Pipeline Error Propagation Analysis

- No new experiments. Uses the metrics produced by Tasks 1–5.
- The analysis traces three layers:
  1. **Detection layer:** Precision and Recall from `cbvd5_test_ap.json` + `cvb_test_ap.json` quantify how many ground-truth cattle the tracker never sees (FN) and how many spurious boxes it must suppress (FP). The deliberate choice of threshold=0.3 (high recall, lower precision) is the primary driver of high FP count.
  2. **Tracking layer:** MOTA=36.61% is suppressed by the high detector FP count (15,612 FP vs 8,722 FN). IDF1=67.31% is the relevant metric for downstream behavior because it measures identity continuity, not frame-level box accuracy. The `min_hits` ablation table (Task 5) justifies mh=3 as the FP suppression gate: lower values raise IDSW, higher values drop Recall and fragment tubelets.
  3. **Behavior layer:** Cross-domain configs (cbvd5→cvb: 0.172, cvb→cbvd5: 0.225) score 3–4× lower than in-domain configs (cbvd5: 0.451, cvb: 0.777). Domain shift entering at the detection stage propagates through tracking fragmentation into behavior window contamination.
- Written directly into the thesis §6.4.2; no separate report file needed.

### Execution Order

Tasks 1, 2, and 3 are independent and can run in any order or in parallel. Task 4 requires a GPU and the checkpoint. Task 5 can start its tracking runs (mh=1, mh=2) in parallel with Task 4 since it reads from segmentation outputs, not detection outputs. Task 6 is blocked on all prior tasks.

## Testing Decisions

These tasks are primarily data pipeline scripts rather than application logic, so testing focuses on output correctness rather than behavioral testing.

- **What makes a good test here:** verify that each output file is non-empty, has the expected schema (correct column names, correct row count where derivable), and that key known values round-trip correctly (e.g., IDF1=67.31 appears in `cvb_idf1.json`, mh=3 row in the ablation CSV matches `tracking_summary_all.json`).
- **Task 1 (ood_summary.csv):** Spot-check that `mAP50` for Freeman = 0.7298 (from `freeman_detection_eval.json`) and `mean_mask_iou` for CattleEyeView = 0.8653 (from `cattleeyeview_maskiou.json`). All other datasets should have null `mean_mask_iou`.
- **Task 2 (cvb_idf1.json):** Assert `IDF1 == 67.31` and `n_videos == 447`. Assert `cbvd5_idf1.json` contains `computable: false`.
- **Task 3 (v2 log CSVs):** For `videomae_cbvd5_v2.csv`, assert 9 rows (early stopping at epoch 9, confirmed from log), `val_macro_f1` on epoch 1 = 0.4515, and no `train_f1` column present.
- **Task 4 (detection AP JSONs):** Files are non-empty and parseable JSON; `mAP50` field is a float between 0 and 1.
- **Task 5 (ablation CSV):** Four rows with `min_hits ∈ {1, 2, 3, 5}`; mh=3 row matches `cvb_mh3_summary.json` values exactly.
- No unit test files are needed for these one-off scripts. Manual verification against known values is sufficient given the thesis context.

## Out of Scope

- **Controlled environmental perturbation experiments** (rain, fog, noise, brightness) — `results/generalization/perturbation_delta.csv` remains empty. Thesis §5.4.2 will be revised to cover cross-dataset generalization analysis only.
- **CBVD-5 tracking metrics (IDF1/MOTA)** — CBVD-5 annotations do not provide persistent track IDs across frames; these metrics are not computable and no workaround is in scope.
- **MOTA sensitivity analysis by varying `det_thresh`** — requires full detection inference re-runs at multiple thresholds followed by tracking re-runs; the qualitative explanation in the phase4 report is sufficient for the thesis.
- **Re-training any model** — all five VideoMAE configs (v1 and v2) and the RF-DETR detector are final. No retraining is in scope.
- **New dataset evaluation** — Phase 8 OOD evaluation is complete. No additional datasets are added.
- **Inference web app changes** — Phase 9 is complete and frozen.
- **HuggingFace weight uploads** — out of scope for this PRD (separate TODO in CLAUDE.md).

## Further Notes

- The canonical `tracking_summary_all.json` and `tracking_per_video_all.csv` must be preserved before any Task 5 ablation eval run, as `eval_tracking.py` unconditionally overwrites those filenames. This is the highest-risk step in the plan.
- The combined validation mAP@50 = 70.4% (reported in the thesis as the primary detection metric) was computed during training on the merged CBVD-5 + CVB validation set. The Task 4 per-dataset test AP numbers will be reported as complementary breakdowns, not replacements.
- The v2 behavior models were trained on RF-DETR-tracked tubelets (no SAM2 in the tracking loop), which is why they generally outperform v1 models on CBVD-5 and CVB in-domain configs. This distinction should be explicit in the thesis §6.3.2–6.3.3 discussion.
- CBVD-5's test split equals its validation split (no held-out test labels were released by the dataset authors). All CBVD-5 metrics reported are therefore on the validation set; this must be stated clearly in §4.2.1 and wherever CBVD-5 results appear.
