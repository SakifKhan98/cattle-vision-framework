# RF-DETR-Seg Pipeline Runbook

**RF-DETR-Seg parallel pipeline — scripts 07b through 12b**

This document is the step-by-step execution guide for the RF-DETR-Seg pipeline (Issue #1). It mirrors the original SAM2 pipeline (scripts 07–12) but uses the fine-tuned RF-DETR-Seg model for segmentation instead of SAM2, producing outputs in separate `*_rfdetr` directories so both pipelines coexist.

Run all commands from the project root (`~/TXST/Thesis/cattle-vision-framework`).

---

## Prerequisites

Verify before starting:

```bash
# RF-DETR-Seg checkpoint
ls runs/seg_medium_lr5e5/checkpoint_best_ema.pth         # ✅ must exist

# OC-SORT cloned
ls third_party/OC_SORT/trackers/                         # ✅ must exist

# Conda env
conda activate cattletransformer

# Raw data
ls data/raw/cbvd5/labelframes/labelframes/ | head -3     # 687 unique video IDs
ls data/raw/cvb/raw_frames/ | head -3                    # 502 clip directories
```

---

## Step 07b — RF-DETR-Seg Inference

**Script:** `scripts/07b_run_rfdetr_seg.sh`

**What it does:** Runs the fine-tuned RF-DETR-Seg model (Config B EMA, epoch 59) on all raw frames. For each video it writes one `{video_id}_masks.json` containing per-frame bounding boxes + COCO-RLE instance masks.

**Reads:** `data/raw/{cbvd5,cvb}/`
**Writes:** `data/processed/segmentation_rfdetr/{cbvd5,cvb}/{video_id}_masks.json`

**Skip logic:** Already-written `_masks.json` files are silently skipped — safe to resume a partial run.

**Runtime:** CBVD-5 ~6 min (0.5 s/video × 687), CVB ~5.5 h (40 s/video × 502, clips are much longer).

### Commands

```bash
# CBVD-5 (run first, fast)
bash scripts/07b_run_rfdetr_seg.sh --dataset cbvd5

# CVB (long — run in a dedicated terminal, let it finish overnight)
bash scripts/07b_run_rfdetr_seg.sh --dataset cvb

# Single video (for debugging)
bash scripts/07b_run_rfdetr_seg.sh --video_id 618

# Quick end-to-end sanity (3 videos per dataset)
bash scripts/07b_run_rfdetr_seg.sh --sanity
```

### What you'll see

```
[07b] Starting Phase 3 — RF-DETR-Seg Segmentation
  CBVD5: 687 videos
  cbvd5: 100%|████████████████| 687/687 [05:28<00:00, 2.09it/s]
  [cbvd5] Done: 685 written, 2 skipped, 0 failed, 33618 masks, 328.1s
```

The two DINOv2 backbone `[WARNING]` lines and the `torch.meshgrid` `UserWarning` are harmless — ignore them.

### Verify

```bash
ls data/processed/segmentation_rfdetr/cbvd5/ | wc -l    # expect 687
ls data/processed/segmentation_rfdetr/cvb/  | wc -l    # expect 502
```

### Status

- ✅ CBVD-5 complete (2026-05-18): 685 written, 2 skipped (sanity videos), 33 618 masks, 5.5 min
- 🔄 CVB running (2026-05-18): started ~17:04, ETA ~22:30

---

## Step 08b — OC-SORT Tracking

**Script:** `scripts/08b_run_tracking_rfdetr.sh`

**What it does:** Reads each `_masks.json` and feeds detections into OC-SORT with mask-IoU association. Writes one `{video_id}_tracks.json` per video in the same format as the SAM2 pipeline.

**Reads:** `data/processed/segmentation_rfdetr/{cbvd5,cvb}/`
**Writes:** `data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/{video_id}_tracks.json`

**Skip logic:** None — reruns overwrite existing output. Run only after 07b for that dataset is complete.

**Runtime:** CPU-only. CBVD-5 ~2–5 min (6 frames/video, trivial). CVB longer (more frames/clip) — estimate 30–60 min.

### Commands

```bash
# CBVD-5 (can run in parallel with 07b CVB — they don't share GPU)
bash scripts/08b_run_tracking_rfdetr.sh --dataset cbvd5

# CVB (run after 07b CVB finishes)
bash scripts/08b_run_tracking_rfdetr.sh --dataset cvb

# Both datasets at once (only after both 07b runs are done)
bash scripts/08b_run_tracking_rfdetr.sh

# Single video
bash scripts/08b_run_tracking_rfdetr.sh --video_id 618

# Sanity (3 videos per dataset)
bash scripts/08b_run_tracking_rfdetr.sh --sanity
```

### What you'll see

```
========================================
Step 08b: RF-DETR-Seg OC-SORT Tracking
  Seg input : data/processed/segmentation_rfdetr
  Output    : data/processed/tracking_v2_rfdetr
========================================

  Tracking cbvd5 ...
  [track] 618  → 12 unique tracks, 6 frames
  [track] 341  → 20 unique tracks, 6 frames
  ...
```

### Verify

```bash
ls data/processed/tracking_v2_rfdetr/cbvd5/ | wc -l    # expect 687 JSON files
ls data/processed/tracking_v2_rfdetr/cvb/  | wc -l    # expect 502 JSON files
```

### Status

- 🔲 CBVD-5: pending (ready to run — 07b CBVD-5 is done)
- 🔲 CVB: pending (wait for 07b CVB to finish)

---

## Step 09b — Tubelet Generation

**Script:** `scripts/09b_generate_tubelets_rfdetr.sh`

**What it does:** Reads every `_tracks.json` + corresponding raw frames. Slices each track into 16-frame clips (stride 4, 224×224 px) and writes them as PNG frame stacks. Produces `labels.csv` mapping each tubelet to a behavior label ID.

**Reads:** `data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/`, `data/raw/{cbvd5,cvb}/`
**Writes:** `data/processed/tubelets_rfdetr/{cbvd5,cvb}/`, `data/processed/tubelets_rfdetr/labels.csv`

**Runtime:** Several hours — CPU + disk I/O bound. Expect ~100k–130k tubelets total (same scale as the original pipeline's 125k).

### Commands

```bash
# CBVD-5 only (run as soon as 08b CBVD-5 is done)
bash scripts/09b_generate_tubelets_rfdetr.sh --cbvd5_only

# CVB only
bash scripts/09b_generate_tubelets_rfdetr.sh --cvb_only

# Both datasets (only after both 08b runs are done)
bash scripts/09b_generate_tubelets_rfdetr.sh

# Sanity (3 videos per dataset)
bash scripts/09b_generate_tubelets_rfdetr.sh --sanity
```

### Verify

```bash
wc -l data/processed/tubelets_rfdetr/labels.csv         # expect ~100k–130k + 1 header line
ls data/processed/tubelets_rfdetr/cbvd5/ | head -5
```

### Status

- 🔲 Pending (wait for 08b to complete)

---

## Step 10b — VideoMAE Behavior Training

**Script:** `scripts/10b_train_behavior_rfdetr.sh`

**What it does:** Trains VideoMAE on RF-DETR-Seg tubelets using 5 v2 configs (same split matrix as the original pipeline). Local mode is for smoke tests only (RTX 3060 fits but is slow). Real training must run on HiPE1 via Docker.

**Reads:** `data/processed/tubelets_rfdetr/labels.csv`, `configs/behavior/videomae_*_v2.yaml`
**Writes:** `runs/behavior/videomae_*_v2/checkpoint_best.pt` (on HiPE1)

### 5 Training Configs

| Config file                     | Split                   | Notes                    |
| ------------------------------- | ----------------------- | ------------------------ |
| `videomae_cbvd5_v2.yaml`        | CBVD-5 in-domain        |                          |
| `videomae_cvb_v2.yaml`          | CVB in-domain           |                          |
| `videomae_combined_v2.yaml`     | Combined (primary)      | use for analytics in 12b |
| `videomae_cbvd5_to_cvb_v2.yaml` | Cross-domain CBVD-5→CVB |                          |
| `videomae_cvb_to_cbvd5_v2.yaml` | Cross-domain CVB→CBVD-5 |                          |

### Commands

```bash
# HiPE1 — all 5 configs (real training)
bash scripts/10b_train_behavior_rfdetr.sh --hipe1

# HiPE1 — single config
bash scripts/10b_train_behavior_rfdetr.sh --hipe1 configs/behavior/videomae_combined_v2.yaml

# Local smoke test (RTX 3060)
bash scripts/10b_train_behavior_rfdetr.sh

# Fetch checkpoints from HiPE1 when done
rsync -avz hipe1:~/cattle_behavior/runs/behavior/ runs/behavior/
```

**HiPE1 prerequisites:** SSH alias `hipe1` configured in `~/.ssh/config`, `cattle-behavior` Docker image loaded on HiPE1. See `docs/hipe_ops.md`.

### Status

- 🔲 Pending (wait for 09b to complete, then sync tubelets to HiPE1)

---

## Step 11b — Evaluation

**Script:** `scripts/11b_evaluate_rfdetr.sh`

**What it does:** Runs `src/behavior/evaluate.py` for each v2 config using the checkpoint from 10b. Writes prediction CSVs and confusion matrix PNGs, then copies prediction CSVs into `predictions_rfdetr/` for clean separation from v1 results.

**Reads:** `data/processed/tubelets_rfdetr/labels.csv`, `runs/behavior/videomae_*_v2/checkpoint_best.pt`
**Writes:** `results/behavior/predictions_rfdetr/videomae_*_v2_val.csv`, `results/behavior/confusion_matrices/*_v2_*.png`, appends v2 rows to `results/behavior/f1_per_class.csv`

### Commands

```bash
# All 5 v2 configs (skips any whose checkpoint is missing)
bash scripts/11b_evaluate_rfdetr.sh

# Single config + checkpoint
bash scripts/11b_evaluate_rfdetr.sh \
    configs/behavior/videomae_combined_v2.yaml \
    runs/behavior/videomae_combined_v2/checkpoint_best.pt
```

### Verify

```bash
ls results/behavior/predictions_rfdetr/
cat results/behavior/f1_per_class.csv | grep v2
```

### Status

- 🔲 Pending (wait for 10b checkpoints to be fetched from HiPE1)

---

## Step 12b — Analytics

**Script:** `scripts/12b_generate_analytics_rfdetr.sh`

**What it does:** Builds per-animal behavior timelines from the combined v2 predictions + tracking JSONs, then computes activity budgets, transition matrices, and behavioral deviation. All outputs go to `results/analytics_rfdetr/` — separate from the SAM2 pipeline's `results/analytics/`.

**Reads:** `results/behavior/predictions_rfdetr/`, `data/processed/tracking_v2_rfdetr/`
**Writes:** `results/analytics_rfdetr/timelines/`, `activity_budget.csv`, `transition_matrix.csv`, `behavior_deviation.csv`

### Commands

```bash
# Default (uses videomae_combined_v2 predictions)
bash scripts/12b_generate_analytics_rfdetr.sh

# Override run name
bash scripts/12b_generate_analytics_rfdetr.sh --run videomae_cvb_v2
```

### Verify

```bash
ls results/analytics_rfdetr/
cat results/analytics_rfdetr/activity_budget.csv | head -5
```

### Status

- 🔲 Pending (wait for 11b to complete)

---

## Full Sequential Run (after all data is ready)

Once 07b CVB finishes, the complete chain from scratch is:

```bash
# --- Already done ---
# bash scripts/07b_run_rfdetr_seg.sh --dataset cbvd5   ✅

# --- Run now (CBVD-5 tracking, can start immediately) ---
bash scripts/08b_run_tracking_rfdetr.sh --dataset cbvd5
bash scripts/09b_generate_tubelets_rfdetr.sh --cbvd5_only

# --- Run after 07b CVB finishes ---
bash scripts/08b_run_tracking_rfdetr.sh --dataset cvb
bash scripts/09b_generate_tubelets_rfdetr.sh --cvb_only   # merges into same labels.csv

# --- Send to HiPE1 for training ---
bash scripts/10b_train_behavior_rfdetr.sh --hipe1

# --- After checkpoints come back ---
rsync -avz hipe1:~/cattle_behavior/runs/behavior/ runs/behavior/
bash scripts/11b_evaluate_rfdetr.sh
bash scripts/12b_generate_analytics_rfdetr.sh
```

---

## Output Directories Quick Reference

| Stage        | Output path                                       |
| ------------ | ------------------------------------------------- |
| Segmentation | `data/processed/segmentation_rfdetr/{cbvd5,cvb}/` |
| Tracking     | `data/processed/tracking_v2_rfdetr/{cbvd5,cvb}/`  |
| Tubelets     | `data/processed/tubelets_rfdetr/` + `labels.csv`  |
| Checkpoints  | `runs/behavior/videomae_*_v2/checkpoint_best.pt`  |
| Predictions  | `results/behavior/predictions_rfdetr/`            |
| Analytics    | `results/analytics_rfdetr/`                       |

---

## Parallel Execution Notes

- **07b CBVD-5** and **07b CVB** must be run sequentially (one GPU, one process).
- **08b CBVD-5** can run in a second terminal while **07b CVB** is running — tracking is CPU-only.
- **09b --cbvd5_only** can similarly run in parallel with 07b CVB.
- **10b HiPE1 configs** are independent and can be launched in parallel on HiPE1 if multiple V100 GPUs are available.
