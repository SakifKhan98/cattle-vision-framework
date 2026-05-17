# Pipeline Walkthrough

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

All scripts run from the repository root. Each depends on the output of the previous.
See [docs/setup.md](setup.md) for installation and weight download instructions before running.

---

## Overview

```
01_inspect_data      → 02_prepare_cbvd5  ─┐
                     → 03_prepare_cvb    ─┤→ 04_merge_datasets → 05_train_detector
                                                                       ↓
                                                             06_run_detection
                                                                       ↓
                                        07_run_segmentation ──→ 08_run_tracking
                                        (SAM2 + OC-SORT)       (OC-SORT only)
                                                 ↓
                                       09_generate_tubelets
                                                 ↓
                                       10_train_behavior
                                                 ↓
                                          11_evaluate
                                                 ↓
                                      12_generate_analytics
```

---

## Script 01 — Inspect Data

**What it does:** Prints file counts, video counts, and annotation counts for both datasets.
No output files are written — it is a sanity check only.

**Prerequisites:** `data/raw/cbvd5/` and `data/raw/cvb/` downloaded (see [docs/datasets.md](datasets.md)).

**Command:**
```bash
bash scripts/01_inspect_data.sh
```

**Expected runtime:** < 1 minute.

**Output verification:** Console summary listing video and annotation file counts.

---

## Script 02 — Prepare CBVD-5

**What it does:** Converts AVA-format CSV annotations to COCO detection format with
train/valid/test splits. Handles multi-label priority (Drinking > Foraging > Ruminating > Lying > Standing).

**Prerequisites:** `data/raw/cbvd5/` with `annotations/` and `videos/` subdirectories.

**Command:**
```bash
bash scripts/02_prepare_cbvd5.sh
```

**Expected runtime:** ~5–10 minutes.

**Output:** `data/processed/detection/cbvd5/{train,valid,test}/_annotations.coco.json` + image symlinks.

**Verification:**
```bash
python -c "
import json
for split in ['train','valid','test']:
    d = json.load(open(f'data/processed/detection/cbvd5/{split}/_annotations.coco.json'))
    print(split, len(d['images']), 'images,', len(d['annotations']), 'anns')
"
```

**Note:** CBVD-5 test split = validation split (test video IDs are identical to val IDs — the dataset has no released test labels).

---

## Script 03 — Prepare CVB

**What it does:** Converts CVB JSON annotations to COCO detection format. Filters to only
frames where the image file is actually present on disk (not all annotated frames have images).
Produces train/valid splits (no test split).

**Prerequisites:** `data/raw/cvb/` with `annotations/` and `raw_frames/` subdirectories.

**Command:**
```bash
bash scripts/03_prepare_cvb.sh
```

**Expected runtime:** ~10–20 minutes.

**Output:** `data/processed/detection/cvb/{train,valid}/_annotations.coco.json` + image symlinks.

**Verification:**
```bash
python -c "
import json
for split in ['train','valid']:
    d = json.load(open(f'data/processed/detection/cvb/{split}/_annotations.coco.json'))
    print(split, len(d['images']), 'images,', len(d['annotations']), 'anns')
"
```

---

## Script 04 — Merge Datasets

**What it does:** Merges CBVD-5 and CVB COCO detection datasets into a single combined
dataset for joint training. Re-indexes image and annotation IDs to avoid collisions.

**Prerequisites:** Scripts 02 and 03 complete.

**Command:**
```bash
bash scripts/04_merge_datasets.sh
```

**Expected runtime:** < 2 minutes.

**Output:** `data/processed/detection/combined/{train,valid}/_annotations.coco.json`.

---

## Script 05 — Train RF-DETR Detector

**What it does:** Fine-tunes RF-DETR-Medium on the combined cattle detection dataset
(CBVD-5 + CVB). Saves best checkpoint by total loss + mAP.

**Prerequisites:**
- Script 04 complete
- `weights/rf-detr-medium.pth` downloaded from HuggingFace
- GPU with ≥12 GB VRAM

**Command:**
```bash
bash scripts/05_train_detector.sh
# or specify a config:
bash scripts/05_train_detector.sh configs/detection/rfdetr_combined.yaml
```

**Expected runtime:** Several hours (RTX 3060) / ~1–2 hours (V100).

**Output:** `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`

**Skip:** Download pretrained checkpoint from HuggingFace instead:
```bash
huggingface-cli download sakifkhan98/cattle-vision-framework rfdetr_combined_v1_best.pth \
  --local-dir runs/detection/rfdetr_combined_v1/
mv runs/detection/rfdetr_combined_v1/rfdetr_combined_v1_best.pth \
   runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth
```

**Results:** mAP@50 = 70.4% on combined cross-domain evaluation (see [docs/results.md](results.md)).

---

## Script 06 — Run Detection Inference

**What it does:** Runs the trained RF-DETR detector on all videos in both datasets,
saving per-frame bounding box detections as JSON files.

**Prerequisites:**
- `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`
- `data/raw/{cbvd5,cvb}/videos/` present

**Command:**
```bash
bash scripts/06_run_detection.sh
```

**Expected runtime:** Several hours (all videos in both datasets).

**Output:** `data/processed/tracking/{cbvd5,cvb}/{video_id}_detections.json`

**Verification:**
```bash
ls data/processed/tracking/cbvd5/ | wc -l  # ~687 videos
ls data/processed/tracking/cvb/   | wc -l  # ~502 videos
```

---

## Script 07a — Run SAM2 Segmentation

**What it does:** Runs SAM2 (segment-anything-2) on detection outputs to produce per-frame
instance masks encoded as RLE. These masks are used for precise tubelet cropping.
Generates tracking_v2 JSONs that include both OC-SORT track IDs and mask RLEs.

**Prerequisites:**
- Script 06 complete
- `weights/sam2.1_hiera_large.pt` downloaded
- `sam2` installed: `pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'`
- GPU with ≥12 GB VRAM (memory-heavy)

**Command:**
```bash
bash scripts/07_run_segmentation.sh
```

**Expected runtime:** ~28 min (CBVD-5, 687 videos) + ~5.5 hours (CVB, 502 videos).

**Output:** `data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json` (with `mask_rle` field).

**Total masks produced:** CBVD-5 = 15,900 | CVB = 226,789

**Skip:** Download pre-computed tracking_v2 from HuggingFace (see [docs/setup.md §7](setup.md)).

---

## Script 07b — Train RF-DETR-Seg (Optional)

**What it does:** Trains an RF-DETR-Seg model using SAM2 pseudo-labels as distillation targets.
This produces a fast single-stage detector+segmentor. Run on HiPE1 via Docker.

**Prerequisites:**
- `data/rfdetr_seg/cattle/` — SAM2 pseudo-label training set
- `weights/rf-detr-medium.pth`
- Docker image `cattle-rfdetr-seg:v1` loaded on HiPE1

**Command (HiPE1):**
```bash
bash scripts/07_train_rfdetr_seg.sh
# or see scripts/hipe/README.md for the full Docker run command
```

**Skip:** Download pretrained RF-DETR-Seg checkpoint:
```bash
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-seg-medium.pt \
  --local-dir weights/
```

---

## Script 08 — Run OC-SORT Tracking

**What it does:** Runs OC-SORT (Observation-Centric SORT) multi-object tracker on detection
outputs. Produces tracking JSONs **without** mask RLEs. Only needed if script 07a was skipped
(i.e., you want box-only tracks without SAM2 masks).

**Prerequisites:**
- Script 06 complete
- `third_party/OC_SORT/` cloned (see [docs/setup.md §4.1](setup.md))

**Command:**
```bash
bash scripts/08_run_tracking.sh
```

**Expected runtime:** ~1–3 hours depending on video count.

**Output:** `data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json` (no `mask_rle` field).

**Note:** If script 07a was run, `tracking_v2/` already exists with mask RLEs. Skip this script.

---

## Script 09 — Generate Tubelets

**What it does:** Extracts fixed-length tubelet clips from tracking_v2 outputs. For each track,
slices 16-frame windows at stride 4, crops the bounding-box region (padded by 20 px),
and saves frames as JPEG sequences. Assigns behavior labels via majority vote over annotated frames.

**Prerequisites:**
- Script 07a or 08 complete (or tracking_v2 downloaded from HuggingFace)
- `data/raw/{cbvd5,cvb}/` present (raw frames needed for cropping)

**Command:**
```bash
bash scripts/09_generate_tubelets.sh
```

**Expected runtime:** Several hours (CPU-only, I/O bound, 125,586 clips).

**Output:**
- `data/processed/tubelets/{cbvd5,cvb}/{video_id}/{track_id}/frame_*.jpg`
- `data/processed/tubelets/labels.csv` (columns: dataset, video_id, tubelet_dir, start_frame, end_frame, label_id)

**Verification:**
```bash
wc -l data/processed/tubelets/labels.csv  # ~125,587 lines (header + 125,586 clips)
```

---

## Script 10 — Train VideoMAE Behavior Classifier

**What it does:** Fine-tunes VideoMAE-Base on tubelet clips for 7-class behavior recognition.
Supports 5 training configurations (in-domain, cross-domain, combined). Run on HiPE1 via Docker
due to GPU memory requirements.

**Prerequisites:**
- Script 09 complete
- GPU with ≥12 GB VRAM
- Docker (for HiPE1 path) or conda env (local)

**Command:**
```bash
bash scripts/10_train_behavior.sh                                      # all 5 configs
bash scripts/10_train_behavior.sh configs/behavior/videomae_combined.yaml  # single config
```

**Expected runtime:** ~1–2 hours per config on V100 (5 configs total).

**Output:** `runs/behavior/videomae_{name}_v1/checkpoint_best.pt`

**Skip:** Download pretrained checkpoints from HuggingFace (see [docs/setup.md §5](setup.md)).

---

## Script 11 — Evaluate Behavior Classifiers

**What it does:** Runs evaluation for all 5 VideoMAE configs on their respective validation splits.
Computes macro-F1, per-class F1, precision, recall. Writes prediction CSVs and confusion matrix PNGs.

**Prerequisites:**
- Script 09 complete
- Checkpoints in `runs/behavior/videomae_*/checkpoint_best.pt`

**Command:**
```bash
bash scripts/11_evaluate.sh                                             # all 5 configs
bash scripts/11_evaluate.sh configs/behavior/videomae_combined.yaml \
     runs/behavior/videomae_combined_v1/checkpoint_best.pt              # single config
```

**Expected runtime:** ~10–30 minutes per config (GPU) or several hours (CPU).

**Output:**
- `results/behavior/predictions/{config}_val.csv`
- `results/behavior/confusion_matrices/{config}.png`
- `results/behavior/f1_per_class.csv` (all configs)

**Note:** These CSVs are committed to git for analytics reproducibility.

---

## Script 12 — Generate Analytics

**What it does:** Computes activity timelines, activity budgets, behavior transition matrices,
and welfare-relevant flags from the prediction CSVs + tracking data.

**Prerequisites:**
- Script 11 complete (or `results/behavior/predictions/` downloaded from git)
- `data/processed/tracking_v2/{cbvd5,cvb}/` present (for per-frame timeline alignment)

**Command:**
```bash
bash scripts/12_generate_analytics.sh
```

**Expected runtime:** < 5 minutes (CPU-only).

**Output:**
- `results/analytics/timelines/` (per-video timelines — gitignored, large)
- `results/analytics/activity_budget.csv`
- `results/analytics/transition_matrix.csv`
- `results/analytics/welfare_flags.csv`

---

## End-to-End Pipeline Runner

Run all stages in order (or resume from a specific stage):

```bash
bash scripts/run_pipeline.sh             # all stages
bash scripts/run_pipeline.sh --from 9   # resume from stage 9
bash scripts/run_pipeline.sh --stage 12 # single stage
```

See `scripts/run_pipeline.sh` comments for full usage.

---

## Data Flow Summary

| Script | Input | Output | GPU? |
|--------|-------|--------|------|
| 01 | `data/raw/` | console | no |
| 02 | `data/raw/cbvd5/` | `data/processed/detection/cbvd5/` | no |
| 03 | `data/raw/cvb/` | `data/processed/detection/cvb/` | no |
| 04 | `detection/{cbvd5,cvb}/` | `detection/combined/` | no |
| 05 | `detection/combined/` + `weights/rf-detr-medium.pth` | `runs/detection/rfdetr_combined_v1/` | yes |
| 06 | `raw videos` + detector checkpoint | `tracking/{cbvd5,cvb}/` | yes |
| 07a | `tracking/` + SAM2 checkpoint | `tracking_v2/` (with masks) | yes |
| 07b | `data/rfdetr_seg/` | `runs/seg_medium_lr1e4/` | yes (HiPE1) |
| 08 | `tracking/` (box-only, skip if 07a ran) | `tracking_v2/` (no masks) | no |
| 09 | `tracking_v2/` + `data/raw/` | `tubelets/` (125k clips) | no |
| 10 | `tubelets/` + configs | `runs/behavior/` | yes (HiPE1) |
| 11 | `tubelets/` + checkpoints | `results/behavior/` | yes |
| 12 | `results/behavior/predictions/` + `tracking_v2/` | `results/analytics/` | no |
