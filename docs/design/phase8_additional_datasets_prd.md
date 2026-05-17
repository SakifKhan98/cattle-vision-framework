# Phase 8 — Additional Dataset Evaluation & Generalization Analysis
## Product Requirements Document

**Author:** Sakif Khan  
**Date:** 2026-05-17  
**Status:** Planning

---

## 0. Context and Phase Placement

Phases 1–7 are complete. The pipeline (RF-DETR → SAM2 → OC-SORT → VideoMAE → analytics) runs end-to-end on CBVD-5 and CVB. Phase 7 analytics are committed.

**Phase 8 implements thesis proposal §4.5 — "Generalization and Robustness Evaluation."**

That section is currently unimplemented. It requires running the trained system on four additional datasets that were not used for behavior training: OpenCows2020, Cows2021, CattleEyeView, and Freeman Center. The goal is to assess how the system generalizes across environments, viewpoints, and annotation styles — a core research contribution of the thesis.

These datasets were originally listed as part of Phase 7 Step I but were deferred (see `docs/design/phase7_cleanup_prd.md` §13). They are now formalized as Phase 8.

---

## 1. Scope

Phase 8 covers:
1. Download and preprocessing of each additional dataset.
2. Running the relevant pipeline stages (detect / segment / track / classify / analytics) on each dataset.
3. Computing the thesis-specified evaluation metrics at each stage.
4. Writing per-dataset evaluation scripts and result summaries.

Phase 8 does **not** cover:
- Retraining any model on new data (all models are frozen from Phases 1–5).
- Controlled environmental perturbations (§4.5.2) — those are a separate Phase 9 task.
- Writing the final thesis document (out of scope for the codebase).

---

## 2. Datasets and Their Roles

Based on thesis proposal Table 2 and §4.2.1–§4.2.2:

| Dataset | Data type | Annotations | Pipeline stages to run | Primary metrics |
|---|---|---|---|---|
| **OpenCows2020** | Images (indoor + outdoor UAV) | Boxes, cow IDs (no behavior) | Detection only | mAP@50 (OOD generalization) |
| **Cows2021** | Images + short clips (indoor, top-down) | Oriented boxes, cow IDs (no behavior) | Detection + short-term tracking | mAP@50, IDF1, IDSW |
| **CattleEyeView** | Continuous video (outdoor, top-down) | Boxes, instance masks, track IDs (no behavior) | Detection + Segmentation + Tracking | mAP@50, Mask IoU, IDF1, IDSW |
| **Freeman Center** | Raw ranch videos (real outdoor) | Boxes, behavior, movement labels | Full pipeline (detect→seg→track→classify→analytics) | mAP@50, IDF1, behavior macro-F1, activity budgets |

---

## 3. Thesis Metrics Required Per Dataset

### 3.1 OpenCows2020
- **Detection mAP@50** on their test split using the RF-DETR checkpoint trained on CBVD-5+CVB.
- This quantifies out-of-distribution generalization for the detection stage.
- Report as a single number alongside the 70.4% in-domain result (CLAUDE.md §7).

### 3.2 Cows2021
- **Detection mAP@50** on their images/clips (oriented boxes → convert to axis-aligned for RF-DETR).
- **Short-term tracking IDF1 and IDSW** on their short video clips.
- Identity evaluation follows §4.5.3.3 of the proposal.

### 3.3 CattleEyeView
- **Detection mAP@50** (their annotated boxes).
- **Mask IoU** comparing SAM2 predictions against their ground-truth instance masks (§4.5.3.2).
- **IDF1 and IDSW** from OC-SORT tracking vs. their track IDs.

### 3.4 Freeman Center
- **Detection mAP@50** (if box annotations are available per frame).
- **IDF1 and IDSW** from tracking.
- **Behavior macro-F1** using the VideoMAE combined model (videomae_combined_v1) — this is the primary result: descriptive behavior evaluation under real-world ranch conditions.
- **Activity budgets and behavioral deviation** from `src/analytics/` — same pipeline as CBVD-5/CVB.
- Report separately from CBVD-5/CVB results, labeled "out-of-distribution / descriptive."

---

## 4. Implementation Order

Ordered by increasing effort and dependency:

```
Step A — OpenCows2020 eval (detection only, image dataset)
Step B — Cows2021 eval (detection + short-term tracking, image + clip dataset)
Step C — CattleEyeView eval (detection + segmentation + tracking, continuous video)
Step D — Freeman Center full pipeline (all stages + behavior + analytics)
```

Steps A–C are independent of each other once their data is downloaded. Step D depends on clarifying the Freeman Center annotation format first.

---

## 5. Step-by-Step Implementation Plan

### Step A — OpenCows2020 Evaluation

**Goal:** Compute detection mAP@50 on an out-of-distribution image dataset.

**Data format (from paper):** Images with COCO-format bounding box annotations and cow identity labels. No video, no behavior labels.

**Steps:**
1. Download OpenCows2020 dataset to `data/raw/opencows2020/`.
2. Write `scripts/13_prepare_opencows2020.sh` → `src/data/convert_opencows2020.py`:
   - Convert their annotation format to COCO detection format.
   - Output: `data/processed/detection/opencows2020/{split}/_annotations.coco.json` + images.
3. Write `scripts/14_eval_opencows2020.sh` → calls `src/detection/evaluate.py`:
   - Load the RF-DETR checkpoint (`runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`).
   - Run inference on the OpenCows2020 test split.
   - Compute mAP@50 using pycocotools.
   - Save results to `results/detection/opencows2020_eval.json`.
4. Update `docs/datasets.md` with OpenCows2020 download instructions.
5. Update `results/detection/` summary table.

**Key unknown:** Annotation format of OpenCows2020 (COCO? custom CSV?). Resolve on download.

---

### Step B — Cows2021 Evaluation

**Goal:** Compute detection mAP@50 and short-term tracking IDF1 on a top-down indoor dataset.

**Data format (from paper):** Images and short video clips; annotations include oriented bounding boxes and cow identity labels.

**Steps:**
1. Download Cows2021 to `data/raw/cows2021/`.
2. Write `scripts/15_prepare_cows2021.sh` → `src/data/convert_cows2021.py`:
   - Convert oriented boxes to axis-aligned AABB for RF-DETR evaluation (take the enclosing AABB).
   - Output: `data/processed/detection/cows2021/` in COCO format.
   - For video clips: extract frames and produce frame-level detection annotations.
3. Write `scripts/16_eval_cows2021_detection.sh`:
   - Same evaluation pattern as Step A. Output: `results/detection/cows2021_eval.json`.
4. Write `scripts/17_eval_cows2021_tracking.sh` → `src/tracking/evaluate_tracking.py`:
   - Run OC-SORT on Cows2021 video clips using RF-DETR detections as input.
   - Evaluate against identity ground truth using TrackEval.
   - Output: `results/tracking/cows2021_tracking_eval.json` (IDF1, IDSW).
5. Update `docs/datasets.md`.

**Key unknown:** Whether Cows2021 clips are long enough to produce meaningful IDF1 (short clips may give trivially high IDF1). Note in results if so.

---

### Step C — CattleEyeView Evaluation

**Goal:** Compute detection mAP@50, Mask IoU (segmentation), and IDF1/IDSW (tracking).

**Data format (from paper, Ong et al. 2023):** Continuous top-down outdoor videos with COCO-format annotations including bounding boxes, instance segmentation masks, and track IDs.

**Steps:**
1. Download CattleEyeView to `data/raw/cattleeyeview/`.
2. Write `scripts/18_prepare_cattleeyeview.sh` → `src/data/convert_cattleeyeview.py`:
   - Convert to COCO detection format for the detection eval.
   - Keep masks in COCO RLE/polygon format for segmentation eval.
   - Output: `data/processed/detection/cattleeyeview/` and `data/processed/segmentation/cattleeyeview/`.
3. Write `scripts/19_eval_cattleeyeview_detection.sh`:
   - mAP@50 on their annotated frames. Output: `results/detection/cattleeyeview_eval.json`.
4. Write `scripts/20_eval_cattleeyeview_segmentation.sh` → `src/segmentation/evaluate_masks.py`:
   - Run SAM2 on CattleEyeView frames, prompted with RF-DETR boxes.
   - Compute Mask IoU against ground-truth instance masks (pycocotools).
   - Output: `results/segmentation/cattleeyeview_maskiou.json`.
5. Write `scripts/21_eval_cattleeyeview_tracking.sh`:
   - Run OC-SORT on CattleEyeView video sequences.
   - Evaluate IDF1 and IDSW against their track ID ground truth via TrackEval.
   - Output: `results/tracking/cattleeyeview_tracking_eval.json`.
6. Update `docs/datasets.md`.

---

### Step D — Freeman Center Full Pipeline

**Goal:** Run the full pipeline on real outdoor ranch data and produce behavior evaluation + analytics.

**Data format:** To be confirmed at start of implementation (see §6 Open Questions). The user has the data and labels locally but annotation format and exact paths are unknown.

**Expected steps (subject to revision after annotation format is confirmed):**
1. Clarify annotation format and data location (first thing to do before any code).
2. Write `scripts/22_prepare_freeman.sh` → `src/data/convert_freeman.py`:
   - Map Freeman behavior labels to the 7-class taxonomy (Table 3 in thesis proposal §4.2.2.1).
   - Key mappings: grazing/hay-feeding → Foraging(2), ruminating → Ruminating(4), grooming → Grooming(5), normal/other → Other(6), standing/lying from movement labels.
   - Output: `data/processed/detection/freeman/` and optionally `data/processed/behavior/freeman/`.
3. Write `scripts/23_run_freeman_detection.sh`:
   - Run RF-DETR detector on Freeman Center videos.
   - Evaluate mAP@50 if frame-level box annotations are available.
4. Write `scripts/24_run_freeman_segmentation.sh`:
   - SAM2 segmentation on Freeman Center (same pattern as script 07).
5. Write `scripts/25_run_freeman_tracking.sh`:
   - OC-SORT on Freeman Center, producing `data/processed/tracking_v2/freeman/`.
6. Write `scripts/26_generate_freeman_tubelets.sh`:
   - Tubelet extraction (16 frames, stride 4, 224×224) — same as script 09.
7. Write `scripts/27_eval_freeman_behavior.sh`:
   - Run `src/behavior/evaluate.py` with the `videomae_combined_v1` checkpoint.
   - This is a zero-shot OOD evaluation (no Freeman Center training data).
   - Output: `results/behavior/predictions/freeman_eval.csv`.
8. Extend `scripts/12_generate_analytics.sh` (or write `scripts/28_freeman_analytics.sh`):
   - Run `src/analytics/timeline.py` and `budget.py` on Freeman Center predictions.
   - Output: `results/analytics/` with Freeman-specific summaries.
9. Write `results/behavior/freeman_eval_summary.md` with the behavior macro-F1 breakdown.

---

## 6. New Files and Scripts

The scripts are numbered to continue after the existing 12:

| Script | Source module | Purpose |
|---|---|---|
| `scripts/13_prepare_opencows2020.sh` | `src/data/convert_opencows2020.py` | OpenCows2020 → COCO detection format |
| `scripts/14_eval_opencows2020.sh` | `src/detection/evaluate.py` | RF-DETR mAP@50 on OpenCows2020 |
| `scripts/15_prepare_cows2021.sh` | `src/data/convert_cows2021.py` | Cows2021 → COCO + oriented-box adapter |
| `scripts/16_eval_cows2021_detection.sh` | `src/detection/evaluate.py` | RF-DETR mAP@50 on Cows2021 |
| `scripts/17_eval_cows2021_tracking.sh` | `src/tracking/evaluate_tracking.py` | IDF1 + IDSW on Cows2021 short clips |
| `scripts/18_prepare_cattleeyeview.sh` | `src/data/convert_cattleeyeview.py` | CattleEyeView → COCO detection + mask format |
| `scripts/19_eval_cattleeyeview_detection.sh` | `src/detection/evaluate.py` | RF-DETR mAP@50 on CattleEyeView |
| `scripts/20_eval_cattleeyeview_segmentation.sh` | `src/segmentation/evaluate_masks.py` | SAM2 Mask IoU on CattleEyeView |
| `scripts/21_eval_cattleeyeview_tracking.sh` | `src/tracking/evaluate_tracking.py` | IDF1 + IDSW on CattleEyeView |
| `scripts/22_prepare_freeman.sh` | `src/data/convert_freeman.py` | Freeman Center → 7-class taxonomy + COCO format |
| `scripts/23_run_freeman_detection.sh` | `src/detection/infer_dataset.py` | RF-DETR inference on Freeman Center |
| `scripts/24_run_freeman_segmentation.sh` | `src/segmentation/segment.py` | SAM2 on Freeman Center |
| `scripts/25_run_freeman_tracking.sh` | `src/tracking/track.py` | OC-SORT on Freeman Center |
| `scripts/26_generate_freeman_tubelets.sh` | `src/data/export_tubelets.py` | Tubelet extraction for Freeman Center |
| `scripts/27_eval_freeman_behavior.sh` | `src/behavior/evaluate.py` | VideoMAE combined model on Freeman |
| `scripts/28_freeman_analytics.sh` | `src/analytics/timeline.py` + `budget.py` | Activity budgets + deviation for Freeman |

New source modules:
- `src/data/convert_opencows2020.py`
- `src/data/convert_cows2021.py`
- `src/data/convert_cattleeyeview.py`
- `src/data/convert_freeman.py`
- `src/segmentation/evaluate_masks.py`
- `src/tracking/evaluate_tracking.py`

Note: `src/detection/evaluate.py` likely needs to be written or verified — check whether it already exists.

---

## 7. Result File Locations

```
results/
├── detection/
│   ├── opencows2020_eval.json        ← Step A
│   ├── cows2021_eval.json            ← Step B
│   ├── cattleeyeview_eval.json       ← Step C
│   └── freeman_detection_eval.json   ← Step D
├── segmentation/
│   └── cattleeyeview_maskiou.json    ← Step C
├── tracking/
│   ├── cows2021_tracking_eval.json   ← Step B
│   ├── cattleeyeview_tracking_eval.json ← Step C
│   └── freeman_tracking_eval.json    ← Step D
├── behavior/
│   ├── predictions/
│   │   └── freeman_eval.csv          ← Step D
│   └── freeman_eval_summary.md       ← Step D
└── analytics/
    └── (freeman timelines, budgets, deviation — Step D)
```

---

## 8. Frozen Constraints Carried Forward

All model checkpoints are frozen — no retraining in Phase 8:
- Detector: `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`
- Behavior: `runs/behavior/videomae_combined_v1/checkpoint_best.pt`
- SAM2: `weights/sam2.1_hiera_large.pt`

The 7-class label map (`data/label_map.json`) is frozen. Freeman Center behavior harmonization must map onto the existing IDs, not extend them.

GPU constraints remain unchanged (RTX 3060, 12 GB) for local runs. Steps D's tubelet extraction and VideoMAE inference may need LEAP2 if the Freeman Center dataset is large.

---

## 9. Open Questions

1. **Freeman Center annotation format** — path, schema, and video format not yet confirmed. This must be the first thing resolved before Step D can start. Blocks script 22.

2. **OpenCows2020 annotation format** — the paper (Andrew et al. 2020) describes a custom format; may require inspection on download.

3. **Cows2021 oriented box format** — whether their annotations are COCO-format rotated boxes or a custom angle+box schema; affects the converter design in Step B.

4. **CattleEyeView download access** — dataset may require a request form or license agreement; check before starting Step C.

5. **`src/detection/evaluate.py` existence** — the CLAUDE.md pipeline only mentions `infer_dataset.py`, not a separate evaluate script. Check what exists in `src/detection/` before writing Steps A–D scripts.

6. **Freeman Center dataset size** — if videos are hours-long continuous recordings, tubelet count could exceed 125k (current total for CBVD-5+CVB). May require LEAP2 for tubelets and inference.

---

## 10. Step Completion Checklist

- [ ] Step A — OpenCows2020 detection eval
- [ ] Step B — Cows2021 detection + tracking eval
- [ ] Step C — CattleEyeView detection + segmentation + tracking eval
- [ ] Step D — Freeman Center full pipeline + behavior eval + analytics
- [ ] Update `docs/datasets.md` with download instructions for all 4 datasets
- [ ] Update `CLAUDE.md` §7 Key Results table with Phase 8 results
- [ ] Commit all result CSVs and JSON files to `results/`
