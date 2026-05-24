# Phase 8 — Additional Dataset Evaluation & Generalization Analysis
## Product Requirements Document

**Author:** Sakif Khan
**Date:** 2026-05-17
**Last updated:** 2026-05-24
**Status:** In Progress — Steps A–C complete; Step D detection complete, video pipeline pending

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

| Dataset | Data type | Annotations | Pipeline stages run | Primary metrics |
|---|---|---|---|---|
| **OpenCows2020** | Images (indoor + outdoor UAV) | Boxes only (no behavior, no track IDs) | Detection only | mAP@50 ✅ |
| **Cows2021** | Images (indoor, top-down) | Axis-aligned boxes (no behavior, no track IDs) | Detection only | mAP@50 ✅ |
| **CattleEyeView** | Continuous video (outdoor, top-down) | Boxes + polygon instance masks (no track IDs) | Detection + Segmentation | mAP@50 ✅, Mask IoU ✅ |
| **Freeman Center** | Raw ranch videos (real outdoor) | YOLO boxes + 9-class behavior labels (no track IDs) | Detection ✅ + video pipeline pending | mAP@50 ✅, behavior macro-F1 pending |

> **Tracking evaluation note:** None of the four datasets provide ground-truth track IDs
> in their annotation format, making IDF1/IDSW evaluation impossible without a separate
> annotation effort. This was discovered on inspection; the original plan for Cows2021
> and CattleEyeView tracking eval has been dropped.

---

## 3. Thesis Metrics Required Per Dataset

### 3.1 OpenCows2020 ✅ Complete

- **Detection mAP@50:** **33.3%** (7,039 images, threshold 0.3)
- Result: `results/detection/opencows2020_eval.json`

### 3.2 Cows2021 ✅ Complete

- **Detection mAP@50:** **27.3%** (2,131 images, threshold 0.3)
- Tracking IDF1/IDSW: **not computed** — detection_and_localisation annotations carry no cow IDs.
- Result: `results/detection/cows2021_eval.json`

### 3.3 CattleEyeView ✅ Complete

- **Detection mAP@50:** **47.0%** (2,490 images, threshold 0.3, RF-DETR)
- **Box mAP@50 (RF-DETR-Seg):** **53.6%**
- **Mean Mask IoU:** **86.5%** (over 4,657 matched instances; 91.9% median)
- Tracking IDF1/IDSW: **not computed** — YOLO-format labels carry no track IDs.
- Results: `results/detection/cattleeyeview_eval.json`, `results/segmentation/cattleeyeview_maskiou.json`

### 3.4 Freeman Center — Detection ✅ Complete, Video Pipeline Pending

- **Detection mAP@50:** **73.0%** (6,625 images, threshold 0.3) — highest OOD result, matches in-domain parity
- **Behavior macro-F1:** pending (requires video pipeline: SAM2 → OC-SORT → tubelets → VideoMAE)
- **Activity budgets / behavioral deviation:** pending (same dependency)
- Tracking IDF1/IDSW: **not planned** — CMB image annotations carry no track IDs
- Result so far: `results/detection/freeman_detection_eval.json`

---

## 4. Implementation Order

```
Step A — OpenCows2020 eval (detection only)                  ✅ COMPLETE
Step B — Cows2021 eval (detection only; tracking dropped)    ✅ COMPLETE
Step C — CattleEyeView eval (detection + segmentation)       ✅ COMPLETE
Step D — Freeman Center (detection ✅ + video pipeline ⬜)
```

---

## 5. Step-by-Step Implementation Plan

### Step A — OpenCows2020 Evaluation ✅ COMPLETE

**Result:** mAP@50 = 33.3% (high domain shift — aerial top-down viewpoint)

**Artifacts produced:**
- `src/data/convert_opencows2020.py` — Supervisely JSON → COCO
- `data/processed/detection/opencows2020/` — converted COCO dataset
- `results/detection/opencows2020_eval.json`

---

### Step B — Cows2021 Evaluation ✅ COMPLETE

**Result:** mAP@50 = 27.3% (moderate domain shift — different UK barn, top-down)

**Tracking dropped:** The detection_and_localisation annotations do not include cow
identity IDs, making TrackEval-based IDF1 evaluation impossible.

**Artifacts produced:**
- `src/data/convert_cows2021.py` — Supervisely JSON → COCO (axis-aligned boxes)
- `data/processed/detection/cows2021/test/` — converted COCO dataset
- `results/detection/cows2021_eval.json`

---

### Step C — CattleEyeView Evaluation ✅ COMPLETE

**Results:**
- RF-DETR detection mAP@50 = 47.0%
- RF-DETR-Seg box mAP@50 = 53.6%
- Mean Mask IoU = 86.5% (median 91.9%, over 4,657 matched instances)

**Tracking dropped:** YOLO-format per-frame labels carry no track IDs.

**Artifacts produced:**
- `src/data/convert_cattleeyeview.py` — YOLO bbox + polygon → COCO
- `src/tools/eval_maskiou_ood.py` — RF-DETR-Seg OOD Mask IoU evaluation
- `data/processed/detection/cattleeyeview/test/` — converted COCO dataset
- `results/detection/cattleeyeview_eval.json`
- `results/segmentation/cattleeyeview_maskiou.json`

---

### Step D — Freeman Center Full Pipeline

#### D1 — Annotation format + COCO conversion ✅ COMPLETE

**Annotation format confirmed:**
- Image dataset: `data/raw/freeman-cmb-2024/CMB_dataset/CMB_dataset/{train,val,test}/`
- YOLO format: `{video_id}_frame_{N:06d}.txt` — class_id cx cy w h (9 behavior classes)
- Raw videos: `data/raw/freeman-cmb-2024/freeman-raw-videos/*.avi`
- Resolution: 1920×1080, 39,363 total frames

**9-class → 7-class label mapping** (committed to `src/data/label_utils.py::_FREEMAN_LABEL_MAP`):

| Freeman ID | Freeman class | Canonical ID | Canonical label |
|---|---|---|---|
| 0 | hay feeding | 2 | Foraging/Grazing |
| 1 | grazing | 2 | Foraging/Grazing |
| 2 | normal | 0 | Standing |
| 3 | dominance assertion | 6 | Other |
| 4 | ruminating | 4 | Ruminating |
| 5 | fear response | 6 | Other |
| 6 | grooming | 5 | Grooming |
| 7 | vocalizing | 6 | Other |
| 8 | sniffing | 6 | Other |

**Artifacts produced:**
- `src/data/convert_freeman.py` — YOLO → COCO (class-agnostic detection, behavior ID in attributes)
- `data/processed/detection/freeman/{train,valid,test}/` — 24,507 / 8,231 / 6,625 images
- `scripts/22_prepare_freeman.sh`

#### D2 — Detection evaluation ✅ COMPLETE

**Result:** mAP@50 = **73.0%** (6,625 test images, threshold 0.3)

This is the highest OOD detection result across all four Phase 8 datasets, matching
in-domain performance. Consistent with Freeman's angled/elevated camera perspective
being the same viewpoint family as CBVD-5 and CVB.

**Artifacts produced:**
- `results/detection/freeman_detection_eval.json`
- `scripts/23_eval_freeman_detection.sh`

#### D3 — Video pipeline (RF-DETR inference on raw .avi files) ⬜ PENDING

Run RF-DETR detector frame-by-frame on the raw `.avi` videos in
`data/raw/freeman-cmb-2024/freeman-raw-videos/`, producing per-video detection JSONs
in the same format as `data/processed/tracking/` (used as input to SAM2 and OC-SORT).

**Steps:**
1. Write `scripts/24_run_freeman_detection_video.sh` → extend `src/detection/infer_dataset.py`
   or write `src/detection/infer_video_dir.py` to handle a directory of `.avi` files.
   Output: `data/processed/tracking/freeman/{video_id}.json`

#### D4 — SAM2 segmentation on Freeman videos ⬜ PENDING

**Steps:**
1. Write `scripts/25_run_freeman_segmentation.sh` → `src/segmentation/segment.py`
   with `--dataset freeman` (same pattern as script 07).
   Output: `data/processed/tracking_v2/freeman/{video_id}_tracks.json`

#### D5 — OC-SORT tracking ⬜ PENDING

**Steps:**
1. Write `scripts/26_run_freeman_tracking.sh` → `src/tracking/track.py`
   with `--dataset freeman`.
   Output: `data/processed/tracking_v2/freeman/{video_id}_tracks.json` (with track IDs)

#### D6 — Tubelet generation ⬜ PENDING

**Steps:**
1. Write `scripts/27_generate_freeman_tubelets.sh` → `src/data/export_tubelets.py`
   Same parameters: 16 frames, stride 4, 224×224.
   Output: `data/processed/tubelets/freeman/`

#### D7 — VideoMAE behavior evaluation (zero-shot OOD) ⬜ PENDING

**Steps:**
1. Write `scripts/28_eval_freeman_behavior.sh` → `src/behavior/evaluate.py`
   with `videomae_combined_v1` checkpoint (frozen, no fine-tuning).
   Output: `results/behavior/predictions/freeman_eval.csv`
2. Compute behavior macro-F1 against Freeman behavior labels (mapped via `_FREEMAN_LABEL_MAP`).
3. Write `results/behavior/freeman_eval_summary.md` with per-class breakdown.

#### D8 — Analytics ⬜ PENDING

**Steps:**
1. Write `scripts/29_freeman_analytics.sh` → `src/analytics/timeline.py` + `budget.py`
   on Freeman Center predictions.
   Output: `results/analytics/freeman/` (timelines, activity budgets, behavioral deviation)

---

## 6. Files and Scripts

### Completed

| Script | Source module | Purpose | Status |
|---|---|---|---|
| *(ad hoc)* | `src/data/convert_opencows2020.py` | OpenCows2020 → COCO | ✅ |
| *(ad hoc)* | `src/data/convert_cows2021.py` | Cows2021 → COCO | ✅ |
| *(ad hoc)* | `src/data/convert_cattleeyeview.py` | CattleEyeView → COCO | ✅ |
| *(ad hoc)* | `src/tools/eval_detection_ood.py` | RF-DETR OOD detection eval | ✅ |
| *(ad hoc)* | `src/tools/eval_maskiou_ood.py` | RF-DETR-Seg Mask IoU eval | ✅ |
| `scripts/22_prepare_freeman.sh` | `src/data/convert_freeman.py` | Freeman → COCO | ✅ |
| `scripts/23_eval_freeman_detection.sh` | `src/tools/eval_detection_ood.py` | Freeman detection mAP@50 | ✅ |

### Pending (Freeman video pipeline)

| Script | Source module | Purpose |
|---|---|---|
| `scripts/24_run_freeman_detection_video.sh` | `src/detection/infer_dataset.py` (extended) | RF-DETR on .avi files |
| `scripts/25_run_freeman_segmentation.sh` | `src/segmentation/segment.py` | SAM2 on Freeman |
| `scripts/26_run_freeman_tracking.sh` | `src/tracking/track.py` | OC-SORT on Freeman |
| `scripts/27_generate_freeman_tubelets.sh` | `src/data/export_tubelets.py` | Tubelet extraction |
| `scripts/28_eval_freeman_behavior.sh` | `src/behavior/evaluate.py` | VideoMAE zero-shot eval |
| `scripts/29_freeman_analytics.sh` | `src/analytics/timeline.py` + `budget.py` | Activity budgets |

---

## 7. Result File Locations

```
results/
├── detection/
│   ├── opencows2020_eval.json           ✅ Step A
│   ├── cows2021_eval.json               ✅ Step B
│   ├── cattleeyeview_eval.json          ✅ Step C
│   └── freeman_detection_eval.json      ✅ Step D2
├── segmentation/
│   └── cattleeyeview_maskiou.json       ✅ Step C
├── tracking/
│   └── (none — tracking eval not possible from image annotations)
├── behavior/
│   ├── predictions/
│   │   └── freeman_eval.csv             ⬜ Step D7
│   └── freeman_eval_summary.md          ⬜ Step D7
└── analytics/
    └── freeman/                         ⬜ Step D8
        └── (timelines, budgets, deviation)
```

---

## 8. Frozen Constraints Carried Forward

All model checkpoints are frozen — no retraining in Phase 8:

- Detector: `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`
- Behavior: `runs/behavior/videomae_combined_v1/checkpoint_best.pt`
- SAM2: `weights/sam2.1_hiera_large.pt`

The 7-class label map (`data/label_map.json`) is frozen. Freeman Center behavior
harmonization maps onto the existing IDs via `_FREEMAN_LABEL_MAP` in `src/data/label_utils.py`.

GPU constraints remain unchanged (RTX 3060, 12 GB) for local runs. The Freeman video
pipeline (SAM2 + OC-SORT) will require significant time locally; VideoMAE inference
may need LEAP2 if tubelet count is large.

---

## 9. Resolved Questions

1. ✅ **Freeman Center annotation format** — YOLO behavior labels in `CMB_dataset/`, raw
   `.avi` videos in `freeman-raw-videos/`. Label mapping complete.

2. ✅ **OpenCows2020 annotation format** — Supervisely JSON format; handled by `convert_opencows2020.py`.

3. ✅ **Cows2021 oriented box format** — Axis-aligned boxes in Supervisely JSON; no rotation handling needed.

4. ✅ **CattleEyeView download access** — Dataset available; converted successfully.

5. ✅ **`src/detection/evaluate.py` existence** — Not needed; `src/tools/eval_detection_ood.py`
   was written and reused for all four datasets.

6. ⬜ **Freeman Center video pipeline feasibility** — Number and length of `.avi` videos
   not yet assessed. Check before starting D3.

7. ✅ **Tracking eval feasibility** — Confirmed not possible for any of the four datasets;
   none provide ground-truth track IDs.

---

## 10. Step Completion Checklist

- [x] Step A — OpenCows2020 detection eval (mAP@50 = 33.3%)
- [x] Step B — Cows2021 detection eval (mAP@50 = 27.3%; tracking dropped — no GT track IDs)
- [x] Step C — CattleEyeView detection + segmentation eval (mAP@50 = 47.0%, Mask IoU = 86.5%; tracking dropped)
- [x] Step D1 — Freeman Center annotation format clarified + COCO conversion
- [x] Step D2 — Freeman Center detection eval (mAP@50 = 73.0%)
- [ ] Step D3 — RF-DETR inference on Freeman raw .avi videos
- [ ] Step D4 — SAM2 segmentation on Freeman videos
- [ ] Step D5 — OC-SORT tracking on Freeman videos
- [ ] Step D6 — Tubelet generation for Freeman
- [ ] Step D7 — VideoMAE zero-shot behavior eval on Freeman
- [ ] Step D8 — Activity budgets + analytics for Freeman
- [ ] Update `CLAUDE.md` §7 Key Results table with Phase 8 results
- [ ] Commit all remaining result CSVs and JSON files to `results/`
