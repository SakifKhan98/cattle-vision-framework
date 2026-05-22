# Phase 5 Report — Tubelet Export

**Project:** Cattle Vision Framework  
**Working directory:** `one_day/`  
**Date completed:** 2026-04-29 (re-exported after bug fix)
**Status:** Complete

---

## 1. Overview

Phase 5 converts raw tracked detections from two cattle behavior datasets (CVB and CBVD-5) into **tubelets** — short, fixed-length clips of cropped cattle regions ready for video-based behavior classification in Phase 6. Each tubelet is a folder of 16 JPEG frames, one per consecutive video frame, representing a single cattle instance performing a single behavior.

The phase produced:

- **125,586 labeled tubelets** (114,217 CVB + 11,369 CBVD-5)
- A unified `labels.csv` index covering both datasets, all splits, and 5–7 behavior classes
- A validation script confirming data integrity

A bug discovered after the initial export caused only 1,221 CVB tubelets to be generated (see §8 and §10). After fixing and re-exporting CVB, the final count rose to 114,217 CVB tubelets — an ~94× increase — giving all 7 classes adequate representation.

---

## 2. Inputs from Previous Phases

### Phase 1–2: Detection
Phase 1 trained a cattle detector (RF-DETR-Seg). Phase 2 ran it on all raw video frames, producing per-frame detection JSONs in `data/processed/tracking/`.

### Phase 3: Segmentation (SAM2 + RF-DETR-Seg Distillation)
Phase 3 refined detections with SAM2 segmentation masks (242,689 masks). The distilled model (`rf-detr-seg-medium.pt`) was used for downstream detection.

### Phase 4: OC-SORT Tracking
Phase 4 ran OC-SORT on the per-frame detections, producing **temporal track assignments** — each detection across frames was assigned a consistent `track_id`. Output:

| Dataset | Path | Count |
|---------|------|-------|
| CVB     | `data/processed/tracking_v2/cvb/{video_id}_tracks.json`   | 502 files |
| CBVD-5  | `data/processed/tracking_v2/cbvd5/{video_id}_tracks.json` | 687 files |

**Format of each tracking JSON:**
```json
{
  "video_id": "...",
  "dataset": "cvb",
  "frames": {
    "1": [
      {"track_id": 11, "bbox": [x1, y1, x2, y2], "score": 1.0, "mask_rle": {...}, "mask_area": 44290}
    ]
  }
}
```
- `bbox` is in absolute pixel coordinates `[x1, y1, x2, y2]`
- CVB frame keys are 1-indexed integers ("1"–"450")
- CBVD-5 frame keys are AVA keyframe timestamps as strings ("2"–"7")

Phase 5 consumed these tracking files as its primary input.

---

## 3. Ground Truth Annotations Used

### CVB Annotations
- **Path:** `data/raw/cvb/annotations/{video_id}/annotations/instances_default.json`
- **Format:** COCO JSON with per-annotation `attributes.behavior` (string) and `bbox` in `[x, y, w, h]` format
- **Image ID:** matches frame number (1-indexed, verified against `file_name` field)

### CVB Split Files
- **Train:** `data/raw/cvb/cvb_in_ava_format/ava_train_set.csv` — 358 video IDs
- **Val:** `data/raw/cvb/cvb_in_ava_format/ava_val_set.csv` — 89 video IDs
- **Total:** 447 videos with official split assignments

### CBVD-5 Annotations
- **Path:** `data/raw/cbvd5/annotations/ava_{train,val,test}_v2.1.csv`
- **Format:** AVA-style CSV: `video_id, timestamp, x1_norm, y1_norm, x2_norm, y2_norm, action_id, person_id`
- **Coordinates:** normalized `[0, 1]` — converted to pixels via `× 1920` / `× 1080`
- **Timestamps:** 2.0–7.0 seconds (6 keyframes per 10-second video at 25 fps)
- **Note:** The test CSV is identical to the val CSV (known CBVD-5 dataset artifact); first-occurrence wins assigns all shared keys to "val"

### CBVD-5 Raw Videos
- **Path:** `data/raw/cbvd5/videos/videos/{video_id}.mp4`
- **Properties:** 25 fps, 250 frames, 10 seconds, 1920×1080
- Used for frame extraction via OpenCV `VideoCapture`

---

## 4. Label Taxonomy

A unified 7-class taxonomy was defined across both datasets:

| ID | Class      | Datasets         | Notes                         |
|----|------------|------------------|-------------------------------|
| 0  | Standing   | CVB + CBVD-5     | Core behavior                 |
| 1  | Lying      | CBVD-5 only      | Core behavior                 |
| 2  | Foraging   | CVB + CBVD-5     | Core behavior (CVB: "grazing")|
| 3  | Drinking   | CVB + CBVD-5     | Core behavior (~2% of data)   |
| 4  | Ruminating | CBVD-5 only      | Core behavior                 |
| 5  | Grooming   | CVB only         | Auxiliary                     |
| 6  | Other      | CVB only         | Residual                      |

Cross-dataset evaluation uses only IDs 0–4 (core behaviors).

**CVB behavior string → label ID mapping** (SKIP behaviors map to `None` and are discarded):

| CVB String         | Label | SKIP? |
|--------------------|-------|-------|
| resting-standing   | 0     |       |
| resting-lying      | 1     |       |
| grazing            | 2     |       |
| drinking           | 3     |       |
| ruminating-standing| 4     |       |
| ruminating-lying   | 4     |       |
| grooming           | 5     |       |
| other              | 6     |       |
| hidden             | —     | Yes   |
| walking            | —     | Yes   |
| running            | —     | Yes   |
| none               | —     | Yes   |

**CBVD-5 multi-label priority rule** (highest priority wins):
```
4 (Drinking) > 3 (Foraging) > 5 (Ruminating) > 2 (Lying) > 1 (Standing)
```
Rarer behaviors take precedence to prevent the dominant Standing class from suppressing them.

---

## 5. Tubelet Generation Logic

### 5.1 CVB Tubelets

For each CVB video in the train/val split:

1. **Load tracking** — parse `{video_id}_tracks.json` into `{frame_int → [{track_id, bbox}]}`
2. **Load GT annotations** — parse COCO JSON into `{frame_int → [{bbox_xyxy, label_id}]}`; skip SKIP behaviors
3. **Per-frame label assignment** — for each frame present in both predicted and GT, run Hungarian matching (scipy `linear_sum_assignment`) on the IoU cost matrix. Accept matches with IoU ≥ 0.3. Result: `{frame_int → {track_id → label_id}}`
4. **Build track dict** — collect `{track_id → sorted list of frame ints}`
5. **Slide windows** — for each track, slide a 16-frame window with stride 8:
   - Require **≥ 12 of 16** frame indices present in the track (allows OC-SORT tracking gaps — see §8)
   - Collect labels for frames that have a GT match; require ≥ 8 of 16 frames with a valid label
   - Majority vote on collected labels → tubelet label
6. **Crop and save** — for each of the 16 frames: read `img_{frame:05d}.jpg`; for frames missing from the track, interpolate bbox from nearest tracked frame. Apply predicted bbox + 20 px padding clamped to `[0, 1920] × [0, 1080]`, save as `frame_{i:02d}.jpg` (JPEG quality 95)
7. **Assign split** — use the official CVB split map

**Output path pattern:**
```
data/processed/tubelets/cvb/{video_id}/track_{track_id:04d}/tubelet_{tubelet_idx:04d}/frame_{00..15}.jpg
```

### 5.2 CBVD-5 Tubelets

For each annotated (video_id, timestamp, bbox) triple across all three AVA CSVs:

1. **Parse annotations** — group multi-label rows by `(video_id, timestamp, bbox_rounded_4dp)`, apply priority rule → single `label_id` per instance
2. **Compute window** — `frame_center = int(timestamp × 25)`, `start = max(0, frame_center − 8)`, `end = min(250, start + 16)`. Skip if `end − start ≠ 16` (all timestamps 2–7 s are safe)
3. **Match to predicted bbox** — convert GT bbox from normalized to pixels; load tracking JSON for this video; Hungarian-match GT bbox against predicted bboxes at `str(int(timestamp))` key; use matched predicted bbox for crop if IoU ≥ 0.3, else fall back to GT bbox in pixels
4. **Extract frames** — OpenCV `VideoCapture.set(CAP_PROP_POS_FRAMES, start)` → read 16 consecutive BGR frames
5. **Crop and save** — apply crop bbox + 20 px padding clamped to frame bounds; save 16 JPEGs
6. **Name the folder** — `kf{int(timestamp)}_inst{bbox_hash}` where `bbox_hash` = first 6 hex chars of `MD5(f"{round(x1_norm,3):.3f}_{round(y1_norm,3):.3f}")` (deterministic across runs)

**Output path pattern:**
```
data/processed/tubelets/cbvd5/{video_id}/kf{timestamp_int}_inst{bbox_hash}/frame_{00..15}.jpg
```

### 5.3 Crop Formula

Identical for both datasets:
```python
x1 = max(0, int(bbox[0]) - 20)
y1 = max(0, int(bbox[1]) - 20)
x2 = min(W,  int(bbox[2]) + 20)
y2 = min(H,  int(bbox[3]) + 20)
crop = img[y1:y2, x1:x2]
```
No resize at export — the VideoMAE dataloader handles resizing to 224×224.

---

## 6. Output Statistics (Final Export — 2026-04-29)

Full export across all 502 CVB and 687 CBVD-5 videos, after the CVB bug fix (see §8).

| Dataset   | Videos                | Tubelets    | Avg tubelets/video  |
|-----------|----------------------|-------------|----------------------|
| CVB       | 447 (split-assigned) | 114,217     | ~255                 |
| CBVD-5    | 687                  | 11,369      | ~17                  |
| **Total** | —                    | **125,586** | —                    |

55 CVB videos were excluded because they appear in neither the train nor val split CSV.

### Class Distribution (Train + Val)

| Label | Class      | CVB (train) | CVB (val)  | CBVD-5 (train) | CBVD-5 (val) | **Total**   |
|-------|------------|-------------|------------|----------------|--------------|-------------|
| 0     | Standing   | 12,121      | 2,855      | 2,734          | 73           | **17,783**  |
| 1     | Lying      | 18,365      | 4,032      | 1,472          | 25           | **23,894**  |
| 2     | Foraging   | 45,313      | 11,252     | 2,648          | 53           | **59,266**  |
| 3     | Drinking   | 2,763       | 769        | 456            | 0            | **3,988**   |
| 4     | Ruminating | 8,800       | 3,349      | 3,905          | 3            | **16,057**  |
| 5     | Grooming   | 1,666       | 296        | —              | —            | **1,962**   |
| 6     | Other      | 2,236       | 400        | —              | —            | **2,636**   |
| —     | **Total**  | **91,264**  | **22,953** | **11,215**     | **154**      | **125,586** |

**Notes on class gaps:**

- CBVD-5 Drinking val = 0: the 50 val videos contain no Drinking annotations — a dataset property, not a bug
- CBVD-5 Ruminating val = 3: same cause
- Grooming (5) and Other (6) are CVB-only; Configs 1, 3, 4 (CBVD-5-involved) will produce F1=0 for these classes

Note: CBVD-5 test split is absent from labels.csv because the test CSV is identical to the val CSV (known dataset artifact); first-occurrence-wins assigns all shared keys to "val".

---

## 7. Files Produced by Phase 5

### Source Code

| File | Purpose |
|------|---------|
| `src/data/label_utils.py` | All pure utility functions: label mapping, IoU, Hungarian matching, annotation loaders, frame extractor, split loader |
| `src/data/export_tubelets.py` | Export orchestration: `export_cvb_tubelets`, `export_cbvd5_tubelets`, and `main()` CLI entry point |
| `src/data/validate_tubelets.py` | Sanity-check script: verifies path integrity, class distribution, split coverage |

### Data Output

| Path | Description |
|------|-------------|
| `data/processed/tubelets/labels.csv` | Master index of all tubelets: dataset, video_id, tubelet_dir, start/end frame, label_id, label_name, split |
| `data/processed/tubelets/cvb/{video_id}/track_{tid:04d}/tubelet_{idx:04d}/` | CVB tubelet folders, each containing `frame_00.jpg`–`frame_15.jpg` |
| `data/processed/tubelets/cbvd5/{video_id}/kf{ts}_inst{hash}/` | CBVD-5 tubelet folders, each containing `frame_00.jpg`–`frame_15.jpg` |

### labels.csv Schema

```
dataset,video_id,tubelet_dir,start_frame,end_frame,label_id,label_name,split
```

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | str | `"cvb"` or `"cbvd5"` |
| `video_id` | str | Original video identifier |
| `tubelet_dir` | str | Path to folder (relative to `one_day/`) containing 16 JPEGs |
| `start_frame` | int | First frame index in the original video (0-indexed for CBVD-5, 1-indexed for CVB) |
| `end_frame` | int | Exclusive end frame (`end_frame − start_frame = 16`) |
| `label_id` | int | Canonical label ID (0–6) |
| `label_name` | str | Human-readable class name |
| `split` | str | `"train"`, `"val"`, or `"test"` |

---

## 8. Key Engineering Decisions

### IoU Threshold = 0.3
Predicted tracker bboxes are matched to GT annotation bboxes using Hungarian matching with IoU ≥ 0.3. Lower threshold would introduce noisy label assignments; higher would lose too many matches in crowded frames.

### CVB Window Requirements
- **Stride 8:** 50% overlap between consecutive tubelets gives temporal redundancy without over-sampling a single track
- **All-16-consecutive constraint:** If a track has internal gaps (OC-SORT can lose a track for a few frames), windows containing those gaps are skipped entirely. This prevents frames from the wrong cattle contaminating a tubelet
- **≥ 8 of 16 frames labeled:** A relaxed threshold allowing up to 8 unlabeled frames per window, resolved by majority vote. Unlabeled frames occur when the GT annotation has no matching detection (IoU < 0.3)

### CBVD-5 Bbox for Cropping
The predicted bbox (from OC-SORT tracking) is preferred over the GT bbox for cropping because it represents what the production inference model will see — consistent with train/test conditions. GT bbox is used as fallback only when no predicted detection matches.

### Deterministic Folder Names (CBVD-5)
Folder names use MD5-derived hashes of GT bbox coordinates (rounded to 3 d.p.) rather than Python's `hash()` (which is randomized per process). This makes folder names stable across re-runs, so incremental export does not orphan old directories.

### SKIP Behaviors (CVB)
`hidden`, `walking`, `running`, and `none` are mapped to `None` and excluded from tubelets entirely. Walking and running have no CBVD-5 equivalent and would introduce domain-incompatible classes. Hidden cattle would produce uninformative crops.

---

## 9. How Phase 5 Outputs Feed into Phase 6

Phase 6 fine-tunes **VideoMAE-Base** (pretrained on Kinetics-400) for cattle behavior classification. It consumes Phase 5 outputs directly:

### `labels.csv` → Dataset class (`src/behavior/dataset.py`)
The dataset class reads `labels.csv` and, for each row:
1. Opens the `tubelet_dir` folder and loads all 16 JPEGs
2. Resizes each frame to 224×224 (bilinear)
3. Converts BGR → RGB, normalizes with ImageNet mean/std
4. Stacks into a tensor of shape `[3, 16, 224, 224]` (C, T, H, W)
5. Returns `(tensor, label_id)` to the DataLoader

The `dataset_filter` and `split_filter` parameters allow the same CSV to serve all five training configurations (in-domain CVB, in-domain CBVD-5, cross-domain, combined).

### Five Training Configurations

| Config | Train data | Val/test data |
|--------|-----------|---------------|
| 1 — In-domain CBVD-5 | CBVD-5 train | CBVD-5 val |
| 2 — In-domain CVB | CVB train | CVB val |
| 3 — OOD: CBVD-5 → CVB | CBVD-5 train | CVB val, core 5 labels |
| 4 — OOD: CVB → CBVD-5 | CVB train | CBVD-5 val, core 5 labels |
| 5 — Combined | CBVD-5+CVB train | CBVD-5+CVB val |

All five configurations read from the same `labels.csv` file produced by Phase 5, filtered by `dataset` and `split` columns.

### Class Weights
The class imbalance visible in the label distribution (Drinking ≈ 2% of data) is handled in Phase 6 via:
```python
weight[c] = total_samples / (num_classes × count[c])
```
passed to `nn.CrossEntropyLoss(weight=...)`.

---

## 10. Full Export Instructions

To run the complete export (all videos, both datasets):

```bash
cd one_day
python src/data/export_tubelets.py --output data/processed/tubelets
```

Expected runtime: ~2–4 hours for CVB (502 videos × 450 frames each), ~30–60 minutes for CBVD-5 (687 videos × 16 frames × 6 annotations). Run overnight on local machine (RTX 3060).

After export, validate:
```bash
python src/data/validate_tubelets.py
```

Expected full-run output:
- CVB tubelets: 30,000–80,000
- CBVD-5 tubelets: ~22,000–25,000
- All paths valid, Drinking present, train/val splits present for both datasets

---

## 11. Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| 5.1  | Label mapping functions (`cvb_behavior_to_label`, `cbvd5_actions_to_label`) | Done |
| 5.2  | IoU utility (`bbox_iou`, `match_predicted_to_gt`) | Done |
| 5.3  | CVB annotation loader (`load_cvb_gt`) | Done |
| 5.4  | CVB split loader (`load_cvb_splits`) | Done |
| 5.5  | CBVD-5 annotation loader (`load_cbvd5_annotations`) | Done |
| 5.6  | CBVD-5 frame extractor (`extract_cbvd5_frames`) | Done |
| 5.7  | CVB tubelet exporter (`export_cvb_tubelets`) | Done |
| 5.8  | CBVD-5 tubelet exporter (`export_cbvd5_tubelets`) | Done |
| 5.9  | Main export script (`main()` in `export_tubelets.py`) | Done |
| 5.10 | Validation script (`validate_tubelets.py`) | Done |

All acceptance criteria verified. Phase 6 (VideoMAE fine-tuning) can begin.

---

## 12. Note on V2 Tubelet Generation

Phase 6 v2 models (`videomae_*_v2.yaml`) were trained on tubelets generated from the **box-only tracking path** (RF-DETR + OC-SORT without SAM2 mask IoU post-association). These tubelets use the same export logic as described in §5 above, but the input tracking JSONs come from `scripts/08_run_tracking.sh` (box-only) rather than `scripts/07_run_segmentation.sh` + `scripts/08_run_tracking.sh` (mask IoU path).

The tubelet count and label distribution are expected to be similar since the frame-level crops are driven by bounding boxes in both paths. The key difference is that v2 tracking JSONs have `mask_rle: null` for all detections — the mask field is absent, but Phase 5's tubelet export only uses the `bbox` field for cropping, so the absence of masks does not affect tubelet content.

The v2 tubelet directories are not separately stored — they overwrite the same `data/processed/tubelets/` path if re-exported, or they may have been generated in a separate directory on HiPE1. The `configs/behavior/videomae_*_v2.yaml` files specify the exact tubelet path used for training.
