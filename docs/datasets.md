# Datasets

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

---

## 1. CBVD-5 (Cattle Behavior Video Dataset — 5 classes)

### Overview

| Property | Value |
|----------|-------|
| Source | Surveillance cameras in indoor dairy barn |
| Videos | ~687 short clips |
| Resolution | Varies (typically 1920×1080) |
| Annotation type | Bounding box + action label (AVA format, CSV) |
| Behaviors labeled | 5 (Standing, Lying, Foraging, Drinking, Ruminating) |
| Splits | Train / Valid / Test (test = val — same video IDs, no separate test release) |
| Size | ~12 GB |

### Download

Obtain from the official CBVD-5 paper and repository. After downloading, place files at:

```
data/raw/cbvd5/
├── annotations/
│   └── *.csv           ← AVA-format bounding box + action labels
└── videos/
    └── *.mp4           ← raw video clips
```

**Verify:**
```bash
ls data/raw/cbvd5/videos/ | wc -l    # ~687 files
ls data/raw/cbvd5/annotations/ | wc -l
```

### Annotation Format

Each CSV row: `video_id, timestamp_sec, x1, y1, x2, y2, action_id, track_id`

Action IDs map to behaviors:

| CBVD-5 Action ID | Behavior | Label ID |
|------------------|----------|----------|
| 1 | stand | 0 (Standing) |
| 2 | lying down | 1 (Lying) |
| 3 | foraging | 2 (Foraging) |
| 4 | drinking water | 3 (Drinking) |
| 5 | rumination | 4 (Ruminating) |

**Multi-label priority:** CBVD-5 annotations are multi-label (one bbox can have multiple action IDs
on the same frame). Resolution: highest-priority label wins.

Priority order (descending): Drinking (3) > Foraging (2) > Ruminating (4) > Lying (1) > Standing (0)

Rationale: rarer behaviors carry more information and must not be suppressed by the dominant Standing class.

### Quirks

- **Test = validation:** CBVD-5 has no separate test set. The test split of `data/processed/detection/cbvd5/test/` contains the same video IDs as the validation split. Always report results on `--split val`.
- **Annotation density:** Not all frames of a video are annotated; annotations appear at keyframe intervals.

---

## 2. CVB (Cattle Video Behavior)

### Overview

| Property | Value |
|----------|-------|
| Source | Outdoor surveillance cameras at cattle farm |
| Videos | ~502 clips |
| Resolution | Varies |
| Annotation type | Per-frame bounding box + behavior string |
| Behaviors labeled | 7 (all 7 canonical classes including Grooming and Other) |
| Splits | Train / Valid (no test split) |
| Size | ~15 GB |

### Download

Obtain from the official CVB paper and repository. After downloading:

```
data/raw/cvb/
├── annotations/
│   └── *.json          ← per-frame behavior annotations
└── raw_frames/
    └── {video_id}/
        └── frame_{idx:06d}.jpg   ← extracted frames
```

**Verify:**
```bash
ls data/raw/cvb/raw_frames/ | wc -l   # ~502 video directories
```

### Annotation Format

JSON with per-frame entries: `{frame_idx, track_id, bbox: [x,y,w,h], behavior: string}`

Behavior string mapping:

| CVB String | Label ID | Notes |
|------------|----------|-------|
| `resting-standing` | 0 (Standing) | |
| `resting-lying` | 1 (Lying) | |
| `grazing` | 2 (Foraging) | Equivalent to Foraging |
| `drinking` | 3 (Drinking) | |
| `ruminating-standing` | 4 (Ruminating) | Activity only; posture ignored |
| `ruminating-lying` | 4 (Ruminating) | |
| `grooming` | 5 (Grooming) | CVB-only class |
| `other` | 6 (Other) | CVB-only catch-all |
| `hidden` | **SKIP** | Not a behavior |
| `walking` | **SKIP** | No CBVD-5 equivalent |
| `running` | **SKIP** | < 1% of instances |
| `none` | **SKIP** | Unlabeled |

### Quirks

- **Frame presence:** Not all annotated frames have corresponding image files in `raw_frames/`. `src/data/convert_cvb.py` filters to frames where the image file actually exists.
- **CVB frame presence threshold for tubelets:** A tubelet (16 frames) must have at least 12 of 16 frames present in a track to be included. This was relaxed from 16/16 to recover more training data.
- **Label assignment:** For tubelet clips (16 frames), behavior is assigned by majority vote over all frames with valid annotations.

---

## 3. Canonical 7-Class Taxonomy

Both datasets are unified under the same 7 label IDs, defined in `data/label_map.json` and `src/data/label_utils.py`.

| Label ID | Behavior | CBVD-5 | CVB |
|----------|----------|--------|-----|
| 0 | Standing | ✓ | ✓ |
| 1 | Lying | ✓ | ✓ |
| 2 | Foraging/Grazing | ✓ | ✓ |
| 3 | Drinking | ✓ | ✓ |
| 4 | Ruminating | ✓ | ✓ |
| 5 | Grooming | — | ✓ |
| 6 | Other | — | ✓ |

**Cross-dataset evaluation** uses only IDs 0–4. CBVD-5 contains no Grooming or Other annotations.

---

## 4. Tubelet Format

After running script 09, tubelets are stored as:

```
data/processed/tubelets/
├── labels.csv
├── cbvd5/
│   └── {video_id}/
│       └── {track_id}/
│           ├── frame_000000.jpg
│           ├── frame_000001.jpg
│           └── ...   (16 frames per tubelet)
└── cvb/
    └── {video_id}/
        └── {track_id}/
            └── frame_*.jpg
```

### labels.csv Schema

| Column | Type | Description |
|--------|------|-------------|
| `dataset` | str | `cbvd5` or `cvb` |
| `video_id` | str | numeric folder name (e.g., `341`) |
| `tubelet_dir` | str | path to tubelet directory (e.g., `data/processed/tubelets/cbvd5/341/kf6_instc85ac7`) |
| `start_frame` | int | first frame index in the original video |
| `end_frame` | int | last frame index |
| `label_id` | int | canonical behavior ID (0–6) |

`track_id` is encoded as the last component of `tubelet_dir` (e.g., `kf6_instc85ac7`).

### Tubelet Parameters (Frozen)

| Parameter | Value |
|-----------|-------|
| Frames per tubelet | 16 |
| Stride (CBVD-5) | 4 frames |
| Stride (CVB) | 8 frames (50% overlap) |
| Crop size | 224×224 px |
| Crop padding | 20 px around bbox, clamped to frame boundaries |
| Total tubelets | 125,586 |

---

## 5. Phase 7 Additional Datasets

Additional cattle surveillance datasets will be integrated in Phase 7 for analytics.
When a new dataset is added:

1. Add a download section to this file (follow the CBVD-5/CVB template above)
2. Add a `scripts/0X_prepare_{dataset}.sh` script
3. Map the dataset's behavior strings to label IDs in `data/label_map.json`
4. Add a config in `configs/behavior/` if used for training

Placeholder: `data/raw/[future_datasets]/`

---

## 6. Processed Data Contracts

### Detection JSON (intermediate — not committed)

`data/processed/tracking/{dataset}/{video_id}_detections.json`
```json
{"frames": [{"frame_idx": 0, "detections": [{"bbox": [x, y, w, h], "score": 0.92}]}]}
```

### Tracking JSON (tracking_v2 — not committed, available on HuggingFace)

`data/processed/tracking_v2/{dataset}/{video_id}_tracks.json`
```json
{
  "frames": [
    {
      "frame_idx": 0,
      "tracks": [
        {
          "track_id": 1,
          "bbox": [x, y, w, h],
          "mask_rle": {"counts": "...", "size": [h, w]}
        }
      ]
    }
  ]
}
```

`mask_rle` is present only when script 07a (SAM2 segmentation) was run.
When script 08 (box-only tracking) was run instead, `mask_rle` is absent.

### Predictions CSV (committed — needed for analytics)

`results/behavior/predictions/{config}_val.csv`

| Column | Description |
|--------|-------------|
| `dataset` | `cbvd5` or `cvb` |
| `video_id` | numeric video folder name |
| `tubelet_dir` | path (track_id is last component) |
| `start_frame` | tubelet start frame in source video |
| `end_frame` | tubelet end frame |
| `label_id` | ground truth behavior label (0–6) |
| `pred_label_id` | predicted behavior label |
| `logit_0` … `logit_6` | raw model logits per class |
