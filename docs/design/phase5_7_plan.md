# Phase 5–7 Implementation Plan

## Cattle Vision Framework — Tubelet Export → VideoMAE → Analytics

**Date:** 2026-04-27 (last updated 2026-04-29)
**Working directory:** `one_day/` (all paths below are relative to this root)
**Status:** Phases 1–6.7 complete. CVB rsync done (33GB on HiPE1, 5.6G free). Config 5 (combined) training NOW RUNNING on HiPE1 GPU 0. Configs 1+2 queued after Config 5 finishes. Configs 3+4 queued after 1+2.

---

## 1. Current State Summary

| Phase                         | Output                                                  | Status                                      |
| ----------------------------- | ------------------------------------------------------- | ------------------------------------------- |
| 1 — Detection training        | `runs/detection/` checkpoints                           | Done                                        |
| 2 — Detection inference       | `data/processed/tracking/` detection JSONs              | Done                                        |
| 3 — SAM2 segmentation         | 242,689 masks generated                                 | Done                                        |
| 3b — RF-DETR-Seg distillation | `rf-detr-seg-medium.pt` (Config B, ep59)                | Done                                        |
| 4 — OC-SORT tracking          | `data/processed/tracking_v2/{cbvd5,cvb}/*_tracks.json` | Done                                        |
| **5 — Tubelet export**        | `data/processed/tubelets/` + `labels.csv`               | **Done (re-exported 2026-04-29)**           |
| **6 — VideoMAE training**     | `runs/behavior/` checkpoints                            | **In progress — Config 5 running on GPU 0** |
| **7 — Analytics**             | behavior timelines, activity budgets                    | Not started                                 |

---

## 2. Frozen Decisions

- **Tubelet length:** 16 frames
- **Tubelet stride (CVB):** 8 frames (50% overlap)
- **Crop type:** raw bbox + 20 px padding, clamped to frame boundaries
- **Label assignment (CVB):** per-frame IoU matching ≥ 0.3 + majority vote over tubelet
- **Label assignment (CBVD-5):** extract 16-frame window from raw .mp4 centered on annotated keyframe; apply priority-based single label
- **Training target:** HiPE1 Tesla V100 16 GB × 2
- **Primary metric:** Macro-F1 (robust to class imbalance)
- **CVB frame presence threshold:** 12 of 16 frames must be present in track (relaxed from original 16/16 — see §5.4)
- **No test split:** CBVD5 "test" video IDs are identical to val video IDs; CVB has no test split. Use `--split val` everywhere in evaluation.

---

## 3. Label Map (Frozen)

### 3.1 Canonical 7-Class Taxonomy

| ID  | Class      | Type                  |
| --- | ---------- | --------------------- |
| 0   | Standing   | core                  |
| 1   | Lying      | core                  |
| 2   | Foraging   | core                  |
| 3   | Drinking   | core                  |
| 4   | Ruminating | core                  |
| 5   | Grooming   | auxiliary (CVB-only)  |
| 6   | Other      | residual (CVB-only)   |

Cross-dataset (OOD) evaluation uses only IDs 0–4. CBVD5 has no Grooming or Other annotations.

### 3.2 CVB Behavior String → Label ID

| CVB String            | Label ID | Notes                                                   |
| --------------------- | -------- | ------------------------------------------------------- |
| `resting-standing`    | 0        | Standing                                                |
| `resting-lying`       | 1        | Lying                                                   |
| `grazing`             | 2        | Foraging                                                |
| `drinking`            | 3        | Drinking                                                |
| `ruminating-standing` | 4        | Ruminating (activity only, posture captured separately) |
| `ruminating-lying`    | 4        | Ruminating                                              |
| `grooming`            | 5        | Grooming                                                |
| `other`               | 6        | Other                                                   |
| `hidden`              | **SKIP** | Not a behavior                                          |
| `walking`             | **SKIP** | No CBVD-5 equivalent                                    |
| `running`             | **SKIP** | < 1% of instances                                       |
| `none`                | **SKIP** | Unlabeled                                               |

### 3.3 CBVD-5 Action ID → Label ID

| CBVD-5 ID | CBVD-5 Name    | Label ID |
| --------- | -------------- | -------- |
| 1         | stand          | 0        |
| 2         | lying down     | 1        |
| 3         | foraging       | 2        |
| 4         | drinking water | 3        |
| 5         | rumination     | 4        |

### 3.4 CBVD-5 Multi-Label Priority Rule

CBVD-5 annotations are multi-label. When a bbox has multiple action IDs, pick one using this priority (highest = wins):

```
4 (Drinking) > 3 (Foraging) > 5 (Ruminating) > 2 (Lying) > 1 (Standing)
```

**Rationale:** Rarer behaviors carry more information and should not be suppressed by the dominant Standing class.

**Common combos observed:** (1,3) Standing+Foraging → 3 Foraging; (2,5) Lying+Ruminating → 4 Ruminating; (1,5) Standing+Ruminating → 4 Ruminating; (1,4) Standing+Drinking → 3 Drinking.

---

## 4. Data Contracts

### 4.1 Inputs

#### CVB Tracking Output

- **Path:** `data/processed/tracking_v2/cvb/{video_id}_tracks.json`
- **Count:** 502 files
- **Format:**
  ```json
  {
    "video_id": "...",
    "dataset": "cvb",
    "frames": {
      "1": [
        {"track_id": 11, "bbox": [x1, y1, x2, y2], "score": 1.0, "mask_rle": {...}, "mask_area": 44290}
      ]
    },
    "stats": {...}
  }
  ```

  - `bbox` is [x1, y1, x2, y2] in **pixel coordinates** (absolute, not normalized)
  - Frame keys are 1-indexed strings ("1" through "450")
  - `mask_rle` is COCO RLE format

#### CVB Ground Truth Annotations

- **Path:** `data/raw/cvb/annotations/{video_id}/annotations/instances_default.json`
- **Format:** COCO JSON with per-annotation `attributes.behavior` (string) and `attributes.track_id` (int)
- **Bbox format:** COCO [x, y, w, h] in pixel coordinates
- **Frame images:** 1-indexed via `image_id` matching `images` array

#### CVB Raw Frames

- **Path:** `data/raw/cvb/raw_frames/{video_id}/img_{frame:05d}.jpg`
- **Frame numbering:** 1-indexed, `img_00001.jpg` through `img_00450.jpg`
- **Resolution:** 1920×1080

#### CVB Official Split

- **Train video IDs:** parsed from `data/raw/cvb/cvb_in_ava_format/ava_train_set.csv` (column 0, splitting on `,`)
- **Val video IDs:** parsed from `data/raw/cvb/cvb_in_ava_format/ava_val_set.csv`
- **Counts:** 358 train, 89 val (total 447 — some tracking videos may not be in either split)
- **No test split** — CVB is a 2-way split only.

---

#### CBVD-5 Tracking Output

- **Path:** `data/processed/tracking_v2/cbvd5/{video_id}_tracks.json`
- **Count:** 688 files (includes 1 empty, 687 with detections)
- **Same format as CVB** — frame keys are the 6 keyframe timestamps as strings ("2", "3", "4", "5", "6", "7")
- **Note:** frame keys here are AVA timestamps (seconds), not actual frame indices

#### CBVD-5 Ground Truth Annotations

- **Path:** `data/raw/cbvd5/annotations/ava_{split}_v2.1.csv` where split ∈ {train, val, test}
- **Format:** `video_id, frame_timestamp, x1_norm, y1_norm, x2_norm, y2_norm, action_id, person_id`
  - Coords are **normalized** (0–1); multiply by 1920 / 1080 for pixels
  - `frame_timestamp` is seconds (2.0, 3.0, ..., 7.0)
  - `person_id` is always "1" — uniquely identify instances by (video_id, timestamp, rounded_bbox)
- **Splits:** 492 train videos / 50 val videos / 50 test videos
- **IMPORTANT:** The 50 "test" video IDs are identical to the 50 val video IDs. `load_cbvd5_annotations()` reads CSVs in train→val→test order; test keys already exist in group_meta as val, so test annotations are silently absorbed into val. There is effectively no separate test split — labels.csv has only `train` and `val`.

#### CBVD-5 Raw Videos

- **Path:** `data/raw/cbvd5/videos/videos/{video_id}.mp4`
- **Properties:** 25 fps, 250 frames, 10s, 1920×1080
- **Count:** 687 videos — all annotated video IDs are covered

---

### 4.2 Outputs (Phase 5) — COMPLETE

#### Tubelet Frame Crops

```
data/processed/tubelets/
├── cbvd5/
│   └── {video_id}/
│       └── kf{timestamp_int}_inst{bbox_hash}/
│           └── frame_{00..15}.jpg       ← 16 frames as JPEG
└── cvb/
    └── {video_id}/
        └── track_{track_id:04d}/
            └── tubelet_{tubelet_idx:04d}/
                └── frame_{00..15}.jpg
```

- Saved as JPEG quality 95
- Crop region: bbox + 20px padding on each side, clamped to [0, W] and [0, H]
- No resize at export — resize happens in the VideoMAE dataloader
- **Total disk size:** ~33GB (6GB CBVD5 + 27GB CVB)

#### labels.csv — FINAL COUNTS

Path: `data/processed/tubelets/labels.csv` — **125,586 rows**

| Dataset | Class          | Train  | Val    |
| ------- | -------------- | ------ | ------ |
| cbvd5   | Standing (0)   | 2,734  | 73     |
| cbvd5   | Lying (1)      | 1,472  | 25     |
| cbvd5   | Foraging (2)   | 2,648  | 53     |
| cbvd5   | Drinking (3)   | 456    | 0      |
| cbvd5   | Ruminating (4) | 3,905  | 3      |
| cvb     | Standing (0)   | 12,121 | 2,855  |
| cvb     | Lying (1)      | 18,365 | 4,032  |
| cvb     | Foraging (2)   | 45,313 | 11,252 |
| cvb     | Drinking (3)   | 2,763  | 769    |
| cvb     | Ruminating (4) | 8,800  | 3,349  |
| cvb     | Grooming (5)   | 1,666  | 296    |
| cvb     | Other (6)      | 2,236  | 400    |
| **ALL** | **TOTAL**      | **102,479** | **23,107** |

**Notes on data limitations:**
- CBVD5 Drinking val = 0 (the 50 val videos have no Drinking annotations — dataset property, not a bug)
- CBVD5 Ruminating val = 3 (same cause — very few in the specific 50 val videos)
- Grooming and Other exist only in CVB; Configs 1/3/4 (CBVD5-involved) will have 0 samples for these classes

---

## 5. Phase 5 — Tubelet Export (COMPLETE)

### 5.1 CVB Tubelet Generation Logic

For each of the 502 CVB videos (447 in split_map, 55 excluded):

**Step 1 — Load tracking data**
Open `tracking_v2/cvb/{video_id}_tracks.json`. Build dict `predicted = {frame_int → [{track_id, bbox_xyxy}]}`. Frames 1–450, bbox in pixel coords.

**Step 2 — Load GT annotations**
Open `annotations/{video_id}/annotations/instances_default.json`. Build `gt = {frame_int → [{bbox_xyxy, label_id}]}`. Convert COCO [x,y,w,h] to [x1,y1,x2,y2]. Map behavior strings using §3.2. Skip SKIP behaviors.

**Step 3 — Per-frame label lookup**
For every frame present in both `predicted` and `gt`, run Hungarian matching (scipy `linear_sum_assignment`) on the IoU cost matrix. Accept match only if IoU ≥ 0.3. Result: `frame_labels = {frame_int → {track_id → label_id}}`.

**Step 4 — Build track dict**
`tracks = {track_id → sorted list of frame ints where this track appears}`

**Step 5 — Slide tubelet windows**
For each track, for each window `[start, start+16)` with stride 8:
- **Require at least 12 of 16 frames present in track** (allows OC-SORT tracking gaps — see §5.4)
- Collect label_ids for frames with a label match; require ≥ 8 of 16 frames labeled
- Majority vote → tubelet_label

**Step 6 — Crop and save frames**
- Load raw frames `img_{frame_int:05d}.jpg` for all 16 frames in window
- For missing track frames: use nearest-neighbor bbox interpolation from `track_bbox`
- Compute crop: `[x1-20, y1-20, x2+20, y2+20]` clamped to [0, 1920] × [0, 1080]
- Save 16 crops as `frame_{i:02d}.jpg` in `data/processed/tubelets/cvb/{video_id}/track_{track_id:04d}/tubelet_{tubelet_idx:04d}/`

**Step 7 — Assign split**
Check `video_id` against `cvb_in_ava_format/ava_train_set.csv` and `ava_val_set.csv`. Videos not in either are excluded.

---

### 5.2 CBVD-5 Tubelet Generation Logic

For each annotated (video_id, timestamp) pair:

**Step 1 — Parse annotations**
Read all three `ava_{split}_v2.1.csv` files. Group by (video_id, timestamp, bbox_rounded). Apply priority rule (§3.4) → single label_id.

**Step 2 — Compute frame index**
`frame_center = int(float(timestamp) * 25)`. Window: `start = frame_center - 8`, `end = frame_center + 8`. Clamp to [0, 250]. Require `end - start == 16`.

**Step 3 — Match to predicted track for crop bbox**
Load tracking file. Convert GT bbox normalized → pixels. Hungarian match to predicted bboxes. If IoU ≥ 0.3: use predicted bbox. Else: fallback to GT bbox pixels.

**Step 4 — Extract 16 frames and save**
Open .mp4 with OpenCV, seek to `start`, read 16 frames. Apply crop + padding + clamp. Save as `frame_{i:02d}.jpg`.

---

### 5.3 Key Script

```bash
# Full export (original)
python3 src/data/export_tubelets.py \
  --output data/processed/tubelets

# Re-export CVB only (merge with existing CBVD5 rows in labels.csv)
python3 -u src/data/export_tubelets.py \
  --output data/processed/tubelets \
  --cvb_only
```

---

### 5.4 Bug Found and Fixed (2026-04-29)

**Original bug:** `export_cvb_tubelets()` required ALL 16 consecutive frames present in OC-SORT track. OC-SORT tracks have gaps (missed detections). Result: only 1,221 CVB tubelets from 447 videos (2.7/video). Grooming and Other classes had 0 tubelets. Ruminating val had only 3 samples.

**Fix applied in `src/data/export_tubelets.py`:**
- Changed frame presence check from `all(f in track_frame_set for f in window)` → `sum(1 for f in window if f in track_frame_set) >= 12`
- Added nearest-neighbor bbox interpolation for missing frames within the window
- Added `--cvb_only` flag to re-export just CVB and merge with existing CBVD5 rows

**Result after fix:** 114,217 CVB tubelets (average 255/video). All 7 classes present.

---

## 6. Phase 6 — VideoMAE Behavior Classification

### 6.1 Model

**VideoMAE-Base pretrained on Kinetics-400.**

- HuggingFace model ID: `MCG-NJU/videomae-base-finetuned-kinetics`
- Input: 16 frames × 224×224 × 3 (RGB, normalized with ImageNet mean/std)
- Classification head replaced: `model.classifier = nn.Linear(768, 7)`

### 6.2 Dataset Class (`src/behavior/dataset.py`) — COMPLETE

Reads `labels.csv`. For each row:

- Load 16 JPEG frames from `tubelet_dir`.
- Resize each to 224×224 (bilinear interpolation).
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
- Stack to tensor of shape [3, 16, 224, 224] (channel-first, time second).
- Return `(tensor, label_id)`.

**Filtering parameters (passed via config):**

- `dataset_filter`: `null` (use both), `cbvd5`, or `cvb`
- `split_filter`: `train` or `val` (no `test` split exists)
- `label_subset`: `null` (all 7 classes) or `[0,1,2,3,4]` (core-only for cross-dataset eval)

**Class weights for training:**
Compute `weight[c] = total_samples / (num_classes * count[c])` from training set. Pass to `nn.CrossEntropyLoss(weight=...)`.

### 6.3 Training Config Format (`configs/behavior/`) — COMPLETE

Each YAML config controls one training run:

```yaml
# configs/behavior/videomae_combined.yaml
experiment_name: videomae_combined_v1
model_name: MCG-NJU/videomae-base-finetuned-kinetics
num_classes: 7
labels_csv: data/processed/tubelets/labels.csv

train:
  dataset_filter: null # null = both cbvd5 + cvb
  split_filter: train
  label_subset: null # null = all 7 classes

val:
  dataset_filter: null
  split_filter: val
  label_subset: null

batch_size: 8
grad_accum_steps: 4 # effective batch = 32
num_epochs: 30
lr: 5.0e-5
lr_head: 1.0e-3 # separate LR for classification head
weight_decay: 0.05
warmup_epochs: 3
early_stopping_patience: 8
use_class_weights: true

output_dir: runs/behavior/videomae_combined_v1
```

### 6.4 Five Training Configurations

| Config                | Train set        | Val set                  | Config YAML                  | Status        |
| --------------------- | ---------------- | ------------------------ | ---------------------------- | ------------- |
| 1 — In-domain CBVD-5  | CBVD-5 train     | CBVD-5 val               | `videomae_cbvd5.yaml`        | Not started   |
| 2 — In-domain CVB     | CVB train        | CVB val                  | `videomae_cvb.yaml`          | Not started   |
| 3 — OOD: CBVD-5 → CVB | CBVD-5 train     | CVB val, core 5 only     | `videomae_cbvd5_to_cvb.yaml` | Not started   |
| 4 — OOD: CVB → CBVD-5 | CVB train        | CBVD-5 val, core 5 only  | `videomae_cvb_to_cbvd5.yaml` | Not started   |
| 5 — Combined          | CBVD-5+CVB train | CBVD-5+CVB val           | `videomae_combined.yaml`     | **Must rerun** |

**Run priority:** Config 5 first (combined). Then Configs 1+2 in parallel. Then 3+4.

**Note on Config 5 first run (invalid):** Ran on old labels.csv (12,590 tubelets, CVB broken). Early stopped epoch 12, best val_macro_f1=0.3648. Grooming/Other F1=0.000 (classes absent). Must retrain with fixed 125,586-row labels.csv. Old checkpoint at `runs/behavior/videomae_combined_v1/` should be overwritten.

### 6.5 Training Script (`src/behavior/train.py`) — COMPLETE

```bash
# Inside Docker on HiPE1:
train.py --config configs/behavior/videomae_combined.yaml
```

Outputs per epoch:
- `runs/behavior/{experiment_name}/checkpoint_best.pt` — best macro-F1
- `runs/behavior/{experiment_name}/checkpoint_last.pt` — last epoch
- `runs/behavior/{experiment_name}/log.csv` — epoch, train_loss, val_loss, val_macro_f1

### 6.6 Evaluation Script (`src/behavior/evaluate.py`) — COMPLETE

```bash
python src/behavior/evaluate.py \
  --config configs/behavior/videomae_combined.yaml \
  --checkpoint runs/behavior/videomae_combined_v1/checkpoint_best.pt \
  --split val
```

**Always use `--split val`** — no test split exists in either dataset.

Outputs:
- Per-class F1, precision, recall
- Macro-F1 (primary)
- Confusion matrix as PNG in `results/behavior/confusion_matrices/`
- Summary CSV in `results/behavior/f1_per_class.csv`

### 6.7 HiPE1 Deployment Notes — COMPLETE

- Docker image: `cattle-videomae:v1` — already loaded on HiPE1.
- HiPE1 working directory: `~/cattle_behavior/`
- Scripts on HiPE1: `~/cattle_behavior/src/behavior/{train.py, dataset.py, evaluate.py}`
- Configs on HiPE1: `~/cattle_behavior/configs/behavior/`
- Tubelets on HiPE1: `~/cattle_behavior/data/processed/tubelets/` (CBVD5 complete; CVB rsync in progress as of 2026-04-29)
- labels.csv on HiPE1: already updated to 125,586 rows
- Two V100s available. Config 5 uses `--gpus all` but train.py has no DataParallel/DDP — only GPU 0 is used. Configs 1+2 run after Config 5 finishes, pinned to `--gpus '"device=0"'` / `--gpus '"device=1"'`. Configs 3+4 follow same pattern after 1+2 finish.
- **HiPE1 disk:** 98G total, 88G used, 5.6G free after rsync (tight but sufficient — checkpoints ~330MB each × 5 configs = ~1.6GB total)

**Monitoring from local machine:**
```bash
ssh hipe1 "tail -30 ~/cattle_behavior/logs/combined_v2.log"
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'"
```

**Updating scripts without rebuilding image:**
```bash
rsync src/behavior/train.py hipe1:~/cattle_behavior/src/behavior/
# kill container in tmux, re-run docker run command
```

**Retrieving results:**
```bash
rsync -avz hipe1:~/cattle_behavior/runs/behavior/ runs/behavior/
```

### 6.8 Run Config 5 (Combined) — RUNNING (started 2026-04-29)

**First run (2026-04-28): INVALID** — ran on broken labels.csv (12,590 tubelets). Result discarded.

**Second run: NOW RUNNING on HiPE1 GPU 0.** tmux session: `combined`. Log: `~/cattle_behavior/logs/combined_v2.log`. Train: 102,479 / Val: 23,107.

**Fix applied to docker command** — original mount of individual files caused `ModuleNotFoundError: No module named 'behavior'`. Correct command:

```bash
ssh hipe1
tmux new-session -s combined
cd ~/cattle_behavior
mkdir -p logs

docker run --rm \
    --gpus all \
    --shm-size=16g \
    --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    cattle-videomae:v1 \
    /workspace/behavior/train.py --config configs/behavior/videomae_combined.yaml \
    2>&1 | tee ~/cattle_behavior/logs/combined_v2.log
```

**Key changes from original:** mount `src/behavior/` as `/workspace/behavior/` (whole package, not individual files); `--entrypoint python3`; call `/workspace/behavior/train.py` directly.

Done when: `checkpoint_best.pt` exists and val_macro_f1 > 0.40 (bad-data run hit 0.3648 with 5 effective classes; expect higher with fixed data).

---

### 6.9 Run Configs 1–4

**Sequencing:** Wait for Config 5 to finish, then launch 1+2 in parallel (each locks one GPU). Wait for 1+2 to finish, then launch 3+4 in parallel.

**Config 1 — In-domain CBVD-5 (GPU 0):**
```bash
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    cattle-videomae:v1 \
    /workspace/behavior/train.py --config configs/behavior/videomae_cbvd5.yaml \
    2>&1 | tee ~/cattle_behavior/logs/cbvd5.log
```

**Config 2 — In-domain CVB (GPU 1):**
```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=16g \
    --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    cattle-videomae:v1 \
    /workspace/behavior/train.py --config configs/behavior/videomae_cvb.yaml \
    2>&1 | tee ~/cattle_behavior/logs/cvb.log
```

**Config 3 — OOD CBVD-5 → CVB (GPU 0), after Configs 1+2 done:**
```bash
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    cattle-videomae:v1 \
    /workspace/behavior/train.py --config configs/behavior/videomae_cbvd5_to_cvb.yaml \
    2>&1 | tee ~/cattle_behavior/logs/cbvd5_to_cvb.log
```

**Config 4 — OOD CVB → CBVD-5 (GPU 1), after Configs 1+2 done:**
```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=16g \
    --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    cattle-videomae:v1 \
    /workspace/behavior/train.py --config configs/behavior/videomae_cvb_to_cbvd5.yaml \
    2>&1 | tee ~/cattle_behavior/logs/cvb_to_cbvd5.log
```

**Note on Config 1 (CBVD5 in-domain):** num_classes=7 but Grooming (5) and Other (6) have 0 CBVD5 training samples. F1 for those classes will be 0. Thesis should note this as a dataset limitation. Report macro-F1 over the 5 core classes for fair comparison.

Done when: 4 `checkpoint_best.pt` files exist in respective run dirs.

---

### 6.10 Collect All Results

Run `evaluate.py` for all 5 configs using `--split val`. Compile into `results/behavior/summary_table.csv`:

```csv
config,train_domain,val_domain,macro_f1,f1_standing,f1_lying,f1_foraging,f1_drinking,f1_ruminating,f1_grooming,f1_other
```

**Docker run pattern for evaluation:**
```bash
docker run --rm --gpus all \
    --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    cattle-videomae:v1 \
    /workspace/behavior/evaluate.py \
    --config configs/behavior/videomae_combined.yaml \
    --checkpoint runs/behavior/videomae_combined_v1/checkpoint_best.pt \
    --split val \
    2>&1 | tee ~/cattle_behavior/logs/eval_combined.log
```

Repeat for each config. Done when `summary_table.csv` has 5 rows.

---

## 7. Phase 7 — Analytics

### 7.1 Behavior Timeline (`src/analytics/timeline.py`)

Input: per-tubelet predictions from `evaluate.py` (video_id, track_id, start_frame, end_frame, pred_label_id).

Processing:
1. Resolve overlapping tubelets (stride=8, length=16): weighted average of logit outputs before argmax.
2. Apply median filter (window=5 frames) to smooth noise.
3. Merge consecutive frames with same label into segments.

Output per video per track: `results/analytics/timelines/{video_id}_{track_id}.csv`

```csv
track_id,label_id,label_name,start_frame,end_frame,start_sec,end_sec,duration_sec
```

### 7.2 Activity Budget and Behavioral Deviation (`src/analytics/budget.py`)

Input: timeline CSVs.

Outputs:
1. `results/analytics/activity_budget.csv` — per track per video: `% time in each behavior`
2. `results/analytics/transition_matrix.csv` — behavior-to-behavior transition probabilities
3. `results/analytics/behavior_deviation.csv` — per track: deviation from dataset-specific baseline

Per the approved thesis proposal (§4.6.3), the third output is **behavioral deviation analysis**,
not clinical welfare flags. Implementation:
- Compute dataset-level median % for each behavior (e.g., median % lying across all CVB val tracks)
- Per track: report the absolute deviation from that median for each behavior
- Flag tracks whose deviation exceeds 1.5× the interquartile range as notable outliers
- No hard clinical thresholds (< 8 hrs/day etc.) — those are veterinary standards not supported by
  short-clip video datasets

Schema: `dataset, video_id, track_id, behavior, pct_time, baseline_median, deviation, is_outlier`

---

## 8. Task Status

| Task | Description | Status |
| ---- | ----------- | ------ |
| 5.1–5.6 | Label utils, IoU, loaders | Done |
| 5.7–5.9 | CVB + CBVD5 export, main script | Done (re-exported 2026-04-29) |
| 5.10 | Validation script | Done |
| 6.1 | TubeletDataset | Done |
| 6.2 | train.py | Done |
| 6.3 | evaluate.py | Done |
| 6.4 | 5 YAML configs | Done |
| 6.5 | Dockerfile.videomae | Done |
| 6.6 | Local sanity check | Done |
| 6.7 | HiPE1 deployment | Done |
| 6.8 | Config 5 Combined | **Running on HiPE1 GPU 0 (started 2026-04-29)** |
| 6.9 | Configs 1–4 | Not started — queued after Config 5 finishes |
| 6.10 | Evaluate all 5 | Not started |
| 7.1 | Timeline builder | Not started |
| 7.2 | Activity budget + behavioral deviation | Not started |
| 7.3 | ~~Welfare flags~~ → behavioral deviation | Renamed — see §7.2 |

---

## 9. File Checklist

```
src/data/label_utils.py              ✓ Done
src/data/export_tubelets.py          ✓ Done (fixed 2026-04-29)
src/data/validate_tubelets.py        ✓ Done
src/behavior/dataset.py              ✓ Done
src/behavior/train.py                ✓ Done
src/behavior/evaluate.py             ✓ Done
Dockerfile.videomae                  ✓ Done
configs/behavior/videomae_sanity.yaml ✓ Done
configs/behavior/videomae_*.yaml     ✓ Done (5 files)
src/analytics/timeline.py            ✗ Not started
src/analytics/budget.py              ✗ Not started

data/processed/tubelets/labels.csv   ✓ 125,586 rows
data/processed/tubelets/cbvd5/       ✓ 11,369 tubelets (~6GB)
data/processed/tubelets/cvb/         ✓ 114,217 tubelets (~27GB)

runs/behavior/videomae_combined_v1/      ⟳ Running (Config 5, started 2026-04-29)
runs/behavior/videomae_cbvd5_v1/         ✗ Queued (after Config 5 finishes)
runs/behavior/videomae_cvb_v1/           ✗ Queued (after Config 5 finishes)
runs/behavior/videomae_cbvd5_to_cvb_v1/ ✗ Queued (after Configs 1+2 finish)
runs/behavior/videomae_cvb_to_cbvd5_v1/ ✗ Queued (after Configs 1+2 finish)
results/behavior/summary_table.csv   ✗ Not started
results/analytics/                   ✗ Not started
```

---

## 10. Execution Order — Current Position

```
✓ Tasks 5.1–5.10     (tubelet export complete, re-exported with fix)
✓ Tasks 6.1–6.7      (code + Docker + HiPE1 deployment complete)
✓ CVB rsync done     (24.3GB transferred, 33GB total on HiPE1, 5.6G free)
✓ HiPE1 verified     (125,587 rows in labels.csv, du -sh tubelets/ = 33G)
⟳ 6.8 running        (Config 5 Combined, GPU 0, tmux session: combined)
→ 6.9 Configs 1+2    (GPU 0 + GPU 1 in parallel, after Config 5 done)
→ 6.9 Configs 3+4    (GPU 0 + GPU 1 in parallel, after Configs 1+2 done)
→ 6.10               (evaluate all 5, --split val)
→ 7.1 → 7.2 → 7.3   (analytics, locally, fast)
```

**Monitor Config 5 from local machine:**

```bash
ssh hipe1 "tail -20 ~/cattle_behavior/logs/combined_v2.log"
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
```
