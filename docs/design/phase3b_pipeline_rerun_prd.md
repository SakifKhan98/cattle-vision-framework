# Phase 3b — RF-DETR-Seg Pipeline Re-run (Handoff Option 1)

## Product Requirements Document

**Author:** Sakif Khan  
**Date:** 2026-05-17  
**Status:** Planning — scope confirmed, implementation started

---

## 0. Context and Narrative Reframe

Phases 1–7 are complete using the SAM2 path: RF-DETR (detection-only) → SAM2 (segmentation) → OC-SORT (tracking) → VideoMAE (behavior) → analytics. But the thesis's central model contribution is **RF-DETR-Seg**: a student model distilled from SAM2 teacher pseudo-labels, achieving 85% detection mAP@50:95 and 79% segmentation mAP@50:95 on Config B (lr=5e-5, epoch 59 EMA).

The current pipeline narrative — "off-the-shelf models chained together" — understates this contribution. The narrative must be reframed as: _SAM2 teacher produces pseudo-labels → RF-DETR-Seg student is distilled → downstream pipeline consumes the student model._

This PRD covers the full pipeline re-run using RF-DETR-Seg (Handoff Option 1: re-run everything), followed by Phase 8 generalization evaluation on four additional datasets.

**Blocking PRD:** `docs/design/phase8_additional_datasets_prd.md` (Steps A–D deferred until RF-DETR-Seg pipeline is complete).

---

## 1. Scope

This PRD covers:

1. **RF-DETR-Seg segmentation inference** on CBVD-5 and CVB — replaces SAM2 in script 07.
2. **OC-SORT tracking re-run** with RF-DETR-Seg masks — produces new tracking_v2 data.
3. **Tubelet re-generation** from new tracks — produces new 125k+ tubelet clips.
4. **VideoMAE behavior re-training** on new tubelets — 5 configs (CBVD-5, CVB, combined, and two cross-domain).
5. **Evaluation and analytics re-run** on new behavior predictions.
6. **Phase 8 execution** (OpenCows2020, Cows2021, CattleEyeView, Freeman Center) on top of the RF-DETR-Seg pipeline.

This PRD does **not** cover:

- Retraining the RF-DETR-Seg model (already done on HiPE1; Config B EMA checkpoint archived).
- Retraining the RF-DETR detection-only model (frozen from Phase 1).
- Modifying the VideoMAE architecture or training hyperparameters (frozen from Phase 6).
- Controlled environmental perturbations (thesis §4.5.2 — deferred to Phase 9).
- Writing the final thesis document.

---

## 2. Pipeline Data Paths

All RF-DETR-Seg pipeline outputs use distinct directories to preserve SAM2 results for comparison:

| Stage         | SAM2 Path (Phases 1–7)          | RF-DETR-Seg Path (this PRD)            |
| ------------- | ------------------------------- | -------------------------------------- |
| Segmentation  | `data/processed/segmentation/`  | `data/processed/segmentation_rfdetr/`  |
| Tracking      | `data/processed/tracking_v2/`   | `data/processed/tracking_v2_rfdetr/`   |
| Tubelets      | `data/processed/tubelets/`      | `data/processed/tubelets_rfdetr/`      |
| Behavior runs | `runs/behavior/videomae_*_v1/`  | `runs/behavior/videomae_*_v2/`         |
| Predictions   | `results/behavior/predictions/` | `results/behavior/predictions_rfdetr/` |
| Analytics     | `results/analytics/`            | `results/analytics_rfdetr/`            |

The model checkpoint is at `runs/seg_medium_lr5e5/checkpoint_best_ema.pth` (406 MB, Config B EMA, epoch 59, copied from HiPE1 via `/home/sakif/cattle_logs/`).

---

## 3. Implementation Modules

### 3.1 Segmentation — `src/segmentation/rfdetr_seg_infer.py` (COMPLETED)

Deep module that loads the RF-DETR-Seg checkpoint and produces instance segmentation masks in the same `_masks.json` format as SAM2. Interface: `model.predict(image_path, threshold) → sv.Detections` with mask extraction, xyxy-to-xywh conversion, and RLE encoding via `mask_utils.py`.

**Config:** `configs/segmentation/rfdetr_seg.yaml`
**Shell wrapper:** `scripts/07b_run_rfdetr_seg.sh`

Key decisions:

- Score threshold 0.3 (same as SAM2).
- Frame-by-frame stateless inference: `model.predict()` is called once per frame in a loop. There is no `batch_size` setting — the model API does not expose batching at this call site.
- Output format matches tracking input contract: `bbox` in xywh, `mask_rle` as COCO RLE dict, `mask_area` as pixel count, frame keys as string integers.
- CBVD-5: sparse keyframes (6 per video), processed independently.
- CVB: 450-frame continuous clips, processed frame by frame (no re-prompting needed — RF-DETR-Seg is stateless per frame, unlike SAM2 which needed K=15 re-prompting to prevent drift).

**Verified:** Model loads, inference works (10 cows/frame at 1080p), output format passes `load_segmentation_json()` in tracking, tracking runs end-to-end on single-video test.

### 3.2 Tracking — `scripts/08b_run_tracking_rfdetr.sh` (TO WRITE)

Points OC-SORT at RF-DETR-Seg output. Uses mask IoU association (default). `src/tracking/track.py` already accepts `--seg_dir` and `--output_dir` flags — the shell script only wires the correct paths:

```bash
python src/tracking/track.py \
  --dataset cbvd5 \
  --seg_dir data/processed/segmentation_rfdetr/cbvd5 \
  --output_dir data/processed/tracking_v2_rfdetr/cbvd5
```

Repeat for `cvb`.

### 3.3 Tubelet Generation — `scripts/09b_generate_tubelets_rfdetr.sh` (TO WRITE)

Points tubelet extraction at RF-DETR-Seg tracking output. Same 16-frame, stride-4, 224×224 parameters (frozen from Phase 5).

`src/data/export_tubelets.py` accepts the following CLI flags — the shell script wires paths:

```bash
python src/data/export_tubelets.py \
  --output data/processed/tubelets_rfdetr \
  --cvb_tracking data/processed/tracking_v2_rfdetr/cvb \
  --cbvd5_tracking data/processed/tracking_v2_rfdetr/cbvd5
```

Note: the flags are `--output` (root output dir), `--cvb_tracking`, and `--cbvd5_tracking`. There is no `--tracking_dir` or `--output_dir` flag.

### 3.4 Behavior Training — `scripts/10b_train_behavior_rfdetr.sh` (TO WRITE)

Trains VideoMAE on new tubelets. `src/behavior/train.py` accepts only `--config`; there is no CLI override for individual YAML fields. Therefore, **5 new v2 YAML config files have been created** in `configs/behavior/` — one per experiment:

| v1 Config (SAM2 tubelets)         | v2 Config (RF-DETR-Seg tubelets)         |
| --------------------------------- | ---------------------------------------- |
| `videomae_combined.yaml`          | `videomae_combined_v2.yaml`              |
| `videomae_cbvd5.yaml`             | `videomae_cbvd5_v2.yaml`                 |
| `videomae_cvb.yaml`               | `videomae_cvb_v2.yaml`                   |
| `videomae_cbvd5_to_cvb.yaml`      | `videomae_cbvd5_to_cvb_v2.yaml`          |
| `videomae_cvb_to_cbvd5.yaml`      | `videomae_cvb_to_cbvd5_v2.yaml`          |

Each v2 config differs from its v1 counterpart in exactly two fields:
- `labels_csv: data/processed/tubelets_rfdetr/labels.csv`
- `output_dir: runs/behavior/videomae_*_v2`

All other hyperparameters (batch size, grad accum, epochs, lr, early stopping) are unchanged from Phase 6.

Must be run on HiPE1 (Docker, `--shm-size=16g`) due to RTX 3060 VRAM constraints and training duration.

### 3.5 Evaluation — `scripts/11b_evaluate_rfdetr.sh` (TO WRITE)

Runs `src/behavior/evaluate.py` on new checkpoints against new tubelets. Produces per-class F1, confusion matrices, and prediction CSVs in `results/behavior/predictions_rfdetr/`.

### 3.6 Analytics — `scripts/12b_generate_analytics_rfdetr.sh` (TO WRITE)

Runs `src/analytics/timeline.py` and `budget.py` on new predictions and tracking data. Outputs to `results/analytics_rfdetr/`.

### 3.7 Phase 8 Scripts — `scripts/13-28` (TO WRITE, deferred)

From `docs/design/phase8_additional_datasets_prd.md`. Steps A–D evaluate RF-DETR-Seg pipeline on four additional datasets. Scripts reference RF-DETR-Seg output paths.

---

## 4. Implementation Decisions

| Decision                   | Choice                                               | Rationale                                                                                                                       |
| -------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Data directory strategy    | Separate `_rfdetr` directories for each stage        | Preserves SAM2 results for comparison; avoids clobbering committed Phase 1–7 data; enables per-stage ablation                   |
| Output format              | Identical to SAM2 `_masks.json`                      | Tracking code is unchanged; downstream contracts are preserved                                                                  |
| Segmentation model         | RF-DETR-Seg-Medium, Config B (lr=5e-5), EMA epoch 59 | Best validation result from 100-epoch hyperparameter comparison; achieves 85% det mAP@50:95, 79% seg mAP@50:95 on pseudo-labels |
| Inference batching         | Frame-by-frame (stateless `model.predict()` per frame) | `model.predict()` API does not expose batching; each call processes one image independently; safe for 1080p + 10+ mask predictions on RTX 3060 12 GB |
| Inference resolution       | Model default (internal resize)                      | RF-DETR-Seg handles resize internally; no need to pre-resize frames                                                             |
| CVB frame processing       | Stateless per-frame inference                        | Unlike SAM2 which needed temporal propagation, RF-DETR-Seg has no memory across frames — each frame is independent              |
| Tracking association       | Mask IoU (not box IoU)                               | Preserves the improved IDF1 from mask-based association (67.31% with SAM2); expected similar gains with RF-DETR-Seg masks       |
| Behavior training location | HiPE1 (Docker, V100)                                 | VideoMAE training requires >12 GB VRAM and multi-day training; local RTX 3060 insufficient for full 5-config grid               |
| Tubelet parameters         | 16 frames, stride 4, 224×224                         | Frozen from Phase 5 — not changed                                                                                               |
| Run directory naming       | `videomae_*_v2/`                                     | Distinguishes from v1 (SAM2 tubelets); v1 checkpoints remain for comparison                                                     |
| Behavior v2 configs        | New YAML files (`*_v2.yaml`) in `configs/behavior/`  | `train.py` only accepts `--config`; no CLI field overrides; new files differ from v1 only in `labels_csv` and `output_dir`      |
| Phase 8 model              | RF-DETR-Seg for all 4 datasets                       | Consistent with Option 1 narrative; detection head used where segmentation not needed (OpenCows2020, Cows2021)                  |

---

## 5. Behavior Config Grid

Same 5 configurations as Phase 6, retrained on RF-DETR-Seg tubelets:

| Config                        | Train        | Val          | Classes | Purpose                   |
| ----------------------------- | ------------ | ------------ | ------- | ------------------------- |
| `videomae_cbvd5_v2`           | CBVD-5       | CBVD-5       | 5 (0–4) | CBVD-5 in-domain          |
| `videomae_cvb_v2`             | CVB          | CVB          | 7 (0–6) | CVB in-domain             |
| `videomae_combined_v2`        | CBVD-5 + CVB | CBVD-5 + CVB | 7 (0–6) | Combined in-domain        |
| `videomae_cbvd5_to_cvb_v2`   | CBVD-5       | CVB          | 5 (0–4) | CBVD-5 → CVB cross-domain |
| `videomae_cvb_to_cbvd5_v2`   | CVB          | CBVD-5       | 5 (0–4) | CVB → CBVD-5 cross-domain |

Class weights computed from new tubelet label distribution per config (same logic as `src/behavior/dataset.py`).

---

## 6. Testing Decisions

### What makes a good test

- Test external behavior: verify output files exist, have correct schema, and contain expected data shapes.
- Test data contracts: verify downstream stage can load upstream output without errors.
- Test edge cases: empty frames, single-detection frames, videos with no detections.
- Do NOT test model accuracy or ML metrics (those are evaluation, not testing).

### What to test

- **Segmentation output format**: verify `_masks.json` files have correct keys, field types, and non-null masks.
- **Tracking input compatibility**: verify `load_segmentation_json()` succeeds on all RF-DETR-Seg mask files.
- **Tracking output format**: verify `_tracks.json` files have correct schema with `track_id`, `bbox`, `mask_rle`, and valid stats.
- **Tubelet integrity**: verify generated tubelets have correct frame count (16), resolution (224×224), and corresponding label entries in `labels.csv`.
- **Pipeline integration**: verify a single video flows end-to-end (segmentation → tracking → tubelets) without errors.

### Prior art

Existing `tests/` directory contains unit tests for `mask_utils.py` and label utilities. Follow the same pytest pattern with fixtures for sample tracking JSONs.

---

## 7. Out of Scope

- Retraining RF-DETR-Seg (already done, checkpoint archived).
- Retraining RF-DETR detection-only model.
- Modifying VideoMAE architecture, training hyperparameters, or loss functions.
- Re-designing the tubelet extraction algorithm.
- Controlled environmental perturbations (Phase 9).
- Writing the final thesis document.
- Uploading new weights to HuggingFace (deferred until all training complete and results validated).
- Running Phase 8 Steps A–D until RF-DETR-Seg pipeline re-run is complete and validated.

---

## 8. Execution Order

Steps must run sequentially due to data dependencies:

```
Step 1: RF-DETR-Seg segmentation (script 07b) → segmentation_rfdetr/
    ↓
Step 2: Tracking (script 08b)                  → tracking_v2_rfdetr/
    ↓
Step 3: Tubelet generation (script 09b)        → tubelets_rfdetr/
    ↓
Step 4: Behavior training (script 10b)         → runs/behavior/videomae_*_v2/
    ↓
Step 5: Evaluation (script 11b)                → results/behavior/predictions_rfdetr/
    ↓
Step 6: Analytics (script 12b)                 → results/analytics_rfdetr/
    ↓
Step 7: Phase 8 (scripts 13–28)                → results/detection/, tracking/, behavior/, analytics/
```

Steps 1–3 run locally on RTX 3060. Step 4 runs on HiPE1 (Docker, V100). Steps 5–6 run locally. Step 7 runs locally or on HiPE1 depending on dataset size.

---

## 9. HiPE1 Execution Plan

HiPE1 is required for Step 4 (VideoMAE training — multi-day, >12 GB VRAM) and optional for other steps. See `docs/hipe_ops.md` for SSH config, conda activation, and Docker setup.

### 9.1 Which Steps Run Where

| Step | Script | Location | Reason |
|---|---|---|---|
| 1. Segmentation | 07b | Local (RTX 3060) | ~1 hr, fits in 12 GB VRAM |
| 2. Tracking | 08b | Local (RTX 3060) | ~30 min, CPU/GPU light |
| 3. Tubelets | 09b | Local (CPU) | I/O bound, no GPU needed |
| **4. Behavior train** | **10b** | **HiPE1 (V100, Docker)** | **~days, requires 16 GB VRAM** |
| 5. Evaluation | 11b | Local or HiPE1 | ~30 min on either |
| 6. Analytics | 12b | Local (CPU) | ~minutes |
| 7. Phase 8 | 13–28 | Local; Freeman may need HiPE1 | Depends on dataset size |

### 9.2 Step 4: Behavior Training on HiPE1

**One-time data sync (before training):**

```bash
# Sync the full repo to HiPE1
rsync -avz --progress \
  src/ configs/ data/processed/tubelets_rfdetr/ \
  hipe1:~/cattle_behavior_rfdetr/

# Sync Docker image if not already loaded
rsync -avz --progress cattle-videomae-v1.tar.gz hipe1:~/cattle_behavior_rfdetr/
ssh hipe1 "cd ~/cattle_behavior_rfdetr && docker load < cattle-videomae-v1.tar.gz"
```

**Training command (one config, run in tmux):**

```bash
ssh hipe1
cd ~/cattle_behavior_rfdetr
tmux new-session -s behavior_v2
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro \
  -v $(pwd)/src:/workspace/src:ro \
  cattle-behavior \
  python src/behavior/train.py --config configs/behavior/videomae_combined_v2.yaml \
  2>&1 | tee logs/combined_v2.log
```

Repeat for all 5 v2 configs (`videomae_cbvd5_v2.yaml`, `videomae_cvb_v2.yaml`, `videomae_cbvd5_to_cvb_v2.yaml`, `videomae_cvb_to_cbvd5_v2.yaml`). Use `Ctrl+B c` for new tmux windows, `Ctrl+B d` to detach.

**Retrieving results:**

```bash
for CONFIG in combined cvb cbvd5 cbvd5_to_cvb cvb_to_cbvd5; do
  rsync -avz hipe1:~/cattle_behavior_rfdetr/runs/behavior/videomae_${CONFIG}_v2/ \
    runs/behavior/videomae_${CONFIG}_v2/
done
rsync -avz hipe1:~/cattle_behavior_rfdetr/results/behavior/ results/behavior/
```

### 9.3 Step 3: Tubelet Sync to HiPE1

If tubelets are generated locally (Step 3), they must be synced to HiPE1 before Step 4:

```bash
rsync -avz --progress data/processed/tubelets_rfdetr/ hipe1:~/cattle_behavior_rfdetr/data/processed/tubelets_rfdetr/
```

### 9.4 Monitoring

```bash
# Latest epoch
ssh hipe1 "tail -2 ~/cattle_behavior_rfdetr/logs/combined_v2.log"

# GPU utilization
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# Active containers
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

---

## 10. Behavior v2 Config Files

Five new YAML configs have been created in `configs/behavior/` for v2 training. Each is identical to its v1 counterpart except for two fields:

```yaml
# Changed fields in every *_v2.yaml:
labels_csv: data/processed/tubelets_rfdetr/labels.csv   # was: tubelets/
output_dir: runs/behavior/videomae_<name>_v2             # was: ..._v1
```

This is necessary because `train.py` only accepts `--config` — there is no CLI mechanism to override individual YAML fields. The v1 configs remain unchanged so v1 (SAM2) results are reproducible without modification.

---

## 11. New Files Summary

| File                                              | Type    | Purpose                                |
| ------------------------------------------------- | ------- | -------------------------------------- |
| `configs/segmentation/rfdetr_seg.yaml`            | Config  | RF-DETR-Seg inference parameters       |
| `configs/behavior/videomae_combined_v2.yaml`      | Config  | Combined in-domain training (v2)       |
| `configs/behavior/videomae_cbvd5_v2.yaml`         | Config  | CBVD-5 in-domain training (v2)         |
| `configs/behavior/videomae_cvb_v2.yaml`           | Config  | CVB in-domain training (v2)            |
| `configs/behavior/videomae_cbvd5_to_cvb_v2.yaml`  | Config  | CBVD-5→CVB cross-domain training (v2)  |
| `configs/behavior/videomae_cvb_to_cbvd5_v2.yaml`  | Config  | CVB→CBVD-5 cross-domain training (v2)  |
| `src/segmentation/rfdetr_seg_infer.py`            | Module  | Inference script (COMPLETED)           |
| `scripts/07b_run_rfdetr_seg.sh`                   | Shell   | Segmentation wrapper (COMPLETED)       |
| `scripts/08b_run_tracking_rfdetr.sh`              | Shell   | Tracking wrapper                       |
| `scripts/09b_generate_tubelets_rfdetr.sh`         | Shell   | Tubelet wrapper                        |
| `scripts/10b_train_behavior_rfdetr.sh`            | Shell   | Behavior training wrapper              |
| `scripts/11b_evaluate_rfdetr.sh`                  | Shell   | Evaluation wrapper                     |
| `scripts/12b_generate_analytics_rfdetr.sh`        | Shell   | Analytics wrapper                      |
| `scripts/13-28` (16 scripts)                      | Shell   | Phase 8 dataset eval (deferred)        |
| `runs/seg_medium_lr5e5/checkpoint_best_ema.pth`   | Weights | Config B EMA checkpoint (DONE)         |
| `logs/rfdetr_seg_config_B.txt`                    | Log     | Training log from HiPE1 (DONE)         |

---

## 12. Frozen Constraints

- **7-class label map** — `data/label_map.json`, IDs 0–6, frozen.
- **Tubelet parameters** — 16 frames, stride 4, 224×224, frozen.
- **Dataset splits** — CBVD-5 val split = test split (no separate test set).
- **GPU constraints (RTX 3060)** — batch 2–4, resolution ≤640 for inference.
- **GPU constraints (HiPE1 V100)** — batch 4, grad accum 4 (effective 16), resolution 576 (divisible by 64), BF16 + gradient checkpointing.
- **All models frozen** — no retraining of detector, SAM2, or RF-DETR-Seg.
- **Behavior configs frozen** — same 5 train/val splits as Phase 6; v2 configs change only `labels_csv` and `output_dir`.
- **`results/` is committed** — new result files follow existing conventions.
- **OC-SORT path** — `third_party/OC_SORT` must be cloned for tracking (not in repo).

---

## 13. Verification Checklist

- [ ] Step 1: RF-DETR-Seg segmentation produces `_masks.json` for all CBVD-5 and CVB videos in `segmentation_rfdetr/`
- [ ] Step 2: Tracking produces `_tracks.json` for all videos with valid `association_mode: mask_iou` in `tracking_v2_rfdetr/`
- [ ] Step 3: Tubelet generation produces clips + `labels.csv` in `tubelets_rfdetr/` with correct class distribution
- [ ] Step 4: All 5 VideoMAE v2 configs train to completion without OOM; checkpoints in `videomae_*_v2/`
- [ ] Step 5: Evaluation produces per-class F1, predictions CSV, confusion matrices in `predictions_rfdetr/`
- [ ] Step 6: Analytics produces timelines, budgets, transition matrices, deviation in `analytics_rfdetr/`
- [ ] Step 7: Phase 8 scripts execute on all 4 additional datasets
- [ ] Compare v1 (SAM2) vs v2 (RF-DETR-Seg) metrics: detection mAP, tracking IDF1, behavior macro-F1
- [ ] Update `CLAUDE.md` §7 Key Results table with v2 results
- [ ] Commit all new scripts, configs, and result files
