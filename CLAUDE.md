# Cattle Vision Framework — Claude Code Navigation Guide

> **Other agents:** see `AGENTS.md` (same content, generic format).

## 1. Project Overview

MS thesis project (Texas State University, Sakif Khan, 2026) that recognizes 7 dairy-cattle
behaviors from surveillance video. The pipeline runs RF-DETR detection → SAM2 segmentation →
OC-SORT tracking → VideoMAE classification → activity-budget analytics.

## 2. Repository Map

| Directory       | Purpose                                                          |
| --------------- | ---------------------------------------------------------------- |
| `src/`          | Library modules — no `main()` calls; import from scripts         |
| `scripts/`      | Numbered shell wrappers (01–12) that run the pipeline end-to-end |
| `configs/`      | YAML experiment configs; all hyperparams live here               |
| `docker/`       | One Dockerfile per pipeline stage group + docker-compose.yml     |
| `notebooks/`    | Visualization and figure generation only                         |
| `results/`      | **Committed** small result files (JSON, CSV, PNG)                |
| `runs/`         | Gitignored large checkpoints; download from HuggingFace          |
| `weights/`      | Gitignored pretrained weights; download from HuggingFace         |
| `data/`         | `label_map.json` committed; `raw/` and `processed/` gitignored   |
| `paper/`        | AIIoT26 paper figures and sample images                          |
| `docs/`         | End-user docs; `docs/design/` has internal planning docs         |
| `logs/`         | HiPE1 training logs (committed, informational)                   |
| `tests/`        | Unit tests (`pytest`)                                            |
| `third_party/`  | Gitignored; OC-SORT cloned here during setup                     |
| `_archive/`     | Gitignored; archived large files (seg runs, old OC-SORT copy)    |
| `scripts/hipe/` | HiPE1-specific training scripts (run via Docker on V100)         |

## 3. Pipeline Execution Order

Each script depends on the output of the previous one. Run from repo root.

```
01_inspect_data.sh        reads  data/raw/{cbvd5,cvb}/               → console summary
02_prepare_cbvd5.sh       reads  data/raw/cbvd5/                      → data/processed/detection/cbvd5/
03_prepare_cvb.sh         reads  data/raw/cvb/                        → data/processed/detection/cvb/
04_merge_datasets.sh      reads  data/processed/detection/{cbvd5,cvb}/→ data/processed/detection/combined/
05_train_detector.sh      reads  combined/ + weights/rf-detr-medium.pth → runs/detection/rfdetr_combined_v1/
06_run_detection.sh       reads  raw videos + detector checkpoint      → data/processed/tracking/{cbvd5,cvb}/
07_run_segmentation.sh    reads  tracking/ + weights/sam2.1_hiera_large.pt → data/processed/tracking_v2/
07_train_rfdetr_seg.sh    reads  data/rfdetr_seg/ (SAM2 pseudo-labels) → runs/seg_medium_lr1e4/  [HiPE1]
08_run_tracking.sh        reads  tracking/ (box-only, skip if 07 ran)  → data/processed/tracking_v2/
09_generate_tubelets.sh   reads  tracking_v2/ + data/raw/              → data/processed/tubelets/ (125k clips)
10_train_behavior.sh      reads  tubelets/ + configs/behavior/          → runs/behavior/videomae_*/
11_evaluate.sh            reads  tubelets/ + checkpoints                → results/behavior/
12_generate_analytics.sh  reads  results/behavior/predictions/ + tracking_v2/ → results/analytics/
```

## 4. External Dependencies (NOT in repo)

| Dependency           | How to install                                                                                             | Used by                       |
| -------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------- |
| OC-SORT              | `git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT`                                     | `src/tracking/track.py`       |
| SAM2                 | `pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'`                                    | `src/segmentation/segment.py` |
| SAM2 checkpoint      | Download via SAM2 repo script → `weights/sam2.1_hiera_large.pt`                                            | SAM2 segmentation             |
| RF-DETR backbone     | `huggingface-cli download sakifkhan/cattle-vision-framework rf-detr-medium.pth` → `weights/`               | detector training             |
| Detector checkpoint  | HuggingFace: `rfdetr_combined_v1_best.pth` → `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth` | script 06+                    |
| VideoMAE checkpoints | HuggingFace: `videomae_*_v1.pt` → `runs/behavior/videomae_*_v1/checkpoint_best.pt`                         | scripts 11–12                 |

See `docs/setup.md` for exact download commands.

## 5. Frozen Decisions (do not change)

**7-class label map** — defined in `data/label_map.json` and `src/data/label_utils.py`:

| ID  | Behavior         | Datasets    |
| --- | ---------------- | ----------- |
| 0   | Standing         | CBVD-5, CVB |
| 1   | Lying            | CBVD-5, CVB |
| 2   | Foraging/Grazing | CBVD-5, CVB |
| 3   | Drinking         | CBVD-5, CVB |
| 4   | Ruminating       | CBVD-5, CVB |
| 5   | Grooming         | CVB only    |
| 6   | Other            | CVB only    |

Cross-dataset eval uses IDs 0–4 only.

**Tubelet parameters:** 16 frames, stride 4, 224×224 px, from `configs/behavior/*.yaml`.

**Dataset splits:** CBVD-5 test split = validation split (test labels not released).

**GPU constraints (RTX 3060, 12 GB):** batch 4, grad accum 4 (effective 16), resolution 576
(must be divisible by 64), BF16 mixed precision + gradient checkpointing always on.

## 6. Data Contracts

**Tracking JSON** (`data/processed/tracking_v2/{dataset}/{video_id}_tracks.json`):

```json
{
  "video_id": "...", "dataset": "cvb",
  "frames": {
    "1": [{"track_id": 11, "bbox": [x1,y1,x2,y2], "score": 1.0, "mask_rle": {...}, "mask_area": 44290}]
  },
  "stats": {"total_frames": 450, "total_unique_tracks": 19, "association_mode": "mask_iou"}
}
```

Frame keys are string integers. `bbox` is absolute pixels `[x1,y1,x2,y2]`.
`mask_rle` present only when SAM2 segmentation path was used (script 07).

**labels.csv** (`data/processed/tubelets/labels.csv`):
Columns: `dataset, video_id, tubelet_dir, start_frame, end_frame, label_id`
`track_id` is the last path component of `tubelet_dir` (e.g., `data/processed/tubelets/cbvd5/341/kf6_instc85ac7`).

**Predictions CSV** (`results/behavior/predictions/{run}_val.csv`):
Columns: `dataset, video_id, tubelet_dir, start_frame, end_frame, label_id, pred_label_id, logit_0..6`

## 7. Key Results

| Stage                                       | Metric                          | v1         | v2 (RF-DETR tracks) |
| ------------------------------------------- | ------------------------------- | ---------- | ------------------- |
| Detection                                   | mAP@50 (combined, cross-domain) | 70.4%      | —                   |
| Tracking                                    | IDF1                            | 67.31%     | —                   |
| Tracking                                    | MOTA                            | 36.61%     | —                   |
| Behavior Config 1 (CBVD-5 in-domain)        | macro-F1                        | 0.3149     | **0.4511**          |
| Behavior Config 2 (CVB in-domain)           | macro-F1                        | 0.7607     | **0.7770**          |
| Behavior Config 3 (CBVD-5→CVB cross-domain) | macro-F1                        | 0.1690     | **0.1722**          |
| Behavior Config 4 (CVB→CBVD-5 cross-domain) | macro-F1                        | 0.1789     | **0.2253**          |
| Behavior Config 5 (combined)                | macro-F1                        | **0.7537** | 0.7507              |

v2 models trained on RF-DETR-tracked tubelets (no SAM2 segmentation in tracking loop).
Full per-class breakdown in `results/behavior/f1_per_class.csv`.

## 8. Common Gotchas

- **CBVD-5 test=val**: The dataset has no separate test set; validation split is used as test.
- **CVB frame presence**: Not all annotated frames have corresponding image files. `convert_cvb.py` filters to present frames only.
- **Class weights**: Computed from tubelet label distribution per config (not from raw annotation counts). Logic in `src/behavior/dataset.py`.
- **Docker shm-size**: VideoMAE DataLoader requires `--shm-size=16g` with Docker. See `docker/docker-compose.yml`.
- **conda not in PATH**: On HiPE1, activate with `source /home/zxs12/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer`.
- **OC-SORT path**: `src/tracking/track.py` inserts `third_party/OC_SORT` into `sys.path` at runtime. Must clone before running script 08.
- **`results/` is committed**: Do not add `results/` to `.gitignore`. Only specific large subdirs are gitignored.

## 9. HiPE1 Server

- SSH alias: configure `~/.ssh/config` with `Host hipe1` → see `docs/hipe_ops.md`
- Home dir: `/home/zxs12/`
- Training scripts: `scripts/hipe/` (run via Docker)
- Docker eval pattern: `docker run --gpus all --shm-size=16g -v $(pwd):/workspace cattle-behavior python src/behavior/train.py --config configs/behavior/videomae_combined.yaml`

## 10. Current Status

**Phases 0–7 complete.** Repo is live at `github.com/SakifKhan98/cattle-vision-framework`.

| Phase | What | Status |
|---|---|---|
| 0 | Dataset analysis + label design | ✅ complete |
| 1 | RF-DETR detection (train + inference) | ✅ complete |
| 2 | RF-DETR detection inference on all videos | ✅ complete |
| 3 | SAM2 instance segmentation | ✅ complete |
| 3b | RF-DETR-Seg distillation experiment (HiPE1) | ✅ complete |
| 4 | OC-SORT tracking (IDF1=67.31%, MOTA=36.61%) | ✅ complete |
| 5 | Tubelet generation (125k clips) | ✅ complete |
| 6 | VideoMAE behavior classification (5 configs) | ✅ complete |
| 7 | Behavior analytics (timelines, budgets, deviation) | ✅ complete |
| **8** | **Additional dataset evaluation (generalization)** | **🔲 next** |

**Phase 8 next steps** (see `docs/design/phase8_additional_datasets_prd.md`):
1. OpenCows2020 — detection mAP (OOD generalization, images only)
2. Cows2021 — detection mAP + short-term tracking IDF1
3. CattleEyeView — detection + Mask IoU + IDF1
4. Freeman Center — full pipeline + behavior F1 + activity budgets (Freeman Center annotation format and local path TBD)

**Remaining TODOs:**
- Upload model weights to HuggingFace (`sakifkhan/cattle-vision-framework`)
- Upload tracking_v2 data to `sakifkhan/cattle-vision-data`

## 11. Design Documents

All phase reports and planning docs live in `docs/design/`:

| File | Contents |
|---|---|
| `reports/phase0_report.md` | Dataset analysis, label taxonomy design |
| `reports/phase1_detection_report.md` | RF-DETR training + evaluation |
| `reports/phase3_segmentation_report.md` | SAM2 segmentation |
| `reports/phase3_rfdetr_seg_experiment_report.md` | RF-DETR-Seg distillation (HiPE1) |
| `reports/phase4_tracking_report.md` | OC-SORT tracking, full metrics |
| `reports/phase5_tuebelet_report.md` | Tubelet generation logic + stats |
| `reports/phase6_behavior_classify_report.md` | VideoMAE training + all 5 config results |
| `reports/phase7_analytics_report.md` | Analytics implementation + thesis sections |
| `phase7_cleanup_prd.md` | Phase 7 cleanup checklist (Steps A–I, all complete) |
| `phase8_additional_datasets_prd.md` | Phase 8 plan: additional dataset evaluation |

## Agent skills

### Issue tracker

Issues live in GitHub Issues at `SakifKhan98/cattle-vision-framework`. Use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical label strings: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` at repo root, `docs/adr/` for ADRs. See `docs/agents/domain.md`.
