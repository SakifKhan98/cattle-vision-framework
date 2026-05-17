# Cattle Vision Framework — AI Agent Navigation Guide

## 1. Project Overview

MS thesis project (Texas State University, Sakif Khan, 2026) that recognizes 7 dairy-cattle
behaviors from surveillance video. The pipeline runs RF-DETR detection → SAM2 segmentation →
OC-SORT tracking → VideoMAE classification → activity-budget analytics.

## 2. Repository Map

| Directory | Purpose |
|-----------|---------|
| `src/` | Library modules — no `main()` calls; import from scripts |
| `scripts/` | Numbered shell wrappers (01–12) that run the pipeline end-to-end |
| `configs/` | YAML experiment configs; all hyperparams live here |
| `docker/` | One Dockerfile per pipeline stage group + docker-compose.yml |
| `notebooks/` | Visualization and figure generation only |
| `results/` | **Committed** small result files (JSON, CSV, PNG) |
| `runs/` | Gitignored large checkpoints; download from HuggingFace |
| `weights/` | Gitignored pretrained weights; download from HuggingFace |
| `data/` | `label_map.json` committed; `raw/` and `processed/` gitignored |
| `paper/` | AIIoT26 paper figures and sample images |
| `docs/` | End-user docs; `docs/design/` has internal planning docs |
| `logs/` | HiPE1 training logs (committed, informational) |
| `tests/` | Unit tests (`pytest`) |
| `third_party/` | Gitignored; OC-SORT cloned here during setup |
| `_archive/` | Gitignored; archived large files (seg runs, old OC-SORT copy) |
| `scripts/hipe/` | HiPE1-specific training scripts (run via Docker on V100) |

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

| Dependency | How to install | Used by |
|------------|---------------|---------|
| OC-SORT | `git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT` | `src/tracking/track.py` |
| SAM2 | `pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'` | `src/segmentation/segment.py` |
| SAM2 checkpoint | Download via SAM2 repo script → `weights/sam2.1_hiera_large.pt` | SAM2 segmentation |
| RF-DETR backbone | `huggingface-cli download sakifkhan/cattle-vision-framework rf-detr-medium.pth` → `weights/` | detector training |
| Detector checkpoint | HuggingFace: `rfdetr_combined_v1_best.pth` → `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth` | script 06+ |
| VideoMAE checkpoints | HuggingFace: `videomae_*_v1.pt` → `runs/behavior/videomae_*_v1/checkpoint_best.pt` | scripts 11–12 |

See `docs/setup.md` for exact download commands.

## 5. Frozen Decisions (do not change)

**7-class label map** — defined in `data/label_map.json` and `src/data/label_utils.py`:

| ID | Behavior | Datasets |
|----|----------|---------|
| 0 | Standing | CBVD-5, CVB |
| 1 | Lying | CBVD-5, CVB |
| 2 | Foraging/Grazing | CBVD-5, CVB |
| 3 | Drinking | CBVD-5, CVB |
| 4 | Ruminating | CBVD-5, CVB |
| 5 | Grooming | CVB only |
| 6 | Other | CVB only |

Cross-dataset eval uses IDs 0–4 only.

**Tubelet parameters:** 16 frames, stride 4, 224×224 px, from `configs/behavior/*.yaml`.

**Dataset splits:** CBVD-5 test split = validation split (test labels not released).

**GPU constraints (RTX 3060, 12 GB):** batch 4, grad accum 4 (effective 16), resolution 576
(must be divisible by 64), BF16 mixed precision + gradient checkpointing always on.

## 6. Data Contracts

**Tracking JSON** (`data/processed/tracking_v2/{dataset}/{video_id}_tracks.json`):
```json
{"frames": [{"frame_idx": 0, "tracks": [{"track_id": 1, "bbox": [x,y,w,h], "mask_rle": {...}}]}]}
```
`mask_rle` present only when SAM2 segmentation path was used (script 07).

**labels.csv** (`data/processed/tubelets/labels.csv`):
Columns: `dataset, video_id, tubelet_dir, start_frame, end_frame, label_id`
`track_id` is the last path component of `tubelet_dir` (e.g., `data/processed/tubelets/cbvd5/341/kf6_instc85ac7`).

**Predictions CSV** (`results/behavior/predictions/{run}_val.csv`):
Columns: `dataset, video_id, tubelet_dir, start_frame, end_frame, label_id, pred_label_id, logit_0..6`

## 7. Key Results

| Stage | Metric | Value |
|-------|--------|-------|
| Detection | mAP@50 (combined, cross-domain) | 70.4% |
| Tracking | IDF1 | 67.31% |
| Tracking | MOTA | 36.61% |
| Behavior Config 1 (CBVD-5 in-domain) | macro-F1 | 0.3149 |
| Behavior Config 2 (CVB in-domain) | macro-F1 | 0.7607 |
| Behavior Config 3 (CBVD-5→CVB) | macro-F1 | 0.1690 |
| Behavior Config 4 (CVB→CBVD-5) | macro-F1 | 0.1789 |
| Behavior Config 5 (combined) | macro-F1 | **0.7537** |

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

## 10. Current Status (Phase 7)

- Phases 0–6 complete, repo cleaned and structured.
- **TODO (Step I):** Implement `src/analytics/timeline.py` and `src/analytics/budget.py` (currently empty stubs).
- **TODO:** Upload model weights to HuggingFace (`sakifkhan/cattle-vision-framework`) and tracking_v2 data to `sakifkhan/cattle-vision-data`.
- **TODO:** `git init` and push to GitHub.
- See `docs/design/phase7_cleanup_prd.md` §12 for step-by-step checklist.
