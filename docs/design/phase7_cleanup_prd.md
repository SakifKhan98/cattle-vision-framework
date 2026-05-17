# Phase 7 — Cleanup, Reorganization & GitHub Publication
## Product Requirements Document

**Author:** Sakif Khan
**Date:** 2026-05-16 (revised 2026-05-16)
**Status:** Planning — PRD finalized, no implementation started
**Working directory (before reorg):** `~/TXST/Thesis/cattle-vision-framework/`
**Target GitHub repo root:** promoted contents of `one_day/`

---

## 0. Context

Phases 1–6 are complete. The codebase evolved through trial-and-error: data in two locations, multiple Dockerfiles and training scripts scattered at the root and in subdirectories, third-party repos copied in-place, intermediate training runs accumulating large checkpoint files, and documentation fragmented across `docs/`, `one_day/docs/`, and planning markdown files.

Phase 7 has two sub-goals:

1. **Analytics** — implement `src/analytics/timeline.py` and `budget.py` (currently empty stubs); add additional cattle surveillance datasets for Phase 7 analysis
2. **Cleanup + Publication** — reorganize, document, and publish to GitHub (this document covers this)

This PRD covers sub-goal 2. Sub-goal 1 (analytics implementation and new datasets) is listed as Step I at the end and implemented after the repo is clean.

---

## 1. Decisions Frozen

| Decision | Choice | Rationale |
|----------|--------|-----------|
| New repo root | Promote `one_day/` to root (flatten) | Cleaner clone experience; eliminates confusing nesting |
| OC-SORT | Remove from working tree; user clones official repo to `third_party/OC_SORT/` (gitignored) during setup | Keeps third-party source out of git; no submodule complexity; preserves internal import path |
| SAM2 | Remove copied repo; install via `pip install 'sam2 @ git+...'` | Installed package; no internal import path dependency |
| Intermediate seg runs | Move to `_archive/` (not delete), gitignore checkpoint `.pth` files | Safe — verify nothing breaks before final deletion |
| tracking_v2/cvb_mh5, cvb_mh7 | Move to `_archive/runs/tracking_experiments/` | Experimental min_hits variants; not used by main pipeline |
| tracking_v2/cbvd5 + cvb | Upload to HuggingFace as dataset release | Enables quickstart path (skip scripts 06–08); cbvd5=20MB, cvb=641MB |
| Model weights distribution | HuggingFace Hub | Designed for large model files; free; gives permanent URLs |
| "Deletion" strategy | Move to `_archive/` folder, gitignore it | Safe fallback while we verify nothing downstream breaks |
| Docker strategy | Docker Compose — one image per pipeline stage | Avoids dep conflicts across stages; supports both stage-by-stage and end-to-end run |
| Prediction CSVs | Commit to git (17 MB total) | Allows analytics reproduction without re-running inference |
| results/segmentation/viz/ | Gitignore (50 MB of sample images) | Large media; not needed for reproducibility |
| .docx phase reports | Convert to Markdown (extract figures separately) | Readable by humans and AI agents; diffable in git |
| GitHub Actions CI | Not needed yet — commit and push manually | No deployment requirements for thesis repo |
| data_raw/ at root | Already deleted by user | Resolved |

---

## 2. Current State Inventory

### 2.1 Root level (`cattle-vision-framework/`)

| Path | Size | Keep? | Action |
|------|------|-------|--------|
| `CLAUDE.md` | small | ✓ | Rewrite; move to new repo root |
| `env.yaml` | small | ✓ | Rename to `environment.yml`, move to new root |
| `rf-detr-medium.pth` | ~200 MB | ✓ (gitignored) | Move to `weights/rf-detr-medium.pth`, gitignore, reference HuggingFace |
| `rf-detr-seg-medium.pt` | ~200 MB | ✓ (gitignored) | Move to `weights/`, upload to HuggingFace, gitignore |
| `data_raw/` | — | ✗ | Already deleted by user |
| `sam2/` | 193 MB | ✗ | Move to `_archive/third_party/sam2/`; document as `pip install sam2` |
| `docs/` | small | partial | Merge useful `.md` files into new `docs/`; convert `.docx` to Markdown |
| `cattle-vision-framework-plan.md` | small | ✗ | Move to `_archive/planning/` |
| `project_structure.md` | small | ✗ | Move to `_archive/planning/` |
| `context-restoration-prompt.md` | small | ✗ | Move to `_archive/planning/` |
| `create_cattle_framework.sh` | small | ✗ | Move to `_archive/planning/` |
| `.codex` | empty | ✗ | Delete (OpenAI Codex config, not relevant) |

### 2.2 `one_day/` contents (becomes new repo root)

#### Files at `one_day/` root

| Path | Keep? | Action |
|------|-------|--------|
| `README.md` | ✓ | Rewrite (currently empty) |
| `LICENSE` | ✓ | Keep as-is |
| `requirements.txt` | ✓ | Populate (currently empty) |
| `Dockerfile` | ✓ | Move to `docker/Dockerfile.detection` (covers RF-DETR detection + RF-DETR-Seg) |
| `Dockerfile.videomae` | ✓ | Move to `docker/Dockerfile.behavior` |
| `train_hipe1.py` | ✓ | Move to `scripts/hipe/train_rfdetr_seg_medium.py` |
| `train_hipe2.py` | ✓ | Move to `scripts/hipe/train_rfdetr_seg_large.py` |
| `cattle-rfdetr-seg-v1.tar.gz` | ✗ (gitignored) | Upload to HuggingFace; move to `_archive/weights/` |
| `cattle-videomae-v1.tar.gz` | ✗ (gitignored) | Upload to HuggingFace; move to `_archive/weights/` |
| `rf-detr-medium.pth` | ✓ (gitignored) | Move to `weights/`, gitignore (consolidate with root-level copy) |

#### Directories

| Path | Size | Keep? | Action |
|------|------|-------|--------|
| `src/` | small | ✓ | Keep as-is (well organized) |
| `configs/` | small | ✓ | Keep as-is |
| `scripts/` | small | ✓ | Keep; add `hipe/` subdirectory |
| `notebooks/` | small | ✓ | Keep as-is |
| `results/` | small | ✓ | Keep; add `analytics/` subdir for Phase 7 outputs |
| `tests/` | small | ✓ | Keep; expand |
| `docs/` | small | ✓ | Reorganize (see §5) |
| `logs/` | small | ✓ | Keep; add `logs/README.md` |
| `OC_SORT/` | 98 MB | ✗ | Move to `_archive/third_party/OC_SORT/`; document as `git clone` to `third_party/OC_SORT/` (gitignored) |
| `models/sam2/` | ~2 GB checkpoint | ✓ (gitignored) | Directory stays; checkpoint gitignored; `sam2` package installed externally |
| `aiIoT_paper/` | small | ✓ | Rename to `paper/`; reorganize images into `paper/sample_images/` |
| `data/label_map.json` | small | ✓ | Keep |
| `data/raw/` | 27 GB | ✓ (gitignored) | Gitignored; detailed download instructions in `docs/datasets.md` |
| `data/processed/` | ~65 GB | ✗ (gitignored) | Fully gitignored; regenerated by running scripts 01–09 |
| `data/rfdetr_seg/` | small | ✓ (gitignored) | Gitignored; training data for RF-DETR-Seg distillation |
| `data/rfdetr_seg_eval/` | small | ✓ (gitignored) | Gitignored |
| `runs/detection/rfdetr_combined_v1/` | 2.4 GB | ✓ (gitignored) | Gitignored; best checkpoint uploaded to HuggingFace; keep `log.txt` + `results.json` committed in `results/detection/` |
| `runs/seg_medium_lr1e4/` | 6.7 GB | ✓ (gitignored) | Move to `_archive/runs/`; gitignore `.pth` files; extract and commit metadata (`log.txt`, `results.json`, `run_config.json`) to `results/segmentation/` |
| `runs/seg_medium_lr5e5/` | 6.7 GB | ✓ (gitignored) | Same as above |
| `runs/seg_medium_lr1e4_baseline/` | tiny | ✓ | Move to `_archive/runs/`; it is just one tfevents file |
| `runs/kaggle_eval_full/` | tiny | ✓ | Move to `results/segmentation/kaggle_eval/` (committed) |
| `runs/tracking/` | empty | ✗ | Delete |
| `runs/behavior/videomae_*/` | 3.9 GB | ✓ (gitignored) | Gitignored; best checkpoints uploaded to HuggingFace; `log.csv` from each committed in `results/behavior/training_logs/` |

---

## 3. Target Repository Structure

```
cattle-vision-framework/              ← GitHub repo root (= promoted one_day/)
│
├── README.md                         ← Complete project overview (rewrite)
├── CLAUDE.md                         ← AI agent navigation guide (rewrite)
├── LICENSE                           ← MIT (keep as-is)
├── environment.yml                   ← Conda env (renamed from env.yaml)
├── requirements.txt                  ← Pip deps with pinned versions (populate)
├── .gitignore                        ← Comprehensive (create)
│
├── docker/
│   ├── docker-compose.yml            ← Orchestrates all pipeline stages (NEW)
│   ├── Dockerfile.data               ← Stages 1–4, 8–9, 12: preprocessing, tracking, tubelets, analytics (NEW)
│   ├── Dockerfile.detection          ← Stages 5–6: RF-DETR detection training + inference (was Dockerfile)
│   ├── Dockerfile.segmentation       ← Stage 7a: SAM2 segmentation (NEW)
│   ├── Dockerfile.rfdetr_seg         ← Stage 7b: RF-DETR-Seg distillation training (split from Dockerfile)
│   ├── Dockerfile.tracking           ← Stage 8: OC-SORT tracking (NEW — or merged into Dockerfile.data)
│   └── Dockerfile.behavior           ← Stages 10–11: VideoMAE training + eval (was Dockerfile.videomae)
│
├── configs/
│   ├── detection/
│   │   ├── rfdetr_cbvd5.yaml
│   │   ├── rfdetr_cvb.yaml
│   │   └── rfdetr_combined.yaml
│   ├── segmentation/
│   │   └── sam2.yaml
│   ├── tracking/
│   │   └── ocsort.yaml
│   └── behavior/
│       ├── videomae_cbvd5.yaml
│       ├── videomae_cvb.yaml
│       ├── videomae_combined.yaml
│       ├── videomae_cbvd5_to_cvb.yaml
│       ├── videomae_cvb_to_cbvd5.yaml
│       └── videomae_sanity.yaml
│
├── data/
│   ├── label_map.json                ← Canonical class definitions
│   ├── raw/                          ← gitignored; see docs/datasets.md for download
│   │   ├── cbvd5/
│   │   ├── cvb/
│   │   └── [future datasets]/        ← Phase 7 additional datasets go here
│   └── processed/                    ← gitignored; generated by scripts 01–09
│       ├── detection/
│       ├── segmentation/
│       ├── tracking/
│       ├── tracking_v2/
│       └── tubelets/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── convert_cbvd5.py
│   │   ├── convert_cvb.py
│   │   ├── merge_coco.py
│   │   ├── make_mini.py
│   │   ├── export_tubelets.py
│   │   ├── label_utils.py
│   │   └── validate_tubelets.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── infer_video.py
│   │   └── infer_dataset.py
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── segment.py
│   │   └── mask_utils.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── track.py
│   │   ├── eval_tracking.py
│   │   ├── render_video.py
│   │   ├── render_behavior_video.py
│   │   └── visualize_tracks.py
│   ├── behavior/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── timeline.py              ← IMPLEMENT in Step I (currently empty)
│   │   └── budget.py                ← IMPLEMENT in Step I (currently empty)
│   ├── tools/
│   │   ├── check_checkpoint.py
│   │   ├── eval_rfdetr_seg.py
│   │   ├── eval_rfdetr_seg_kaggle.py
│   │   ├── plot_training_curves.py
│   │   └── sam2_to_coco_seg.py
│   └── utils/
│       └── __init__.py                  ← empty stubs (io, video, perturbations, metrics) deleted — nothing imports them
│
├── scripts/
│   ├── 01_inspect_data.sh
│   ├── 02_prepare_cbvd5.sh
│   ├── 03_prepare_cvb.sh
│   ├── 04_merge_datasets.sh
│   ├── 05_train_detector.sh
│   ├── 06_run_detection.sh
│   ├── 07_run_segmentation.sh
│   ├── 07_train_rfdetr_seg.sh
│   ├── 08_run_tracking.sh               ← empty stub; write during Step E
│   ├── 09_generate_tubelets.sh          ← empty stub; write during Step E
│   ├── 10_train_behavior.sh             ← empty stub; write during Step E
│   ├── 11_evaluate.sh                   ← empty stub; write during Step E
│   ├── 12_generate_analytics.sh         ← empty stub; write during Step E
│   ├── run_pipeline.sh                  ← NEW: runs all stages end-to-end (see §5)
│   ├── generate_overlays.py
│   ├── plot_training_results.py
│   └── hipe/
│       ├── README.md                    ← HiPE1 usage guide
│       ├── train_rfdetr_seg_medium.py   (was train_hipe1.py)
│       ├── train_rfdetr_seg_large.py    (was train_hipe2.py)
│       └── train_rfdetr_seg_parameterized.py  (was scripts/train_server.py — env-var-driven general trainer)
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_detection_visualization.ipynb
│   ├── 03_tracking_visualization.ipynb
│   ├── 04_behavior_results.ipynb
│   ├── 05_analytics_figures.ipynb
│   └── 06_plot_charts.ipynb
│
├── results/                          ← Committed to git (small files only)
│   ├── detection/
│   │   ├── cbvd5_test_ap.json
│   │   ├── cvb_test_ap.json
│   │   └── combined_ood.json
│   ├── segmentation/
│   │   ├── cbvd5_segmentation_stats.csv
│   │   ├── cbvd5_summary.json
│   │   ├── cvb_segmentation_stats.csv
│   │   ├── cvb_summary.json
│   │   ├── kaggle_eval/             ← moved from runs/kaggle_eval_full/
│   │   ├── seg_medium_lr1e4_metadata/  ← extracted from _archive (log.txt, results.json, run_config.json)
│   │   └── seg_medium_lr5e5_metadata/  ← extracted from _archive
│   ├── tracking/
│   │   ├── cbvd5_idf1.json
│   │   ├── cvb_idf1.json
│   │   ├── tracking_per_video_all.csv
│   │   └── tracking_summary_all.json
│   ├── behavior/
│   │   ├── f1_per_class.csv          ← committed (ground truth results)
│   │   ├── summary_table.csv         ← 5-config comparison
│   │   ├── training_logs/            ← log.csv from each videomae run (committed, small)
│   │   │   ├── videomae_combined_v1.csv
│   │   │   ├── videomae_cvb_v1.csv
│   │   │   ├── videomae_cbvd5_v1.csv
│   │   │   ├── videomae_cbvd5_to_cvb_v1.csv
│   │   │   └── videomae_cvb_to_cbvd5_v1.csv
│   │   ├── predictions/              ← committed (17 MB total, manageable)
│   │   └── confusion_matrices/       ← 5 PNGs (committed, small)
│   ├── generalization/
│   │   ├── ood_summary.csv
│   │   └── perturbation_delta.csv
│   └── analytics/                    ← Phase 7 outputs (new, mostly gitignored)
│       ├── timelines/                ← gitignored (many large CSVs)
│       ├── activity_budget.csv       ← committed
│       ├── transition_matrix.csv     ← committed
│       └── behavior_deviation.csv    ← committed (deviations from dataset baseline; per proposal §4.6.3)
│
├── paper/                            ← Renamed from aiIoT_paper/
│   ├── fig2_training_curves.py
│   ├── fig2_training_curves.png
│   ├── fig2_training_curves.pdf
│   ├── fig3_metric_comparison.py
│   ├── fig3_metric_comparison.png
│   ├── fig3_metric_comparison.pdf
│   └── sample_images/
│       ├── cbvd5_sample.jpg
│       ├── cbvd5_seg.jpg
│       ├── cvb_sample.jpg
│       ├── cvb_seg.jpg
│       └── kaggle_eval_seg_dataset_sample.jpg
│
├── tests/
│   └── test_label_utils.py
│
├── weights/                          ← gitignored; populated by setup or download script
│   ├── rf-detr-medium.pth            ← backbone (download from HuggingFace)
│   ├── rf-detr-seg-medium.pt         ← Phase 3b best (download from HuggingFace)
│   └── sam2.1_hiera_large.pt         ← SAM2 checkpoint (download via SAM2 repo script)
│
├── logs/                             ← HiPE1 training logs (informational, committed)
│   ├── README.md
│   └── *.log
│
├── docs/
│   ├── setup.md                      ← Installation, data download, weight download
│   ├── pipeline.md                   ← End-to-end pipeline walkthrough (scripts 01–12)
│   ├── datasets.md                   ← Dataset details, download links, formats
│   ├── results.md                    ← All results summary across all phases
│   ├── docker.md                     ← Docker and Docker Compose usage guide (NEW)
│   ├── hipe_ops.md                   ← HiPE1 server deployment and monitoring
│   └── design/                       ← Internal design docs (archived for reference)
│       ├── phase5_7_plan.md
│       ├── phase5_report.md
│       ├── phase6_report.md
│       ├── phase7_cleanup_prd.md     ← This file
│       ├── session_handoff_phase6_complete.md
│       └── reports/                  ← Converted phase reports
│           ├── phase0_report.md      ← converted from .docx
│           ├── phase1_detection_report.md
│           ├── phase2_segmentation_report.md
│           ├── phase2_5_rfdetr_seg_report.md
│           └── figures/              ← figures extracted from .docx files
│
├── third_party/                      ← gitignored; populated by setup steps
│   └── OC_SORT/                      ← git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
│
└── _archive/                         ← gitignored; holding area during transition
    ├── README.md                     ← "These are archived artifacts. Safe to delete once pipeline verified."
    ├── planning/                     ← old planning docs from root level
    ├── third_party/                  ← original OC_SORT/ and sam2/ copies (before external install)
    ├── runs/                         ← seg_medium_lr1e4/, seg_medium_lr5e5/, tracking_experiments/
    └── weights/                      ← .tar.gz files before HuggingFace upload
```

---

## 4. Dataset Strategy

### 4.1 Current Datasets (Phases 1–6)

| Dataset | Size | Location | Access |
|---------|------|----------|--------|
| CBVD-5 | ~12 GB | `data/raw/cbvd5/` | [Official paper download link — document in docs/datasets.md] |
| CVB | ~15 GB | `data/raw/cvb/` | [Official paper download link — document in docs/datasets.md] |

Both are gitignored. `docs/datasets.md` must include:
- Full download instructions (official URLs)
- Expected directory structure after download
- File count and size verification commands

### 4.2 Phase 7 Additional Datasets

Additional cattle surveillance datasets will be added to `data/raw/` for analytics. When each is added:
1. Add a download section to `docs/datasets.md`
2. Add a `scripts/0X_prepare_{dataset}.sh` script (numbered to fit in the sequence)
3. Add the dataset's label-to-ID mapping to `data/label_map.json` (extending the 7-class taxonomy or noting incompatibilities)
4. Add a corresponding config in `configs/behavior/` if the dataset is used for behavior training

### 4.3 Preprocessing Pipeline Map

This is the end-to-end data flow that anyone cloning the repo needs to follow. **Every arrow below represents a numbered script.**

```
DOWNLOAD
  └─ data/raw/cbvd5/          (see docs/datasets.md for URL)
  └─ data/raw/cvb/

scripts/01_inspect_data.sh
  reads:   data/raw/{cbvd5,cvb}/
  writes:  console summary (no output files)
  purpose: verify download integrity, count files, check formats

scripts/02_prepare_cbvd5.sh   → src/data/convert_cbvd5.py
  reads:   data/raw/cbvd5/annotations/, data/raw/cbvd5/videos/
  writes:  data/processed/detection/cbvd5/{train,valid,test}/_annotations.coco.json + images
  purpose: convert AVA-format CSV annotations to COCO detection format

scripts/03_prepare_cvb.sh     → src/data/convert_cvb.py
  reads:   data/raw/cvb/annotations/, data/raw/cvb/raw_frames/
  writes:  data/processed/detection/cvb/{train,valid}/_annotations.coco.json + images
  purpose: convert CVB JSON annotations to COCO detection format

scripts/04_merge_datasets.sh  → src/data/merge_coco.py
  reads:   data/processed/detection/cbvd5/, data/processed/detection/cvb/
  writes:  data/processed/detection/combined/
  purpose: merge CBVD-5 and CVB into a single COCO dataset for joint training

scripts/05_train_detector.sh  → src/detection/train.py
  reads:   data/processed/detection/combined/ (or cbvd5/, cvb/)
            configs/detection/rfdetr_combined.yaml
            weights/rf-detr-medium.pth  ← download from HuggingFace first
  writes:  runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth
  runs on: GPU (RTX 3060 local or HiPE1 V100)
  NOTE:    This step is optional for a reader who only wants to use pretrained weights.
           Download runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth from HuggingFace.

scripts/06_run_detection.sh   → src/detection/infer_dataset.py
  reads:   data/raw/{cbvd5,cvb}/videos/, runs/detection/.../checkpoint_best_total.pth
  writes:  data/processed/tracking/{cbvd5,cvb}/{video_id}_detections.json
  purpose: run trained detector on all videos, save detection JSONs for tracking

scripts/07_run_segmentation.sh  → src/segmentation/segment.py
  reads:   data/processed/tracking/{cbvd5,cvb}/*_detections.json
            weights/sam2.1_hiera_large.pt  ← download separately
  writes:  data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json  (with mask_rle)
  runs on: GPU (memory-heavy; RTX 3060 or V100)
  NOTE:    Generates 242,689 masks. Takes several hours.

scripts/07_train_rfdetr_seg.sh  → scripts/hipe/train_rfdetr_seg_medium.py (via Docker)
  reads:   data/rfdetr_seg/cattle/  (SAM2 pseudo-labels from step above)
            weights/rf-detr-medium.pth
  writes:  runs/seg_medium_lr1e4/   (best: checkpoint_best_ema.pth)
  runs on: HiPE1 V100 via Docker
  NOTE:    Parallel to 07_run_segmentation.sh. Produces the distilled RF-DETR-Seg model.
            Download from HuggingFace to skip retraining.

scripts/08_run_tracking.sh   → src/tracking/track.py
  reads:   data/processed/tracking/{cbvd5,cvb}/*_detections.json
  writes:  data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json  (without masks)
  NOTE:    If segmentation step was run, tracking_v2 already exists with masks. This
            script runs OC-SORT separately if you skipped segmentation.

scripts/09_generate_tubelets.sh → src/data/export_tubelets.py
  reads:   data/processed/tracking_v2/{cbvd5,cvb}/
            data/raw/{cbvd5,cvb}/
  writes:  data/processed/tubelets/{cbvd5,cvb}/   (125,586 clips)
            data/processed/tubelets/labels.csv
  runs on: CPU (long-running, ~several hours)

scripts/10_train_behavior.sh → src/behavior/train.py (via Docker on HiPE1)
  reads:   data/processed/tubelets/, configs/behavior/videomae_combined.yaml
  writes:  runs/behavior/videomae_combined_v1/checkpoint_best.pt
  runs on: HiPE1 V100 via Docker (see docker/Dockerfile.behavior)
  NOTE:    Download trained checkpoints from HuggingFace to skip training.

scripts/11_evaluate.sh → src/behavior/evaluate.py
  reads:   data/processed/tubelets/, runs/behavior/.../checkpoint_best.pt
  writes:  results/behavior/predictions/*, results/behavior/confusion_matrices/
  runs on: GPU (or CPU, slow)

scripts/12_generate_analytics.sh → src/analytics/timeline.py + budget.py
  reads:   results/behavior/predictions/, data/processed/tracking_v2/
  writes:  results/analytics/timelines/, results/analytics/activity_budget.csv,
            results/analytics/transition_matrix.csv, results/analytics/behavior_deviation.csv
  runs on: CPU (fast)
```

### 4.4 "Quickstart" Path (Skip Training)

For a reader who wants to reproduce analytics without re-running the expensive GPU inference steps
(scripts 06–08 = detection inference + SAM2 segmentation + tracking, totaling several hours on a V100):

```bash
# 1. Download raw data (docs/datasets.md)
# 2. Run scripts 01-04 (data prep, no GPU needed, ~minutes)
# 3. Download pre-computed tracking_v2/ from HuggingFace dataset release:
#      huggingface-cli download sakifkhan/cattle-vision-data \
#        tracking_v2_cbvd5.tar.gz tracking_v2_cvb.tar.gz
#      tar -xf tracking_v2_cbvd5.tar.gz -C data/processed/tracking_v2/
#      tar -xf tracking_v2_cvb.tar.gz   -C data/processed/tracking_v2/
# 4. Run script 09 (tubelet export, CPU, ~several hours)
# 5. Download model weights from HuggingFace (see docs/setup.md)
# 6. Run scripts 11-12 (evaluate + analytics)
```

Note: Step 3 skips scripts 06–08 entirely. The tracking_v2/ data contains per-frame OC-SORT outputs
with mask RLEs for all CBVD-5 and CVB videos (cbvd5=20MB, cvb=641MB uncompressed).

Document this quickstart path prominently in README.md and docs/setup.md.

---

## 5. Docker Architecture

### 5.1 Design Principles

- One Dockerfile per pipeline stage group (not one per script)
- All images share the same base (`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`)
- `docker-compose.yml` mounts `data/` and `runs/` and `results/` as shared volumes
- Each service maps exactly to one or more numbered scripts
- Services can be run individually OR chained via `scripts/run_pipeline.sh`

### 5.2 Docker Images

| Image | Dockerfile | Stages covered | Key deps |
|-------|-----------|----------------|----------|
| `cattle-data` | `docker/Dockerfile.data` | 1–4, 12 | opencv, pycocotools, pandas, pyyaml |
| `cattle-detection` | `docker/Dockerfile.detection` | 5–6 | rfdetr, supervision, albumentations, tensorboard |
| `cattle-segmentation` | `docker/Dockerfile.segmentation` | 7a (SAM2 inference) | sam2, opencv |
| `cattle-rfdetr-seg` | `docker/Dockerfile.rfdetr_seg` | 7b (RF-DETR-Seg training) | rfdetr (same as detection image, can reuse) |
| `cattle-tracking` | `docker/Dockerfile.tracking` | 8 | ocsort, scipy, lap |
| `cattle-behavior` | `docker/Dockerfile.behavior` | 10–11 | transformers≥4.40, timm, scikit-learn |

Note: Stages 1–4, 9, and 12 can share one lightweight image (`cattle-data`) since they are CPU-only with standard deps.

### 5.3 docker-compose.yml Structure

```yaml
# docker/docker-compose.yml (conceptual — actual file written during Step C)
services:
  data:
    build: {context: .., dockerfile: docker/Dockerfile.data}
    volumes:
      - ../data:/workspace/data
      - ../results:/workspace/results
  detection:
    build: {context: .., dockerfile: docker/Dockerfile.detection}
    runtime: nvidia
    volumes:
      - ../data:/workspace/data
      - ../runs:/workspace/runs
      - ../configs:/workspace/configs:ro
      - ../src:/workspace/src:ro
      - ../weights:/workspace/weights:ro
  segmentation:
    build: {context: .., dockerfile: docker/Dockerfile.segmentation}
    runtime: nvidia
    volumes:
      - ../data:/workspace/data
      - ../weights:/workspace/weights:ro
  tracking:
    build: {context: .., dockerfile: docker/Dockerfile.tracking}
    volumes:
      - ../data:/workspace/data
  behavior:
    build: {context: .., dockerfile: docker/Dockerfile.behavior}
    runtime: nvidia
    volumes:
      - ../data:/workspace/data
      - ../runs:/workspace/runs
      - ../configs:/workspace/configs:ro
      - ../src:/workspace/src:ro
      - ../results:/workspace/results
```

### 5.4 End-to-End Pipeline Runner

`scripts/run_pipeline.sh` — a thin wrapper that runs stages in order:

```bash
# Usage:
#   ./scripts/run_pipeline.sh           # run all stages
#   ./scripts/run_pipeline.sh --from 5  # resume from stage 5
#   ./scripts/run_pipeline.sh --stage 11 # run just stage 11

# Each stage calls docker compose run <service> python3 src/... or bash scripts/N_*.sh
```

---

## 6. .gitignore Policy

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/

# Large data (download separately — see docs/datasets.md)
data/raw/
data/processed/
data/rfdetr_seg/
data/rfdetr_seg_eval/

# Model weights and checkpoints (download from HuggingFace — see docs/setup.md)
weights/
runs/
models/
*.pth
*.pt
*.tar.gz
*.ckpt

# Archive — safe to delete after verifying pipeline
_archive/

# Third-party source repos (populated by setup steps — see docs/setup.md)
third_party/

# Large result artifacts (keep summaries, not raw outputs)
# NOTE: results/behavior/predictions/ is COMMITTED (17 MB, needed for analytics)
results/analytics/timelines/
results/tracking/videos/
results/tracking/visualizations/
results/segmentation/viz/

# Notebook outputs
notebooks/.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

---

## 7. Documentation Plan

### 7.1 README.md (rewrite from empty)

```
1. Project Overview (2 paragraphs + pipeline diagram)
2. Key Results table — all phases
   - Detection: 70.4% mAP@50 cross-domain
   - Tracking: IDF1=67.31%, MOTA=36.61%
   - Behavior: all 5 VideoMAE configs (from f1_per_class.csv)
3. Quick Start — 3 paths:
   a. Full reproduction (download data + run scripts 01–12)
   b. Skip training (download pretrained weights from HuggingFace + run scripts 11–12)
   c. Just analytics (clone, download weights, run script 12)
4. Setup (conda env, external deps, weight downloads)
5. Datasets (CBVD-5 and CVB download + format overview)
6. Pipeline (numbered scripts with one-line description each)
7. Docker usage (stage-by-stage + full pipeline)
8. HuggingFace Model Hub (table of all weights with links)
9. Paper Citation (AIIoT26)
10. License
```

### 7.2 CLAUDE.md (rewrite for AI agent navigation)

Rewrite to contain exactly:

1. **Project overview** — what this produces (2 sentences)
2. **Repository map** — one-line purpose for every top-level directory
3. **Pipeline execution order** — scripts/01 through 12 with inputs and outputs
4. **External dependencies** — what is NOT in repo (OC-SORT, SAM2, HuggingFace weights)
5. **Frozen decisions** — label map, tubelet parameters, dataset splits, GPU constraints
6. **Data contracts** — key file formats (tracking JSON, labels.csv, predictions CSV)
7. **Key results** — all final numbers for quick reference
8. **Common gotchas** — CBVD-5 test=val, CVB frame presence threshold, class weight calc, Docker `--shm-size=16g` requirement, `conda` not in PATH by default
9. **HiPE1 server** — SSH alias, home dir `/home/zxs12/`, Docker eval pattern
10. **Current status** — Phase 7 analytics TODO, any open issues

### 7.3 docs/setup.md (new — detailed installation guide)

- Conda environment setup step by step
- External installs: `pip install sam2 @ git+...`, OC-SORT from source
- HuggingFace weight downloads with `huggingface-cli` one-liners
- SAM2 checkpoint download
- Dataset download links for CBVD-5 and CVB
- Expected directory structure after full setup
- Verification commands (`python -c "from src.data.label_utils import ..."`)

### 7.4 docs/pipeline.md (new — step-by-step guide)

For each of scripts 01–12:
- What it does (one paragraph)
- Prerequisites (what must exist before running)
- Exact command (copy-paste)
- Expected runtime
- Output verification command
- Notes / known issues

### 7.5 docs/datasets.md (new — replaces scattered dataset docs)

- CBVD-5: description, paper link, download URL, directory structure, annotation schema, splits, quirks (test=val)
- CVB: same
- Label map: 7-class taxonomy, cross-dataset string mappings (§3.2 from phase5_7_plan.md)
- Tubelet format: directory structure, frame naming, labels.csv schema, count table
- Phase 7 additional datasets: section reserved for future additions

### 7.6 docs/results.md (new — full results summary)

All final numbers in one place:
- Detection: mAP tables per dataset
- Segmentation + RF-DETR-Seg: coverage stats, AP metrics
- Tracking: IDF1, MOTA, MOTP per dataset
- Behavior: full 5-config comparison matrix
- Analytics: behavioral deviation summary (Phase 7, added when done)

### 7.7 docs/docker.md (new — Docker usage guide)

- How to build each image
- How to run a single stage via docker compose
- How to run the full pipeline
- GPU configuration
- Volume structure
- Troubleshooting (shm-size, OOM, etc.)

### 7.8 docs/hipe_ops.md (consolidate existing docs)

Merge from:
- `one_day/docs/` session handoff HiPE1 notes
- Root `docs/hipe_training_ops.md`
- Docker run commands from `phase5_7_plan.md` §6.8–6.9

### 7.9 docs/design/reports/ (.docx → Markdown conversion)

For each `.docx` file:
1. Convert with `pandoc input.docx -o output.md`
2. Review output — fix table formatting, heading levels
3. For each figure in the `.docx`: extract as PNG → `docs/design/reports/figures/`
4. Update figure references in the markdown to `![description](figures/filename.png)`
5. Audience: both human readers (thesis committee) and AI agents — preserve all technical detail

Files to convert:
- `docs/phase0_report.docx` → `docs/design/reports/phase0_report.md`
- `docs/phase_1_detection_report.docx` → `docs/design/reports/phase1_detection_report.md`
- `docs/phase_2_segmentation_report.docx` → `docs/design/reports/phase2_segmentation_report.md`
- `docs/phase_2_5_rfdetr_seg_final.docx` → `docs/design/reports/phase2_5_rfdetr_seg_report.md`
- `docs/reports_for_paper/cattle_vision_combined_report.docx` → `docs/design/reports/combined_report.md`
- `docs/reports_for_paper/phase_1_dataset_design_report.docx` → include in phase1 or separate
- `docs/reports_for_paper/phase_3b_rfdetr_seg_report.docx` → `docs/design/reports/phase3b_rfdetr_seg_report.md`
- `docs/reports_for_paper/phase_3_sam2_segmentation_report.docx` → `docs/design/reports/phase3_sam2_report.md`

Note: `phase_2_5_rfdetr_seg.docx` and `phase_2_5_rfdetr_seg_final.docx` appear to be draft + final. Convert only the final version.

### 7.10 logs/README.md (new, small)

One-line per log file describing which training run it came from and what to look for.

---

## 8. HuggingFace Hub Plan

Create HuggingFace repo: `sakifkhan/cattle-vision-framework`

### 8.1 Model Weights to Upload

| File | Source | Description |
|------|--------|-------------|
| `rf-detr-medium.pth` | `weights/rf-detr-medium.pth` | RF-DETR backbone (COCO pretrained, not our training) |
| `rf-detr-seg-medium.pt` | `weights/rf-detr-seg-medium.pt` | RF-DETR-Seg-Medium cattle-finetuned (Phase 3b, Config B ep59) |
| `rfdetr_combined_v1_best.pth` | `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth` | RF-DETR detector combined (Phase 1) |
| `videomae_combined_v1.pt` | `runs/behavior/videomae_combined_v1/checkpoint_best.pt` | Config 5, macro-F1=0.7537 |
| `videomae_cvb_v1.pt` | `runs/behavior/videomae_cvb_v1/checkpoint_best.pt` | Config 2, macro-F1=0.7607 (best) |
| `videomae_cbvd5_v1.pt` | `runs/behavior/videomae_cbvd5_v1/checkpoint_best.pt` | Config 1, macro-F1=0.3149 |
| `videomae_cbvd5_to_cvb_v1.pt` | `runs/behavior/videomae_cbvd5_to_cvb_v1/checkpoint_best.pt` | Config 3, macro-F1=0.1690 |
| `videomae_cvb_to_cbvd5_v1.pt` | `runs/behavior/videomae_cvb_to_cbvd5_v1/checkpoint_best.pt` | Config 4, macro-F1=0.1789 |

### 8.2 Pre-computed Intermediate Data

Upload to a separate HuggingFace dataset repo `sakifkhan/cattle-vision-data` to support the quickstart path
(see §4.4). Uploading tracking_v2 skips scripts 06–08 (several hours of GPU inference).

| File | Source | Size (uncompressed) | Notes |
|------|--------|---------------------|-------|
| `tracking_v2_cbvd5.tar.gz` | `data/processed/tracking_v2/cbvd5/` | 20 MB | Per-video OC-SORT + mask RLE JSONs for CBVD-5 |
| `tracking_v2_cvb.tar.gz` | `data/processed/tracking_v2/cvb/` | 641 MB | Per-video OC-SORT + mask RLE JSONs for CVB |

Do NOT upload `tracking_v2/cvb_mh5/` or `tracking_v2/cvb_mh7/` — these are experimental min_hits
variants not used by the main pipeline; move to `_archive/runs/tracking_experiments/` (Step A).

### 8.3 Model Cards

Each weight file needs a model card with:
- Architecture overview
- Training config YAML (inline or link)
- Validation metrics (macro-F1, per-class F1)
- Usage code snippet showing how to load and run inference

---

## 9. External Dependencies Documentation

### SAM2
- **Remove:** `one_day/models/sam2/` Python package (the installed package files); keep checkpoint
- **Install:** `pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'`
- **Checkpoint:** `wget <official_url> -O weights/sam2.1_hiera_large.pt`
- **Used by:** `src/segmentation/segment.py` (Phase 3a SAM2 inference)

### OC-SORT
- **Remove from working tree:** `one_day/OC_SORT/` → move to `_archive/third_party/OC_SORT/`
- **Setup step (user action):** `git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT`
- **`third_party/` is gitignored** — users populate it as part of the setup steps in `docs/setup.md`
- **Import path fix required in `src/tracking/track.py`:**
  - Current (line 24): `OCSORT_ROOT = Path(__file__).resolve().parents[2] / "OC_SORT"`
  - After flatten + move: `OCSORT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "OC_SORT"`
  - The import itself (`from trackers.ocsort_tracker.ocsort import OCSort`) stays unchanged — it uses the cloned source tree, not a pip package
- **Used by:** `src/tracking/track.py` (Phase 4 tracking)
- **Do during Step A:** update the path constant before moving OC_SORT/ so the existing pipeline stays runnable

---

## 10. Implementation Order

Execute steps in this order. Each step is labeled for session handoff tracking (§12).

### Step A — Archive (not delete) junk; free disk
**Safety rule: MOVE to `_archive/`, never delete. Verify downstream pipeline still works before any actual deletion.**

1. Create `_archive/` directory with a `README.md` explaining its purpose
2. **Fix import path in `src/tracking/track.py` BEFORE moving OC_SORT:**
   - Line 24: change `/ "OC_SORT"` → `/ "third_party" / "OC_SORT"`
   - This keeps the pipeline runnable immediately after the move
3. Move `one_day/OC_SORT/` → `_archive/third_party/OC_SORT/`
4. Create `third_party/OC_SORT/` by cloning: `git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT`
5. Move `sam2/` (root level) → `_archive/third_party/sam2/`
6. Move `one_day/runs/seg_medium_lr1e4/` → `_archive/runs/seg_medium_lr1e4/`
7. Move `one_day/runs/seg_medium_lr5e5/` → `_archive/runs/seg_medium_lr5e5/`
8. Move `one_day/runs/seg_medium_lr1e4_baseline/` → `_archive/runs/seg_medium_lr1e4_baseline/`
9. Move `one_day/data/processed/tracking_v2/cvb_mh5/` → `_archive/runs/tracking_experiments/cvb_mh5/`
10. Move `one_day/data/processed/tracking_v2/cvb_mh7/` → `_archive/runs/tracking_experiments/cvb_mh7/`
11. Move root-level planning docs → `_archive/planning/`
12. Move `one_day/cattle-rfdetr-seg-v1.tar.gz` and `one_day/cattle-videomae-v1.tar.gz` → `_archive/weights/`
13. Delete all `__pycache__/` directories
14. Delete root-level `.codex`
15. **Verify:** `python -c "from src.data.label_utils import BEHAVIOR_NAMES; print(BEHAVIOR_NAMES)"` still works
16. **Verify tracking import:** `python -c "from src.tracking.track import OCSort"` — should import cleanly from the cloned third_party/OC_SORT/

**Commit-safe metadata to extract BEFORE moving seg runs:**
- Copy `one_day/runs/seg_medium_lr1e4/{log.txt,results.json,run_config.json}` → `results/segmentation/seg_medium_lr1e4_metadata/`
- Copy `one_day/runs/seg_medium_lr5e5/{log.txt,results.json,run_config.json}` → `results/segmentation/seg_medium_lr5e5_metadata/`

### Step B — Upload weights and intermediate data to HuggingFace
1. Create HuggingFace model repo `sakifkhan/cattle-vision-framework`
2. Create HuggingFace dataset repo `sakifkhan/cattle-vision-data`
3. Install `huggingface-hub`: `pip install huggingface-hub`
4. Upload each weight file (see §8.1) to `sakifkhan/cattle-vision-framework` with `huggingface-cli upload`
5. Write model cards for each weight
6. Create tarballs for tracking_v2 and upload to `sakifkhan/cattle-vision-data`:
   ```bash
   tar -czf tracking_v2_cbvd5.tar.gz -C data/processed/tracking_v2 cbvd5/
   tar -czf tracking_v2_cvb.tar.gz   -C data/processed/tracking_v2 cvb/
   huggingface-cli upload sakifkhan/cattle-vision-data tracking_v2_cbvd5.tar.gz --repo-type dataset
   huggingface-cli upload sakifkhan/cattle-vision-data tracking_v2_cvb.tar.gz   --repo-type dataset
   ```
7. Test downloads: verify both weight and dataset downloads work
8. Update `docs/setup.md` with the exact download commands

### Step C — Create new directory structure
1. Create `docker/` directory; move Dockerfiles into it with new names
2. Write `docker/Dockerfile.data`, `docker/Dockerfile.segmentation`, `docker/Dockerfile.rfdetr_seg`, `docker/Dockerfile.tracking` (new images)
3. Write `docker/docker-compose.yml`
4. Create `scripts/hipe/`; move:
   - `train_hipe1.py` → `scripts/hipe/train_rfdetr_seg_medium.py`
   - `train_hipe2.py` → `scripts/hipe/train_rfdetr_seg_large.py`
   - `scripts/train_server.py` → `scripts/hipe/train_rfdetr_seg_parameterized.py`
5. Write `scripts/hipe/README.md`
6. Create `weights/` directory (empty, gitignored)
7. Rename `aiIoT_paper/` → `paper/`; create `paper/sample_images/` and move image files
8. Move `runs/kaggle_eval_full/` → `results/segmentation/kaggle_eval/`
9. Create `results/analytics/`, `results/behavior/training_logs/`
10. Move `runs/behavior/videomae_*/log.csv` → `results/behavior/training_logs/videomae_{name}.csv`
11. Write `scripts/run_pipeline.sh` (end-to-end runner)
12. Consolidate weight files: ensure `weights/rf-detr-medium.pth` exists (one copy from `one_day/` and root); delete duplicates

### Step D — Flatten root (promote one_day/ to repo root)
This is the most structural change. Do this in one atomic operation.

1. Verify `one_day/data/raw/` is complete (data_raw/ at root already deleted by user)
2. Move all items from `one_day/` up to `cattle-vision-framework/` root
3. Handle docs merge: `one_day/docs/` → merge with root `docs/`
4. Remove the now-empty `one_day/` directory
5. Verify all script invocations with `grep -r "one_day" scripts/` — update any hardcoded paths
6. **Verify imports:** `python -c "from src.data.label_utils import BEHAVIOR_NAMES"`

### Step E — Script audit + create/rewrite core files

**E.0 — Delete empty unused stubs (do this first):**
- Delete `src/tracking/track_utils.py` — 0 bytes, nothing imports it
- Delete `src/utils/io.py`, `src/utils/video.py`, `src/utils/perturbations.py`, `src/utils/metrics.py` — all 0 bytes, nothing imports them
- Keep `src/utils/__init__.py` (needed for package structure)
- Keep `src/analytics/timeline.py` and `budget.py` — intentional stubs for Step I
- Verify: `grep -r "track_utils\|from src.utils\." src/ scripts/` should return no hits after deletion

**E.1 — Write empty numbered pipeline scripts** (01, 08–12 are currently 0 bytes):
- `scripts/01_inspect_data.sh` — write: loop over `data/raw/{cbvd5,cvb}/`, print file counts and sizes
- `scripts/08_run_tracking.sh` — write: call `src/tracking/track.py` for cbvd5 and cvb
- `scripts/09_generate_tubelets.sh` — write: call `src/data/export_tubelets.py`
- `scripts/10_train_behavior.sh` — write: docker compose run behavior (or direct python call with config)
- `scripts/11_evaluate.sh` — write: call `src/behavior/evaluate.py` for all 5 configs
- `scripts/12_generate_analytics.sh` — write: call `src/analytics/timeline.py` + `budget.py`

**E.2 — Core files:**
1. Write `.gitignore` (see §6)
2. Write `README.md` (see §7.1)
3. Write `CLAUDE.md` (see §7.2)
4. Populate `requirements.txt` with pinned deps from `environment.yml`
5. Rename `env.yaml` → `environment.yml`

### Step F — Write new documentation
1. Write `docs/setup.md`
2. Write `docs/pipeline.md`
3. Write `docs/datasets.md`
4. Write `docs/results.md`
5. Write `docs/docker.md`
6. Write `docs/hipe_ops.md` (consolidate existing)
7. Write `logs/README.md`
8. Write `_archive/README.md`

### Step G — Convert .docx reports to Markdown
For each `.docx` listed in §7.9:
1. `pandoc input.docx -o output.md --extract-media=docs/design/reports/figures/`
2. Review and fix formatting
3. Update figure paths in markdown

### Step H — Git repository setup
1. `git init` in the new repo root
2. Add `.gitignore`
3. `git add` carefully — run `git status` and verify no large files are staged
4. Sanity check: `git ls-files | xargs du -sh 2>/dev/null | sort -rh | head -20`
5. Initial commit: "Initialize cattle-vision-framework repository (Phases 1–6 complete)"
6. Create GitHub repo: `cattle-vision-framework`
7. `git remote add origin <url>` and `git push -u origin main`

### Step I — Phase 7 analytics implementation
After repo is clean and pushed:
1. Implement `src/analytics/timeline.py` per `docs/design/phase5_7_plan.md` §7.1
2. Implement `src/analytics/budget.py` per thesis proposal §4.6 (behavioral deviation, NOT welfare flags —
   see §14 Closed Questions for rationale)
3. Update `scripts/12_generate_analytics.sh` to call both modules
4. Run analytics: `bash scripts/12_generate_analytics.sh`
5. Add additional datasets to `data/raw/` (when identified — see §13 Open Questions)
6. Commit `results/analytics/{activity_budget.csv, transition_matrix.csv, behavior_deviation.csv}`

---

## 11. Key Invariants to Preserve

1. **`src/` imports** — all use `from src.X.Y import ...` relative to repo root. After flattening, verify `__init__.py` files exist and Python path is correct.

2. **Config YAML paths** — relative to working directory (repo root). Paths like `data/processed/tubelets/labels.csv` remain correct after flattening since `one_day/` becomes root.

3. **`data/label_map.json`** — referenced by `src/data/label_utils.py`. Do not move.

4. **`results/` is committed** — do not add `results/` to `.gitignore`. Only gitignore specific large subdirectories.

5. **`results/behavior/predictions/` is committed** — user decision: these CSVs (~17 MB) are committed for analytics reproducibility.

6. **7-class label map** — frozen, no edits during cleanup.

7. **OC-SORT import path** — `src/tracking/track.py` line 24 uses `parents[2] / "OC_SORT"`. Update to `parents[2] / "third_party" / "OC_SORT"` during Step A (before moving the directory). The import statement itself (`from trackers.ocsort_tracker.ocsort import OCSort`) is unchanged — it relies on the cloned source tree.

8. **Hardcoded `_ONE_DAY` paths** — `src/data/export_tubelets.py` and `src/tracking/eval_tracking.py` reference tracking_v2 via a hardcoded `_ONE_DAY` path variable. After flattening (Step D), these paths will resolve correctly since `one_day/` becomes root — but must verify with `grep -r "_ONE_DAY\|one_day" src/` after Step D and fix any broken references.

---

## 12. Session Handoff Markers

Check off each step as it is completed. If a session ends mid-step, note the last completed sub-step here.

- [x] Step A — Archive junk; extract metadata from seg runs
- [x] Step B — HuggingFace upload
- [x] Step C — New directory structure (docker/, scripts/hipe/, weights/, paper/)
- [x] Step D — Flatten root (promote one_day/ to repo root)
- [x] Step E — Script audit (delete empty stubs, write empty numbered scripts) + core files (.gitignore, README.md, CLAUDE.md, requirements.txt, environment.yml)
- [x] Step F — New documentation (setup.md, pipeline.md, datasets.md, results.md, docker.md, hipe_ops.md)
- [x] Step G — .docx → Markdown conversion
- [x] Step H — Git init, initial commit, GitHub push (https://github.com/SakifKhan98/cattle-vision-framework)
- [ ] Step I — Phase 7 analytics (timeline.py, budget.py, additional datasets, script 12)

---

## 13. Open Questions (Remaining)

1. **Additional Phase 7 datasets** — the specific datasets have not been identified yet. This blocks part of Step I. Will be defined when Phase 7 analytics implementation begins.

---

## 14. Closed Questions (Resolved)

**OC-SORT import in `src/tracking/track.py`** — RESOLVED. File uses `sys.path.insert` with `parents[2] / "OC_SORT"` and `from trackers.ocsort_tracker.ocsort import OCSort`. Cannot use pip package (different import path). Solution: clone official repo to `third_party/OC_SORT/` (gitignored); update path in track.py line 24. See §9 and Step A.

**"Welfare flags" → "behavioral deviation analysis"** — RESOLVED. The earlier phase5_7_plan.md §7.2
used clinical welfare thresholds (< 8 hrs/day lying, < 4 hrs foraging, < 2 drinking events) and called
the output `welfare_flags.csv`. The approved thesis proposal (§4.6.3) uses the softer framing
"behavioral deviation analysis" — deviations from dataset-specific baselines, without hard clinical
thresholds. This is the correct framing for the committee. Output renamed to `behavior_deviation.csv`.
`budget.py` should compute deviation from per-dataset median baselines, not apply fixed clinical rules.

**`results/behavior/predictions/` format** — RESOLVED. Schema: `dataset, video_id, tubelet_dir, start_frame, end_frame, label_id, pred_label_id, logit_0..6`. Note: `track_id` is NOT an explicit column — it is the last component of `tubelet_dir` (e.g., `data/processed/tubelets/cbvd5/341/kf6_instc85ac7`). `timeline.py` must parse `track_id` from this path. `video_id` is a numeric folder name (e.g., `341`).

**GitHub Actions CI** — RESOLVED. Not needed. Commit and push manually for now.

**Redundant seg run metadata** — RESOLVED. Commit metadata from both `seg_medium_lr1e4` and `seg_medium_lr5e5` runs to `results/segmentation/`. Thesis mentions both experiments.

**tracking_v2/cvb_mh5 and cvb_mh7** — RESOLVED. Experimental min_hits variants not used by main pipeline. Move to `_archive/runs/tracking_experiments/`. Upload only `tracking_v2/cbvd5` and `tracking_v2/cvb` to HuggingFace (§8.2).
