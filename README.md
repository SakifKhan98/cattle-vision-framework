# Cattle Vision Framework

MS thesis project for multi-behavior recognition in dairy cattle from surveillance video.
The pipeline runs RF-DETR object detection → SAM2 segmentation → OC-SORT tracking →
VideoMAE video classification → activity-budget analytics across two public datasets
(CBVD-5 and CVB), recognizing 7 behaviors at the individual animal level.

**Texas State University · Sakif Khan · 2026**

---

## Key Results

| Stage                                      | Metric                               | Value      |
| ------------------------------------------ | ------------------------------------ | ---------- |
| Detection                                  | mAP@50 cross-domain (combined model) | 70.4%      |
| Tracking                                   | IDF1                                 | 67.31%     |
| Tracking                                   | MOTA                                 | 36.61%     |
| Behavior — Config 1: CBVD-5 in-domain      | macro-F1                             | 0.3149     |
| Behavior — Config 2: CVB in-domain         | macro-F1                             | 0.7607     |
| Behavior — Config 3: CBVD-5 → CVB transfer | macro-F1                             | 0.1690     |
| Behavior — Config 4: CVB → CBVD-5 transfer | macro-F1                             | 0.1789     |
| Behavior — Config 5: combined training     | macro-F1                             | **0.7537** |

Full results: `results/behavior/f1_per_class.csv`, `results/behavior/summary_table.csv`.

---

## Quick Start

Three paths depending on your goal:

### Path A — Full reproduction (train everything from scratch)

```bash
# 1. Install environment
conda env create -f environment.yml
conda activate cattletransformer

# 2. Install external deps (OC-SORT + SAM2)
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'

# 3. Download datasets (see docs/datasets.md for URLs)
#    → data/raw/cbvd5/   and   data/raw/cvb/

# 4. Download backbone weight
huggingface-cli download sakifkhan/cattle-vision-framework \
    rf-detr-medium.pth --local-dir weights/

# 5. Run the full pipeline
bash scripts/01_inspect_data.sh
bash scripts/02_prepare_cbvd5.sh
bash scripts/03_prepare_cvb.sh
bash scripts/04_merge_datasets.sh
bash scripts/05_train_detector.sh          # GPU, ~8h on RTX 3060
bash scripts/06_run_detection.sh           # GPU, ~2h
bash scripts/07_run_segmentation.sh        # GPU, several hours
bash scripts/09_generate_tubelets.sh       # CPU, several hours
bash scripts/10_train_behavior.sh          # GPU, ~12h per config on HiPE1
bash scripts/11_evaluate.sh
bash scripts/12_generate_analytics.sh
```

### Path B — Skip GPU training (use pretrained weights)

```bash
# After steps 1–3 from Path A:
# Download pre-trained checkpoints
huggingface-cli download sakifkhan/cattle-vision-framework \
    rfdetr_combined_v1_best.pth videomae_combined_v1.pt \
    --local-dir weights/

# Download pre-computed tracking_v2 (skips scripts 06–08)
huggingface-cli download sakifkhan/cattle-vision-data \
    tracking_v2_cbvd5.tar.gz tracking_v2_cvb.tar.gz --repo-type dataset
tar -xf tracking_v2_cbvd5.tar.gz -C data/processed/tracking_v2/
tar -xf tracking_v2_cvb.tar.gz   -C data/processed/tracking_v2/

bash scripts/09_generate_tubelets.sh
bash scripts/11_evaluate.sh
bash scripts/12_generate_analytics.sh
```

### Path C — Analytics only (predictions already committed)

```bash
# Predictions are committed at results/behavior/predictions/
# Install environment (Path A step 1) then:
bash scripts/12_generate_analytics.sh
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate cattletransformer

# External dependencies (not pip-installable)
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
pip install 'sam2 @ git+https://github.com/facebookresearch/sam2.git'

# Download SAM2 checkpoint
# Follow: https://github.com/facebookresearch/sam2#model-checkpoints
# Place at: weights/sam2.1_hiera_large.pt
```

See `docs/setup.md` for detailed step-by-step instructions.

---

## Datasets

| Dataset | Size   | Domain          | Behaviors   | Format         |
| ------- | ------ | --------------- | ----------- | -------------- |
| CBVD-5  | ~12 GB | Indoor barn     | 5 (IDs 0–4) | AVA-format CSV |
| CVB     | ~15 GB | Outdoor pasture | 7 (IDs 0–6) | COCO JSON      |

Both datasets are gitignored. Download links and directory structure: `docs/datasets.md`.

---

## Pipeline

The pipeline consists of 12 numbered scripts. Each reads from the previous step's output.

| Script                     | Stage                           | Runtime            | GPU |
| -------------------------- | ------------------------------- | ------------------ | --- |
| `01_inspect_data.sh`       | Verify downloads                | ~1 min             | No  |
| `02_prepare_cbvd5.sh`      | Convert CBVD-5 to COCO          | ~5 min             | No  |
| `03_prepare_cvb.sh`        | Convert CVB to COCO             | ~5 min             | No  |
| `04_merge_datasets.sh`     | Merge into combined dataset     | ~2 min             | No  |
| `05_train_detector.sh`     | Train RF-DETR detector          | ~8h (RTX 3060)     | Yes |
| `06_run_detection.sh`      | Run detector on all videos      | ~2h                | Yes |
| `07_run_segmentation.sh`   | SAM2 segmentation + OC-SORT     | Several hours      | Yes |
| `07_train_rfdetr_seg.sh`   | Train RF-DETR-Seg (HiPE1)       | ~24h (V100)        | Yes |
| `08_run_tracking.sh`       | Box-only OC-SORT (if skip 07)   | ~1h                | No  |
| `09_generate_tubelets.sh`  | Extract 125k tubelet clips      | Several hours      | No  |
| `10_train_behavior.sh`     | Train VideoMAE classifier       | ~12h/config (V100) | Yes |
| `11_evaluate.sh`           | Evaluate all 5 configs          | ~1h                | Yes |
| `12_generate_analytics.sh` | Activity budget + welfare flags | ~5 min             | No  |

See `docs/pipeline.md` for per-script prerequisites, commands, and expected outputs.

---

## Docker

Each pipeline stage group has a Dockerfile. Use Docker Compose to run stages:

```bash
# Single stage
docker compose -f docker/docker-compose.yml run detection bash scripts/05_train_detector.sh

# Full pipeline
bash scripts/run_pipeline.sh
```

See `docs/docker.md` for full Docker usage including GPU configuration.

---

## HuggingFace Model Hub

All weights are at `sakifkhan/cattle-vision-framework`. Pre-computed data at `sakifkhan/cattle-vision-data`.

| Weight                        | Description                       | macro-F1 |
| ----------------------------- | --------------------------------- | -------- |
| `rfdetr_combined_v1_best.pth` | RF-DETR detector (combined)       | —        |
| `rf-detr-seg-medium.pt`       | RF-DETR-Seg (distilled from SAM2) | —        |
| `videomae_combined_v1.pt`     | Config 5: combined training       | 0.7537   |
| `videomae_cvb_v1.pt`          | Config 2: CVB in-domain           | 0.7607   |
| `videomae_cbvd5_v1.pt`        | Config 1: CBVD-5 in-domain        | 0.3149   |
| `videomae_cbvd5_to_cvb_v1.pt` | Config 3: CBVD-5→CVB transfer     | 0.1690   |
| `videomae_cvb_to_cbvd5_v1.pt` | Config 4: CVB→CBVD-5 transfer     | 0.1789   |

---

## Citation

If you use this work, please cite:

<!-- ```bibtex
@inproceedings{khan2026cattlevision,
  title={Multi-Behavior Recognition in Dairy Cattle from Surveillance Video},
  author={Khan, Sakif},
  booktitle={IEEE AIIoT 2026},
  year={2026}
}
``` -->

---

## License

MIT — see `LICENSE`.
