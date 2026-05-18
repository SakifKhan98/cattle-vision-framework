# Cattle Vision Framework — Domain Context

## Glossary

### Pipeline stages

**Detection** — RF-DETR inference on raw video frames, producing per-frame bounding boxes. Outputs to `data/processed/tracking/{dataset}/`.

**Segmentation** — Instance segmentation on detected boxes. Two paths exist: SAM2 (Phase 1–7, canonical) and RF-DETR-Seg (Phase 3b distillation experiment). Outputs to `data/processed/tracking_v2/` (SAM2) or `data/processed/segmentation_rfdetr/` (RF-DETR-Seg).

**Tracking** — OC-SORT multi-object tracking producing `_tracks.json` files with per-frame bbox, mask RLE, and track IDs. Output to `data/processed/tracking_v2/` or `data/processed/tracking_v2_rfdetr/`.

**Tubelet** — A fixed-length clip (16 frames, stride 4, 224×224 px) extracted around a single tracked animal, used as input to VideoMAE behavior classification.

**Behavior classification** — VideoMAE inference on tubelets, predicting one of 7 behavior classes per tubelet.

**Activity budget** — Per-animal or per-herd time distribution across the 7 behavior classes, derived from behavior predictions over a full video.

**Behavioral deviation** — Difference between an individual animal's activity budget and the herd baseline. Used as a welfare indicator.

### Evaluation types

**In-domain** — Train and evaluate on the same dataset (e.g., CBVD-5 train → CBVD-5 val).

**Cross-domain** — Train on one dataset, evaluate on another (e.g., CVB → CBVD-5). Tests generalization.

**Zero-shot OOD** — Apply a frozen trained model to a dataset never seen during training (e.g., combined VideoMAE model on Freeman Center).

**Descriptive evaluation** — OOD results reported as observational findings, not performance benchmarks, because ground truth label mapping is approximate.

### Dataset roles

**CBVD-5** — Indoor dairy surveillance, 5 behavior classes (IDs 0–4), CBVD-5 annotation format. Already fully processed through Phase 7.

**CVB** — Outdoor dairy video, 7 behavior classes (IDs 0–6), Supervisely-style string labels. Already fully processed through Phase 7.

**Freeman Center (freeman-cmb-2024)** — Pasture clips (55 videos, 5–30 sec each), YOLO format, 9 behavior classes. Texas State University IACUC-approved. Used for zero-shot OOD evaluation (Phase 8). Raw videos at `data/raw/freeman-cmb-2024/freeman-raw-videos/`.

**Cows2021** — Detection and re-ID dataset, DatasetNinja format (Supervisely JSON). No behavior labels. Used for detection mAP + tracking IDF1 evaluation only (Phase 8).

**OpenCows2020** — Detection and re-ID dataset, DatasetNinja format. No behavior labels. Used for detection mAP evaluation only (Phase 8).

**CattleEyeView** — Detection dataset (downloading). Evaluation scope: detection + Mask IoU + IDF1 (Phase 8).

### Label mapping

**7-class taxonomy** — The canonical behavior label set used throughout the pipeline:

| ID | Behavior         | Datasets    |
|----|------------------|-------------|
| 0  | Standing         | CBVD-5, CVB |
| 1  | Lying            | CBVD-5, CVB |
| 2  | Foraging/Grazing | CBVD-5, CVB |
| 3  | Drinking         | CBVD-5, CVB |
| 4  | Ruminating       | CBVD-5, CVB |
| 5  | Grooming         | CVB only    |
| 6  | Other            | CVB only    |

**Freeman Center label mapping** (partially resolved — see open decisions):

| Freeman class     | Maps to                  | Status      |
|-------------------|--------------------------|-------------|
| hay feeding       | Foraging/Grazing (2)     | Resolved    |
| grazing           | Foraging/Grazing (2)     | Resolved    |
| normal            | TBD                      | **Deferred** — decide after class distribution analysis |
| dominance assertion | Other (6)              | Tentative   |
| ruminating        | Ruminating (4)           | Resolved    |
| fear response     | Other (6)                | Tentative   |
| grooming          | Grooming (5)             | Resolved    |
| vocalizing        | Other (6)                | Tentative   |
| sniffing          | Other (6)                | Tentative   |

### Notebook types

**Per-dataset analysis notebook** — One notebook per dataset (`notebooks/analysis_{dataset}.ipynb`). Contains: dataset provenance, data inventory, annotation analysis, resolution/quality, label mapping to 7-class taxonomy, preprocessing gap analysis. Independently runnable. Thesis-citeable.

**Dataset comparison notebook** — `notebooks/dataset_comparison.ipynb`. Imports summary stats from per-dataset notebooks, produces cross-dataset comparative figures for the thesis.

### Script naming convention

**`XX_*.sh`** — Canonical Phase 1–7 SAM2 pipeline scripts.

**`XXb_*.sh`** — RF-DETR-Seg parallel pipeline scripts (Phase 3b re-run). Use `_rfdetr` data directories, never overwrite SAM2 outputs.

## Open decisions

- **Freeman Center "normal" class mapping** — defer until `notebooks/analysis_freeman.ipynb` class distribution is complete. Options: map to Standing (0) or drop from evaluation.
- **Tentative Freeman Center label mappings** — dominance assertion, fear response, vocalizing, sniffing → Other (6). Confirm against paper during notebook analysis.
