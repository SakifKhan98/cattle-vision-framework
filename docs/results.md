# Results Summary

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

All final results across Phases 1–6. Per-class files are in `results/`.

---

## Phase 1 — Object Detection (RF-DETR)

Model: RF-DETR-Medium fine-tuned on combined CBVD-5 + CVB detection dataset.
Checkpoint: `runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth`

| Evaluation | mAP@50 | Notes |
|------------|--------|-------|
| CBVD-5 in-domain (test=val) | see `results/detection/cbvd5_test_ap.json` | |
| CVB in-domain (val) | see `results/detection/cvb_test_ap.json` | |
| Cross-domain (combined OOD) | **70.4%** | primary reported metric |

Full per-class breakdown: `results/detection/combined_ood.json`

---

## Phase 2 — Detection Inference

Script 06 runs the detector on all videos to generate tracking inputs.
No standalone metrics — output quality feeds into tracking evaluation.

---

## Phase 3a — SAM2 Segmentation

Model: SAM2.1-Hiera-Large (off-the-shelf, no fine-tuning)
Reprompt interval: every 15 frames

| Dataset | Videos | Masks generated | Coverage rate | Runtime |
|---------|--------|-----------------|---------------|---------|
| CBVD-5 | 687 | 15,900 | 100% | 27.8 min |
| CVB | 502 | 226,789 | 100% | 329 min |

Total masks: **242,689**
Full stats: `results/segmentation/{cbvd5,cvb}_summary.json`

---

## Phase 3b — RF-DETR-Seg Distillation

Model: RF-DETR-Seg-Medium fine-tuned on SAM2 pseudo-labels (Config B, lr=5e-5, epoch 59).
Checkpoint: `weights/rf-detr-seg-medium.pt`

Evaluation: `results/segmentation/kaggle_eval/`
Training metadata: `results/segmentation/seg_medium_lr1e4_metadata/`, `results/segmentation/seg_medium_lr5e5_metadata/`

---

## Phase 4 — OC-SORT Multi-Object Tracking

Tracker: OC-SORT, IoU threshold 0.5, evaluated on all 447 videos across both datasets.

| Metric | Value |
|--------|-------|
| Videos evaluated | 447 |
| MOTA | **36.61%** |
| MOTP | 77.42% |
| IDF1 | **67.31%** |
| Precision | 65.69% |
| Recall | 77.41% |
| Total ground truth tracks | 38,609 |
| Total predicted tracks | 45,499 |
| True positives | 29,887 |
| ID switches | 141 (avg 0.32 per video) |

Full per-video breakdown: `results/tracking/tracking_per_video_all.csv`
Summary JSON: `results/tracking/tracking_summary_all.json`

---

## Phase 5 — Tubelet Generation

125,586 tubelet clips extracted (16 frames each, 224×224 px).
Labels: `data/processed/tubelets/labels.csv`

---

## Phase 6 — VideoMAE Behavior Classification

Model: VideoMAE-Base fine-tuned on 16-frame tubelet clips. 5 training configurations.
Primary metric: **macro-F1** (robust to class imbalance).

### Config Summary

| Config | Train set | Eval set | **Macro-F1** |
|--------|-----------|----------|-------------|
| 1 | CBVD-5 | CBVD-5 (val) | 0.3149 |
| 2 | CVB | CVB (val) | **0.7607** |
| 3 | CBVD-5 | CVB (OOD) | 0.1690 |
| 4 | CVB | CBVD-5 (OOD) | 0.1789 |
| 5 | CBVD-5 + CVB | both (val) | **0.7537** |

### Per-Class F1 — Config 5 (Combined, Best for Analytics)

| Class | F1 | Precision | Recall |
|-------|----|-----------|--------|
| Standing | 0.870 | 0.891 | 0.850 |
| Lying | 0.823 | 0.847 | 0.800 |
| Foraging | 0.980 | 0.979 | 0.982 |
| Drinking | 0.876 | 0.827 | 0.931 |
| Ruminating | 0.772 | 0.772 | 0.771 |
| Grooming | 0.722 | 0.799 | 0.659 |
| Other | 0.233 | 0.184 | 0.315 |
| **Macro** | **0.754** | | |

### Per-Class F1 — Config 2 (CVB In-Domain, Highest F1)

| Class | F1 | Precision | Recall |
|-------|----|-----------|--------|
| Standing | 0.860 | 0.828 | 0.895 |
| Lying | 0.845 | 0.880 | 0.813 |
| Foraging | 0.979 | 0.984 | 0.974 |
| Drinking | 0.881 | 0.896 | 0.866 |
| Ruminating | 0.801 | 0.786 | 0.817 |
| Grooming | 0.715 | 0.830 | 0.628 |
| Other | 0.243 | 0.208 | 0.293 |
| **Macro** | **0.761** | | |

### Per-Class F1 — Config 1 (CBVD-5 In-Domain)

| Class | F1 | Notes |
|-------|----|-------|
| Standing | 0.906 | |
| Lying | 0.303 | |
| Foraging | 0.896 | |
| Drinking | 0.000 | No Drinking samples in CBVD-5 val set |
| Ruminating | 0.100 | |
| Grooming | 0.000 | CBVD-5 has no Grooming class |
| Other | 0.000 | CBVD-5 has no Other class |
| **Macro** | **0.315** | 7-class denominator (penalizes absent classes) |

**Thesis note:** For Config 1, report 5-class macro-F1 (IDs 0–4 only) to be fair to
the CBVD-5 dataset, which lacks Grooming and Other annotations.

### OOD (Cross-Domain) Results

| Config | Macro-F1 | Key observation |
|--------|----------|-----------------|
| 3: CBVD-5 → CVB | 0.1690 | Large domain gap (indoor barn → outdoor farm) |
| 4: CVB → CBVD-5 | 0.1789 | Same domain gap in reverse |

OOD results near chance-level are a key thesis finding: the domain gap between
indoor surveillance (CBVD-5) and outdoor surveillance (CVB) is severe for behavior recognition.

Full CSV: `results/behavior/f1_per_class.csv`
Confusion matrices: `results/behavior/confusion_matrices/`
Training logs: `results/behavior/training_logs/`
Prediction CSVs: `results/behavior/predictions/`

---

## Phase 7 — Analytics (In Progress)

Outputs will be written to `results/analytics/` after Phase 7 implementation (Step I).

| Output | Description |
|--------|-------------|
| `activity_budget.csv` | Per-animal and herd-level time-budget percentages per behavior |
| `transition_matrix.csv` | Behavior transition probabilities between consecutive states |
| `welfare_flags.csv` | Alerts for anomalous behavior patterns (e.g., prolonged inactivity) |
| `timelines/` | Per-video per-track behavior timelines (gitignored, large) |

---

## Generalization

Cross-domain detection generalization results: `results/generalization/ood_summary.csv`
Perturbation robustness: `results/generalization/perturbation_delta.csv`
