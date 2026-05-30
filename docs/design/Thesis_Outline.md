# A TRANSFORMER-BASED PERCEPTION AND ANALYTICS FRAMEWORK FOR CATTLE BEHAVIOR ANALYSIS ACROSS DIVERSE RANCH ENVIRONMENTS

*Thesis Document Outline (Revised)*

by **Md Sakif Uddin Khan**

**Committee Members:**
Dr. Damian Valles Molina (Chair)
Dr. Bahram Asiabanpour
Dr. Merritt Drewery

---

## Front Matter

- Title Page
- Copyright
- Dedication
- Acknowledgments
- Table of Contents
- List of Tables
- List of Figures
- List of Abbreviations
- Abstract

---

# I. Introduction

## 1.1 Background and Motivation

## 1.2 Challenges in Vision-Based Cattle Behavior Analysis

## 1.3 Problem Statement

## 1.4 Research Objectives

> **Note:** Research contributions to be stated here (moved from former Ch. VI per advisor feedback).

---

# II. Background

## 2.1 Cattle Behavior and Vision-Based Monitoring

## 2.2 Computer Vision Pipelines for Animal Monitoring

## 2.3 Evolution of Learning-Based Vision Models *(New)*

> **Note:** Cover ML vs. DL distinction and CNN limitations as motivation for transformer architectures.

### 2.3.1 Emergence of Transformers in Vision

## 2.4 Transformers in Vision Systems

## 2.5 Knowledge Distillation in Vision Models *(New)*

> **Note:** Covers teacher-student distillation theory — directly motivates the SAM2.1 pseudo-label training approach used in Chapter V.

---

# III. Literature Review

## 3.1 Vision-Based Cattle Monitoring Systems

## 3.2 Identity Preservation and Instance Segmentation

## 3.3 Behavior Recognition in Livestock

## 3.4 Transformer-Based Models in Animal Vision

## 3.5 Generalization and Domain Shift

## 3.6 Summary of Research Gaps

---

# IV. Dataset

## 4.1 Overview of Data Sources

## 4.2 Selected Datasets

### 4.2.1 CBVD-5: Indoor Barn Dataset

### 4.2.2 CVB: Outdoor Paddock Dataset

### 4.2.3 Kaggle Cow Segmentation Dataset (Cross-Domain Evaluation)

## 4.3 Behavior Definition, Selection, and Harmonization Strategy

### 4.3.1 Behavior Sources and Eligibility

### 4.3.2 Core Behavior Set

### 4.3.3 Auxiliary and Residual Behaviors

### 4.3.4 Harmonization Rules

### 4.3.5 Temporal Granularity Considerations

## 4.4 Dataset Harmonization

## 4.5 Annotation Strategy

---

# V. Methodology

## 5.1 System Overview

## 5.2 Transformer-Based Perception Pipeline

### 5.2.1 Detection and Instance Segmentation Module

#### 5.2.1.1 RF-DETR-Seg-Medium Architecture

#### 5.2.1.2 SAM2.1 Teacher-Student Pseudo-Label Training

#### 5.2.1.3 Training Configuration: Config A vs. Config B

#### 5.2.1.4 EMA Checkpoint Selection (Config B, Epoch 59)

#### 5.2.1.5 Training Metric Disclosure: Pseudo-Label Fidelity vs. Ground-Truth Accuracy

### 5.2.2 Multi-Object Tracking

#### 5.2.2.1 OC-SORT Algorithm

#### 5.2.2.2 Integration Approach and Pipeline Connection

## 5.3 Spatiotemporal Behavior Recognition *(New)*

### 5.3.1 Tubelet Generation from Identity-Preserved Tracks

### 5.3.2 Behavior Recognition Model: VideoMAE

## 5.4 Generalization and Robustness Evaluation *(New)*

### 5.4.1 In-Domain vs. Out-of-Distribution Setup

### 5.4.2 Controlled Environmental Perturbations

The detection stage was evaluated under five classes of synthetic environmental
perturbation — brightness reduction, Gaussian sensor noise, motion blur, synthetic fog,
and synthetic rain — at two severity levels each, across four OOD datasets and the CBVD-5
in-domain test set (CVB excluded — floor-effect baseline of 5.67% mAP50 renders deltas
uninterpretable). Perturbations were applied in-memory using albumentations before RF-DETR
inference; confidence threshold held fixed at 0.3 to match all prior evaluations.

Brightness reduction was the dominant vulnerability across all datasets. Reducing
brightness to 50% caused mAP50 drops of 14–59 pp; reducing to 25% caused drops of 26–72
pp, representing 81–99% relative collapses. Stronger clean baselines did not confer
protection — Freeman Center (72.98% clean) lost 72 pp absolute at high severity, the
largest absolute loss of any dataset. Synthetic fog was second (7–36 pp at high severity),
followed by rain (3–11 pp). Gaussian noise and motion blur were the most robust classes:
noise caused < 11 pp even at high severity and produced slight improvements on two
datasets; motion blur caused < 8 pp universally and marginal improvements on two datasets.
These robustness properties emerged without augmentation-based hardening during training.

The in-domain CBVD-5 dataset showed amplified fog sensitivity (high fog −24 pp, 53%
relative) compared to OOD datasets, suggesting texture cues specific to indoor barn
imaging are particularly susceptible to contrast reduction.

Full results: `results/generalization/perturbation_delta.csv` (60 rows).
Detailed analysis: `docs/design/reports/phase9_perturbation_report.md`.

### 5.4.3 Evaluation Metrics

#### 5.4.3.1 Detection Metrics

#### 5.4.3.2 Instance Segmentation Metrics

#### 5.4.3.3 Multi-Object Tracking Metrics

#### 5.4.3.4 Behavior Recognition Metrics

#### 5.4.3.5 System-Level Evaluation

## 5.5 Computational Resources and Software Environment

### 5.5.1 Hardware Platform

### 5.5.2 Software Stack and Primary Libraries

### 5.5.3 Environment Setup and Working Plan

---

# VI. Results and Discussion

> **Note:** Results sections mirror Methodology sections 1:1 per advisor feedback (synchronized structure).

## 6.1 Detection and Instance Segmentation Results

### 6.1.1 In-Domain Performance on CBVD-5 and CVB

### 6.1.2 Cross-Domain Evaluation: Kaggle Cow Segmentation Dataset

### 6.1.3 Config A vs. Config B Training Analysis

## 6.2 Identity-Preserving Tracking Results

### 6.2.1 Quantitative Results: IDF1, MOTA, MOTP, and IDSW

### 6.2.2 Hyperparameter Ablation: min_hits Study

### 6.2.3 MOTA Sensitivity Analysis: Upstream Detection False Positives

### 6.2.4 Qualitative Track Visualization

## 6.3 Spatiotemporal Behavior Recognition Results

### 6.3.1 Tubelet Quality and Track-Based Input Analysis

### 6.3.2 VideoMAE Fine-Tuning Results on CBVD-5

### 6.3.3 Cross-Dataset Behavior Evaluation on CVB

### 6.3.4 Behavior Timeline and Activity Budget Output Examples

## 6.4 Discussion

### 6.4.1 Generalization Across Indoor and Outdoor Environments

### 6.4.2 Pipeline Error Propagation Analysis

Error propagation is traced across four layers of the pipeline:

**Detection layer.** The detector operates at threshold=0.3, deliberately favouring
recall over precision. On CBVD-5 (test=val, note: no separate test split exists),
mAP50=45.91% with AR@100=24.28%. On CVB, mAP50=5.67% with AR@100=4.31%, reflecting
severe domain shift — the detector was checkpoint-selected on CBVD-5 validation only.
In the canonical tracking evaluation on CVB (447 videos), 15,612 FP detections entered
the tracker against 8,722 FN, with recall=77.41% and precision=65.69%.

**Tracking layer.** IDF1=67.31% is the primary metric for downstream behavior because it
measures identity continuity rather than frame-level accuracy. MOTA=36.61% is suppressed
by the FP count (15,612) and is not the relevant signal for tubelet quality. The
min_hits=3 ablation sweep (min_hits ∈ {1, 2, 3, 5}) showed IDF1, MOTA, and ID switches
invariant across all values (IDF1=67.31%, IDS=141 for all four), confirming that the FP
source is multi-frame persistent detector activations (fence posts, shadows) rather than
single-frame spurious detections. The min_hits gate cannot reduce these; they are filtered
at tubelet label assignment (Phase 5) via Hungarian matching with IoU ≥ 0.3.

**Behavior layer.** Cross-domain configs score 3–4× lower than in-domain: CBVD-5→CVB
macro-F1=0.172 (v2), CVB→CBVD-5=0.225 (v2), vs. CBVD-5 in-domain=0.451 and CVB
in-domain=0.777. The domain gap originating at the detection stage (5.67% CVB mAP50)
propagates through tracking fragmentation into behavior window contamination — low-quality
tubelets from missed or misidentified cattle degrade the VideoMAE input distribution.
Note: CBVD-5 test=val (no separate test split released); all CBVD-5 figures use the
validation split as test.

**Perturbation sensitivity layer.** Brightness degradation compounds the domain-shift
penalty: datasets with larger OOD gap already start at lower clean mAP50, and under
high-severity brightness perturbation all datasets collapse toward near-zero detection
(0.99%–4.5%). Fog represents a practical outdoor risk (Freeman: −36 pp at high severity).
Rain, noise, and blur are operationally benign (< 11 pp at high severity across all
datasets). Full data: `results/generalization/perturbation_delta.csv`.

### 6.4.3 Limitations and Caveats

---

# VII. Conclusion and Future Work

> **Note:** Framework contributions and outcomes summarized here (moved from former Ch. VI per advisor feedback).

## 7.1 Summary of Contributions

### 7.1.1 Dataset Contribution

### 7.1.2 End-to-End Framework

### 7.1.3 Generalization Insights

### 7.1.4 Behavior Output Analysis

### 7.1.5 IoT Deployment Contribution

## 7.2 Summary of Work Completed

## 7.3 Future Work

### 7.3.1 Behavior Recognition Expansion

### 7.3.2 Analytics Layer Development

### 7.3.3 Expanded Dataset Coverage

### 7.3.4 Real-Time Edge Deployment

---

# References

---

# Appendices

## Appendix A: Config A vs. Config B Full Training Parameters

## Appendix B: CVB Ground Truth Deduplication Procedure

## Appendix C: OC-SORT Integration Notes
