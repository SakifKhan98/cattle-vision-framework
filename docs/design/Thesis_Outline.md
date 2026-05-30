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

> See `docs/design/reports/phase9_perturbation_report.md` §5 for full draft.

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

> See `docs/design/reports/phase9_perturbation_report.md` §7 for full draft.

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
