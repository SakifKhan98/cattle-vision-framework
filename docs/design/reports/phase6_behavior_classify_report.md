# Phase 6 Report — VideoMAE Behavior Classification

**Project:** Cattle Vision Framework

**Working directory:** `one_day/`

**Date started:** 2026-04-29

**Date completed:** 2026-05-06

**Status:** ALL 5 CONFIGS COMPLETE — evaluation done, all results local. Checkpoints, predictions CSVs, confusion matrices, and per-class F1 logs retrieved from HiPE1 (`/home/zxs12/cattle_behavior/`). Per-class F1 numbers confirmed from `evaluate.py` (ground truth).

---

## 1. Overview

Phase 6 fine-tunes a pretrained video transformer (VideoMAE-Base) to classify cattle behavior from short video clips. It consumes the tubelet data produced in Phase 5 — 125,586 labeled 16-frame clips across two cattle datasets — and produces five trained models covering different combinations of training and evaluation domains.

The five training configurations are designed to answer three research questions:

1. **In-domain performance:** How well does a model trained and tested on the same dataset perform? (Configs 1 and 2)
2. **Cross-domain generalization:** Does a model trained on one dataset generalize to another filmed in a different environment? (Configs 3 and 4)
3. **Combined training:** Does training on both datasets together improve robustness over in-domain training alone? (Config 5)

The primary output of Phase 6 is a `summary_table.csv` with per-class and macro-F1 scores across all five configurations — the core experimental results of the thesis.

---

## 2. Why VideoMAE

### 2.1 The Problem with Image-Based Approaches

Cattle behavior classification is inherently temporal. Several behaviors look nearly identical in a single frame:

- **Standing** vs **Ruminating-standing**: both show a stationary cow. Ruminating involves subtle jaw movement visible only over time.
- **Foraging** vs **Drinking**: both involve a lowered head posture. The distinction requires seeing motion pattern over multiple frames.
- **Lying** vs **Sleeping-lying**: posture alone is insufficient; activity level over time distinguishes them.

A frame-level classifier trained on individual JPEG crops would see only one moment in time and would confuse these behaviors routinely. A video model that sees the sequence of 16 frames can capture motion patterns, head movement trajectories, and activity transitions that are invisible in any single frame.

### 2.2 What VideoMAE Is

VideoMAE (Video Masked Autoencoder) is a self-supervised video representation learning method. The model is first pretrained on large-scale video data (Kinetics-400, 400 action classes, ~240,000 YouTube clips) by randomly masking 90% of video patches and learning to reconstruct them. This forces the model to develop rich spatio-temporal representations without requiring any labels during pretraining.

The architecture is a **Vision Transformer (ViT)** extended to video:

- Input: 16 frames × 224×224 pixels, divided into 3D patches (2 frames × 16×16 pixels)
- Total patches: 1,568 per clip
- Transformer: 12 layers, 768-dimensional hidden state, 12 attention heads
- Output: a 768-dimensional clip-level embedding used for classification

The pretrained model (`MCG-NJU/videomae-base-finetuned-kinetics`) has already learned to distinguish human actions. Fine-tuning it on cattle behavior transfers that temporal reasoning capability to a new domain.

### 2.3 Why Not Use a 3D CNN (C3D, SlowFast)?

3D CNNs like C3D and SlowFast are strong video models but:

- They require large amounts of training data to learn temporal features from scratch. With ~100k tubelets, a Vision Transformer pretrained via masked autoencoding outperforms 3D CNNs that were not pretrained on comparable scale.
- VideoMAE achieved state-of-the-art on Kinetics-400 and Something-Something-v2 at the time of publication, demonstrating its temporal modeling capacity.
- The ViT backbone has strong transfer properties from image pretraining (through masked autoencoding), giving it a head start even in a domain as different as cattle video.

### 2.4 Why VideoMAE-Base and Not a Larger Variant

VideoMAE also comes in Large and Huge variants. VideoMAE-Base was chosen because:

- HiPE1 has two Tesla V100 16 GB GPUs. VideoMAE-Large at batch size 8 would exceed 16 GB VRAM.
- The dataset (~100k training tubelets) is not large enough to justify the parameter count of VideoMAE-Large without significant risk of overfitting.
- Base is the standard benchmark variant; using it ensures results are comparable to the literature.

---

## 3. Data Inputs from Phase 5

### 3.1 Tubelet Structure

Each tubelet is a folder of 16 JPEG frames (224×224 after dataloader resizing) representing a single cattle instance. Tubelets were produced from OC-SORT tracks (CVB) and annotated keyframe windows (CBVD-5).

```text
data/processed/tubelets/
├── cbvd5/
│   └── {video_id}/kf{timestamp}_inst{hash}/frame_{00..15}.jpg
└── cvb/
    └── {video_id}/track_{id:04d}/tubelet_{idx:04d}/frame_{00..15}.jpg
```

All tubelets are indexed in a single `labels.csv`:

```text
dataset, video_id, tubelet_dir, start_frame, end_frame, label_id, label_name, split
```

### 3.2 Dataset Statistics (Final Export — Phase 5 Fixed)

Total: 125,586 labeled tubelets.

| Class          | Train       | Val        | Total       |
| -------------- | ----------- | ---------- | ----------- |
| Standing (0)   | 14,855      | 2,928      | 17,783      |
| Lying (1)      | 19,837      | 4,057      | 23,894      |
| Foraging (2)   | 47,961      | 11,305     | 59,266      |
| Drinking (3)   | 3,219       | 769        | 3,988       |
| Ruminating (4) | 12,705      | 3,352      | 16,057      |
| Grooming (5)   | 1,666       | 296        | 1,962       |
| Other (6)      | 2,236       | 400        | 2,636       |
| **Total**      | **102,479** | **23,107** | **125,586** |

**By dataset:**

| Dataset | Tubelets | Classes available                                        |
| ------- | -------- | -------------------------------------------------------- |
| CBVD-5  | 11,369   | Standing, Lying, Foraging, Drinking, Ruminating (5 core) |
| CVB     | 114,217  | All 7 classes including Grooming and Other               |

### 3.3 Class Imbalance

Foraging dominates at 47% of training data. Grooming (1.6%) and Drinking (3.1%) are the rarest classes. This imbalance is handled during training via inverse-frequency class weights:

```python
weight[c] = total_samples / (num_classes × count[c])
```

This causes the cross-entropy loss to penalize errors on rare classes more heavily, preventing the model from ignoring them.

### 3.4 No Test Split

CBVD-5's "test" CSV contains the same 50 video IDs as the "val" CSV — a known dataset artifact. CVB has no test split either. All evaluation uses `--split val`. This is a dataset limitation noted in the thesis.

---

## 4. Model Architecture

### 4.1 Base Model

**HuggingFace model ID:** `MCG-NJU/videomae-base-finetuned-kinetics`

The model was pretrained via masked autoencoding on Kinetics-400 and then fine-tuned on Kinetics-400 for action recognition. For cattle behavior classification, the Kinetics-400 classification head (400 classes) is discarded and replaced with a new linear layer.

### 4.2 Architecture Modification

```python
model = VideoMAEForVideoClassification.from_pretrained(model_name)
model.classifier = nn.Linear(768, num_classes)   # 7 for combined/in-domain, 7 for OOD
```

Only the classification head is replaced. All 12 transformer blocks and the patch embedding layer are initialized from the pretrained weights. Fine-tuning updates all parameters but with different learning rates for the backbone vs the head.

### 4.3 Input Format

- **Frames:** 16 consecutive JPEG frames per tubelet
- **Resize:** each frame resized to 224×224 (bilinear interpolation) in the dataloader
- **Normalization:** ImageNet mean/std `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`
- **Tensor shape:** `[batch, 3, 16, 224, 224]` (B, C, T, H, W)
- The VideoMAE model internally re-permutes to `[B, T, C, H, W]` before patch embedding

---

## 5. Training Setup

### 5.1 Hardware

| Resource     | Specification                           |
| ------------ | --------------------------------------- |
| Server       | HiPE1 (Texas State University)          |
| GPUs         | 2× Tesla V100 16 GB                     |
| CPU RAM      | 98 GB                                   |
| Storage      | 98 GB (5.6 GB free after data transfer) |
| Docker image | `cattle-videomae:v1`                    |

Config 5 (Combined) runs single-GPU (GPU 0) — the training script uses standard `model.to(device)` without DataParallel or DDP. Configs 1+2 will run simultaneously on GPU 0 and GPU 1 after Config 5 completes. Same pattern for Configs 3+4.

### 5.2 Hyperparameters (All Configs)

| Parameter                 | Value | Rationale                                        |
| ------------------------- | ----- | ------------------------------------------------ |
| `batch_size`              | 8     | V100 16 GB VRAM limit with VideoMAE-Base         |
| `grad_accum_steps`        | 4     | Effective batch = 32                             |
| `num_epochs`              | 30    | Upper bound; early stopping triggers earlier     |
| `lr`                      | 5e-5  | Low LR for backbone (pretrained weights)         |
| `lr_head`                 | 1e-3  | Higher LR for new classification head            |
| `weight_decay`            | 0.05  | AdamW regularization                             |
| `warmup_epochs`           | 3     | Linear warmup from 0 to target LR                |
| `early_stopping_patience` | 8     | Stop if no val_macro_f1 improvement for 8 epochs |
| `use_class_weights`       | true  | Inverse-frequency weighting in CrossEntropyLoss  |

### 5.3 Learning Rate Schedule

Two-phase cosine schedule with warmup:

- **Epochs 1–3 (warmup):** LR increases linearly from 0 → target LR
- **Epochs 4–30 (decay):** Cosine annealing from target LR → 0

The backbone and head use separate parameter groups: backbone at `lr=5e-5`, head at `lr_head=1e-3`. This prevents the pretrained backbone weights from being updated too aggressively while allowing the new head to learn quickly.

### 5.4 Optimization

- **Optimizer:** AdamW
- **Loss:** `nn.CrossEntropyLoss` with per-class inverse-frequency weights
- **Gradient scaler:** PyTorch AMP (`torch.amp.GradScaler`) with `autocast("cuda")` for mixed-precision training — reduces VRAM usage by ~30% and speeds up training

### 5.5 Checkpointing and Early Stopping

At the end of each epoch:

- `checkpoint_best.pt` — saved whenever `val_macro_f1` exceeds previous best
- `checkpoint_last.pt` — overwritten every epoch
- `log.csv` — appended with `epoch, train_loss, train_f1, val_loss, val_macro_f1`

Training stops when `val_macro_f1` has not improved for 8 consecutive epochs.

### 5.6 Docker Deployment

All scripts are volume-mounted at runtime — no image rebuild needed for code changes:

```bash
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

---

## 6. Five Training Configurations

| Config | Name              | Train domain | Val domain   | Val classes                    | Config YAML                  |
| ------ | ----------------- | ------------ | ------------ | ------------------------------ | ---------------------------- |
| 1      | In-domain CBVD-5  | CBVD-5       | CBVD-5       | 7 (Grooming/Other = 0 samples) | `videomae_cbvd5.yaml`        |
| 2      | In-domain CVB     | CVB          | CVB          | 7                              | `videomae_cvb.yaml`          |
| 3      | OOD: CBVD-5 → CVB | CBVD-5       | CVB          | 5 core only                    | `videomae_cbvd5_to_cvb.yaml` |
| 4      | OOD: CVB → CBVD-5 | CVB          | CBVD-5       | 5 core only                    | `videomae_cvb_to_cbvd5.yaml` |
| 5      | Combined          | CBVD-5 + CVB | CBVD-5 + CVB | 7                              | `videomae_combined.yaml`     |

**Why these configurations:**

- **Configs 1 and 2** establish in-domain baselines: how well can VideoMAE classify cattle behavior when training and test data come from the same cameras, environments, and annotation protocols? These set the upper bound for performance.

- **Configs 3 and 4** test cross-domain generalization — the harder and more practically valuable question. A cattle monitoring system deployed at a new farm should not need retraining from scratch. If Config 3 (train CBVD-5, test CVB) achieves reasonable F1, it suggests the model learns behavior-specific features rather than dataset-specific artifacts. Only the 5 core classes shared by both datasets are evaluated in OOD configs.

- **Config 5** (Combined) is the main thesis result. Training on both datasets forces the model to learn representations that work across visual conditions, camera angles, and annotation styles. The hypothesis is that combined training outperforms either in-domain model on the shared 5-class task, especially for rare classes like Drinking.

**Important note on Config 1:** `num_classes=7` in the config, but CBVD-5 has no Grooming or Other annotations. The model will never see those classes during training and will produce F1=0 for them during evaluation. The thesis reports macro-F1 over the 5 core classes for Config 1 to ensure fair comparison across configurations.

---

## 7. Evaluation

After training completes, each config is evaluated with `evaluate.py`:

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
    --split val
```

**Outputs:**

- Per-class F1, precision, recall
- Macro-F1 (primary metric — robust to class imbalance)
- Confusion matrix saved as PNG in `results/behavior/confusion_matrices/`

**Primary metric rationale:** Accuracy is misleading when classes are imbalanced — a model predicting "Foraging" always would hit ~47% accuracy. Macro-F1 averages F1 equally across classes, giving Drinking (3%) equal weight to Foraging (47%). A model with high macro-F1 must perform well on rare behaviors, which is what matters for welfare monitoring.

Results compiled into:

```text
results/behavior/summary_table.csv
```

with columns: `config, train_domain, val_domain, macro_f1, f1_standing, f1_lying, f1_foraging, f1_drinking, f1_ruminating, f1_grooming, f1_other`

---

## 8. Results — Config 5 (Combined), Complete (Epochs 1–12)

Config 5 is the first and main config to run. Training started 2026-04-29 on HiPE1 GPU 0. Early stopping triggered at epoch 12 (patience=8). Best checkpoint is at epoch 4.

### 8.1 Training Curve

| Epoch | LR       | train_loss | train_f1 | val_loss | val_macro_f1 | Best? |
| ----- | -------- | ---------- | -------- | -------- | ------------ | ----- |
| 1     | 0.00e+00 | 0.5639     | 0.7183   | 0.4974   | 0.7229       | ✓     |
| 2     | 1.67e-05 | 0.2124     | 0.9117   | 0.5845   | 0.7517       | ✓     |
| 3     | 3.33e-05 | 0.1819     | 0.9289   | 0.6681   | 0.7336       |       |
| 4     | 5.00e-05 | 0.1565     | 0.9407   | 0.9726   | **0.7537**   | ✓     |
| 5     | 4.98e-05 | 0.1218     | 0.9535   | 0.8884   | 0.7201       |       |
| 6     | 4.93e-05 | 0.1049     | 0.9613   | 0.9138   | 0.7465       |       |
| 7     | 4.85e-05 | 0.0892     | 0.9668   | 1.1249   | 0.7305       |       |
| 8     | 4.73e-05 | 0.0734     | 0.9719   | 1.0684   | 0.7345       |       |
| 9     | 4.59e-05 | 0.0644     | 0.9754   | 0.8559   | 0.7441       |       |
| 10    | 4.41e-05 | 0.0559     | 0.9798   | 1.0967   | 0.7377       |       |
| 11    | 4.21e-05 | 0.0512     | 0.9809   | 1.2655   | 0.7473       |       |
| 12    | 3.99e-05 | 0.0457     | 0.9840   | 1.3146   | 0.7437       |       |

**Early stopping at epoch 12** — no improvement beyond epoch 4 over 8 epochs. Final best: val_macro_f1=0.7537.

### 8.2 Per-Class Val F1 at Best Checkpoint (Epoch 4) — Confirmed from `evaluate.py`

| Class          | F1        | Notes                                                  |
| -------------- | --------- | ------------------------------------------------------ |
| Foraging (2)   | **0.980** | Dominant class, easy to learn                          |
| Drinking (3)   | **0.876** | Strong despite only 3.1% of training data              |
| Standing (0)   | **0.870** | High — visually distinct posture                       |
| Lying (1)      | **0.823** | High — visually distinct posture                       |
| Ruminating (4) | **0.772** | Good — subtle temporal cue learned                     |
| Grooming (5)   | **0.722** | CVB-only class, learned from ~1,666 train samples      |
| Other (6)      | **0.233** | Weak — catch-all class with no clear visual definition |

### 8.3 Interpretation of Results

**The fixed data works.** The first run of Config 5 (on the broken labels.csv with only 12,590 CVB tubelets) achieved val_macro_f1=0.3648 with Grooming and Other F1 at 0.000. After the Phase 5 fix that produced 114,217 CVB tubelets, macro-F1 jumped to 0.7537 by epoch 4 — more than doubling the score. This confirms the Phase 5 fix was the critical bottleneck.

**VideoMAE transfers well to cattle video.** A macro-F1 of 0.75 on a 7-class behavior problem by epoch 4 — without any domain-specific pretraining — demonstrates that the temporal representations learned on human action recognition (Kinetics-400) carry over to animal behavior. The model does not need to relearn how to detect motion; it only needs to map those representations to cattle-specific behavior patterns.

**Temporal reasoning is working.** Ruminating (F1=0.772) is the strongest evidence. Ruminating cattle are visually similar to standing or lying cattle — the only distinguishing cue is the rhythmic jaw movement over time. An image-based model would struggle with this class. VideoMAE capturing it at 0.772 by epoch 4 suggests it is genuinely using temporal information.

**Other class (F1=0.233) is expected to be weak.** The "Other" class in CVB is a residual catch-all — it covers walking, running, and any behavior not fitting the main 6 categories. There is no consistent visual or temporal pattern. The model predicts it correctly less than a quarter of the time, which is expected and should be noted in the thesis as a dataset limitation rather than a model failure.

**The generalization gap is present but val_macro_f1 is stable.** train_f1 reached 0.98 by epoch 12 while val_macro_f1 plateaued at 0.72–0.75. A gap of ~0.23 indicates overfitting, but val_macro_f1 never collapsed — it hovered in the 0.72–0.75 band across all 12 epochs. The val_loss rose from 0.497 at epoch 1 to 1.315 at epoch 12 while argmax performance remained stable, indicating calibration degradation rather than prediction quality regression.

**Epoch 4 is the final best.** No improvement was made after epoch 4 across all remaining 8 epochs. The cosine decay schedule (epochs 4–12) provided continued improvement in train_f1 but did not transfer to val, confirming the val-plateau was not simply a warmup artifact. Early stopping fired correctly at epoch 12.

### 8.4 Comparison to Baseline (Invalid First Run)

| Run                      | Labels CSV      | Tubelets    | Grooming F1 | Other F1 | Best val_macro_f1 |
| ------------------------ | --------------- | ----------- | ----------- | -------- | ----------------- |
| First run (2026-04-28)   | Broken (12,590) | 1,221 CVB   | 0.000       | 0.000    | 0.3648 (ep 4)     |
| Current run (2026-04-29) | Fixed (125,586) | 114,217 CVB | 0.722       | 0.233    | 0.7537 (ep 4)     |

---

## 9. All Training Run Results — Complete

All 5 configs trained to completion on HiPE1. Every run early-stopped at epoch 12 (patience=8) except Config 4 which stopped at epoch 15. All checkpoints verified at `~/cattle_behavior/runs/behavior/`.

### 9.1 Summary Table

| Config | Name              | Train N | Val N  | Best Macro-F1 | Best Epoch | Stopped Epoch |
| ------ | ----------------- | ------- | ------ | ------------- | ---------- | ------------- |
| 5      | Combined          | 102,479 | 23,107 | **0.7537**    | 4          | 12            |
| 2      | CVB in-domain     | 91,264  | 22,953 | **0.7607**    | 4          | 12            |
| 1      | CBVD-5 in-domain  | 11,215  | 154    | 0.3149        | 4          | 12            |
| 3      | OOD CBVD-5 → CVB  | 11,215  | 22,257 | 0.1690        | 4          | 12            |
| 4      | OOD CVB → CBVD-5  | 91,264  | 154    | 0.1789        | 7          | 15            |

### 9.2 Config 2 — CVB In-Domain (Best Overall)

Training log: `logs/cvb.log`. Train=91,264 / Val=22,953.

| Epoch | LR       | train_loss | train_f1 | val_loss | val_macro_f1 | Best? |
| ----- | -------- | ---------- | -------- | -------- | ------------ | ----- |
| 1     | 0.00e+00 | 0.5421     | 0.7520   | 0.5455   | 0.7307       | ✓     |
| 2     | 1.67e-05 | 0.1855     | 0.9265   | 0.6459   | 0.7322       | ✓     |
| 3     | 3.33e-05 | 0.1591     | 0.9387   | 1.0140   | 0.7052       |       |
| 4     | 5.00e-05 | 0.1420     | 0.9494   | 0.6048   | **0.7607**   | ✓     |
| 5     | 4.98e-05 | 0.1054     | 0.9604   | 0.7320   | 0.7193       |       |
| 6     | 4.93e-05 | 0.0896     | 0.9641   | 0.9598   | 0.7251       |       |
| 7     | 4.85e-05 | 0.0768     | 0.9696   | 0.9305   | 0.7570       |       |
| 8     | 4.73e-05 | 0.0653     | 0.9736   | 1.0582   | 0.7365       |       |
| 9     | 4.59e-05 | 0.0563     | 0.9770   | 1.0137   | 0.7287       |       |
| 10    | 4.42e-05 | 0.0510     | 0.9799   | 1.1997   | 0.7217       |       |
| 11    | 4.22e-05 | 0.0410     | 0.9841   | 1.1460   | 0.7305       |       |
| 12    | 3.99e-05 | 0.0373     | 0.9869   | 1.0827   | 0.7498       |       |

**Per-class val F1 at best epoch (4):**

| Class          | F1    |
| -------------- | ----- |
| Foraging (2)   | 0.979 |
| Drinking (3)   | 0.881 |
| Lying (1)      | 0.845 |
| Standing (0)   | 0.860 |
| Ruminating (4) | 0.801 |
| Grooming (5)   | 0.715 |
| Other (6)      | 0.243 |

CVB in-domain slightly outperforms Combined (0.7607 vs 0.7537). The Combined model's marginal loss is likely due to label noise from the heterogeneous CBVD-5 annotations mixing into CVB's clean tracking-based labels.

### 9.3 Config 1 — CBVD-5 In-Domain

Training log: `logs/cbvd5.log`. Train=11,215 / Val=154.

| Epoch | val_macro_f1 | Best? |
| ----- | ------------ | ----- |
| 1     | 0.2651       | ✓     |
| 2     | 0.2645       |       |
| 3     | 0.2693       | ✓     |
| 4     | **0.3149**   | ✓     |
| 5–12  | 0.25–0.31    | —     |

**Per-class val F1 at best epoch (4):**

| Class          | F1    | Notes                               |
| -------------- | ----- | ----------------------------------- |
| Standing (0)   | 0.906 |                                     |
| Foraging (2)   | 0.896 |                                     |
| Lying (1)      | 0.303 |                                     |
| Ruminating (4) | 0.100 | Only 3 val samples                  |
| Drinking (3)   | 0.000 | 0 val samples — dataset property    |
| Grooming (5)   | 0.000 | CVB-only class, no CBVD-5 train data|
| Other (6)      | 0.000 | CVB-only class, no CBVD-5 train data|

**Interpretation:** Macro-F1=0.3149 is misleading for this config. The 7-class denominator penalises heavily for Drinking (0 val samples) and Grooming/Other (no training data). Report 5-class macro-F1 in the thesis, which should be ~0.45. The tiny val set (154 samples) adds high variance; individual class F1 scores are unreliable. The Standing and Foraging scores (0.90+) show the model is genuinely learning from CBVD-5. The weak Lying and Ruminating scores reflect the extreme scarcity of those val samples (25 and 3, respectively).

### 9.4 Config 3 — OOD CBVD-5 → CVB

Training log: `logs/cbvd5_to_cvb.log`. Train=11,215 (CBVD-5) / Val=22,257 (CVB, 5 core classes).

Best val_macro_f1=**0.1690** (epoch 4, early stopped epoch 12).

**Per-class val F1 at best epoch (4) — confirmed from `evaluate.py`:**

| Class          | F1    |
| -------------- | ----- |
| Ruminating (4) | 0.493 |
| Standing (0)   | 0.215 |
| Lying (1)      | 0.281 |
| Drinking (3)   | 0.188 |
| Foraging (2)   | 0.006 |
| Grooming (5)   | 0.000 |
| Other (6)      | 0.000 |

**Interpretation:** Severe domain gap. A model trained on CBVD-5 (indoor, controlled lighting, fixed overhead cameras, 5 annotated behaviors) completely fails to recognise Foraging (0.006) in CVB outdoor footage. Ruminating (0.492) partially transfers because the jaw-movement temporal cue is camera-independent. This is a key thesis finding: visual appearance of cattle behavior is highly camera- and environment-specific. Foraging in CVB (cows grazing outdoors on grass) looks entirely different from Foraging in CBVD-5 (cows eating from indoor hay bins).

### 9.5 Config 4 — OOD CVB → CBVD-5

Training log: `logs/cvb_to_cbvd5.log`. Train=91,264 (CVB) / Val=154 (CBVD-5, 5 core classes).

Best val_macro_f1=**0.1789** (epoch 7, early stopped epoch 15).

**Per-class val F1 at best epoch (7):**

| Class          | F1    |
| -------------- | ----- |
| Standing (0)   | 0.675 |
| Foraging (2)   | 0.432 |
| Lying (1)      | 0.145 |
| Drinking (3)   | 0.000 |
| Ruminating (4) | 0.000 |
| Grooming (5)   | 0.000 |
| Other (6)      | 0.000 |

**Interpretation:** Also poor, but for different reasons. CVB→CBVD-5 achieves reasonable Standing (0.675) and partial Foraging (0.432) transfer, but fails entirely on Drinking, Ruminating, and the CVB-only classes (0 CBVD-5 val samples for Grooming/Other). The tiny CBVD-5 val set (154 samples) makes macro-F1 volatile. Notable that Config 4 ran 3 extra epochs (stopped at 15 vs 12 for others) — the optimizer found a marginally better representation at epoch 7, then plateaued.

### 9.6 Key Observations Across All Configs

1. **Best result: CVB in-domain 0.7607** — slightly beats Combined 0.7537. The marginal difference (~1%) is likely within noise given the val set size.

2. **OOD generalization is poor (0.17)** — the domain gap between indoor CBVD-5 and outdoor CVB is large enough that cross-domain models approach random performance. This is a significant thesis finding: single-domain training is insufficient for multi-environment deployment.

3. **Epoch 4 is optimal for all large configs** — warmup completes at epoch 3, LR reaches maximum at epoch 4. The model's best generalisation happens at this LR peak, before cosine decay begins. Post-epoch 4, train_f1 continued rising (overfitting) while val_macro_f1 plateaued.

4. **cudnn warnings are harmless** — `CUDNN_STATUS_NOT_SUPPORTED` in all logs means PyTorch fell back to an alternate conv3d execution plan. No impact on results.

5. **`Other` class (F1 ~0.2) is a dataset problem, not a model failure** — catch-all class with no consistent visual definition. Note this in the thesis.

---

## 10. Key Files

| File               | Location                                                                  | Description                                                      |
| ------------------ | ------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Training script    | `src/behavior/train.py`                                                   | Full fine-tuning loop with AMP, early stopping, cosine LR        |
| Dataset class      | `src/behavior/dataset.py`                                                 | `TubeletDataset` — reads labels.csv, loads 16 frames, normalizes |
| Evaluation script  | `src/behavior/evaluate.py`                                                | Per-class F1, confusion matrix, summary CSV                      |
| Combined config    | `configs/behavior/videomae_combined.yaml`                                 | Config 5 hyperparameters                                         |
| CBVD-5 config      | `configs/behavior/videomae_cbvd5.yaml`                                    | Config 1                                                         |
| CVB config         | `configs/behavior/videomae_cvb.yaml`                                      | Config 2                                                         |
| CBVD-5→CVB config  | `configs/behavior/videomae_cbvd5_to_cvb.yaml`                             | Config 3                                                         |
| CVB→CBVD-5 config  | `configs/behavior/videomae_cvb_to_cbvd5.yaml`                             | Config 4                                                         |
| HiPE1 combined log | `~/cattle_behavior/logs/combined_v2.log`                                  | Complete — early stopped epoch 12                                |
| Best checkpoint    | `~/cattle_behavior/runs/behavior/videomae_combined_v1/checkpoint_best.pt` | Final — epoch 4, val_macro_f1=0.7537                             |
| HiPE1 CBVD-5 log   | `~/cattle_behavior/logs/cbvd5.log`                                        | Complete — early stopped epoch 12                                |
| HiPE1 CVB log      | `~/cattle_behavior/logs/cvb.log`                                          | Complete — early stopped epoch 12                                |

---

## 11. Retrieving Results from HiPE1

HiPE1 home directory is `/home/zxs12/` (SSH alias `hipe1`). All checkpoints confirmed present as of 2026-05-05.

```bash
cd ~/TXST/Thesis/cattle-vision-framework/one_day

# Checkpoints + log.csv per run (~330 MB × 5 = ~1.6 GB)
rsync -avz hipe1:/home/zxs12/cattle_behavior/runs/behavior/ runs/behavior/

# Training logs
rsync -avz hipe1:/home/zxs12/cattle_behavior/logs/ logs/
```

Verify after rsync:

```bash
ls runs/behavior/*/checkpoint_best.pt
```

Expected: 5 files, one per config.

---

## 12. Running Evaluation

Two options: on HiPE1 (faster, ~10–15 min per large config on V100) or locally (RTX 3060, ~25–40 min per large config).

### 12.1 On HiPE1 (Recommended)

**Important:** must include `--shm-size=16g` or DataLoader shared memory fills up with `RuntimeError: No space left on device`.

```bash
ssh hipe1
tmux new-session -s eval
cd ~/cattle_behavior
mkdir -p results/behavior logs
```

Paste the full sequential block:

```bash
docker run --rm --gpus all --shm-size=16g --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    -v ~/cattle_behavior/results:/workspace/results \
    cattle-videomae:v1 \
    /workspace/behavior/evaluate.py \
    --config configs/behavior/videomae_combined.yaml \
    --checkpoint runs/behavior/videomae_combined_v1/checkpoint_best.pt \
    --split val 2>&1 | tee logs/eval_combined.log && \
docker run --rm --gpus all --shm-size=16g --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    -v ~/cattle_behavior/results:/workspace/results \
    cattle-videomae:v1 \
    /workspace/behavior/evaluate.py \
    --config configs/behavior/videomae_cvb.yaml \
    --checkpoint runs/behavior/videomae_cvb_v1/checkpoint_best.pt \
    --split val 2>&1 | tee logs/eval_cvb.log && \
docker run --rm --gpus all --shm-size=16g --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    -v ~/cattle_behavior/results:/workspace/results \
    cattle-videomae:v1 \
    /workspace/behavior/evaluate.py \
    --config configs/behavior/videomae_cbvd5_to_cvb.yaml \
    --checkpoint runs/behavior/videomae_cbvd5_to_cvb_v1/checkpoint_best.pt \
    --split val 2>&1 | tee logs/eval_cbvd5_to_cvb.log && \
docker run --rm --gpus all --shm-size=16g --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    -v ~/cattle_behavior/results:/workspace/results \
    cattle-videomae:v1 \
    /workspace/behavior/evaluate.py \
    --config configs/behavior/videomae_cbvd5.yaml \
    --checkpoint runs/behavior/videomae_cbvd5_v1/checkpoint_best.pt \
    --split val 2>&1 | tee logs/eval_cbvd5.log && \
docker run --rm --gpus all --shm-size=16g --entrypoint python3 \
    -v ~/cattle_behavior/data:/workspace/data:ro \
    -v ~/cattle_behavior/runs:/workspace/runs \
    -v ~/cattle_behavior/configs:/workspace/configs:ro \
    -v ~/cattle_behavior/src/behavior:/workspace/behavior:ro \
    -v ~/cattle_behavior/results:/workspace/results \
    cattle-videomae:v1 \
    /workspace/behavior/evaluate.py \
    --config configs/behavior/videomae_cvb_to_cbvd5.yaml \
    --checkpoint runs/behavior/videomae_cvb_to_cbvd5_v1/checkpoint_best.pt \
    --split val 2>&1 | tee logs/eval_cvb_to_cbvd5.log
```

Detach: `Ctrl+B D`. Monitor: `ssh hipe1 "tail -3 ~/cattle_behavior/logs/eval_combined.log"`

Retrieve results after all 5 finish:

```bash
rsync -avz hipe1:/home/zxs12/cattle_behavior/results/behavior/ results/behavior/
rsync -avz hipe1:/home/zxs12/cattle_behavior/logs/ logs/
```

### 12.2 Locally (Alternative)

Requires conda env active. Estimated ~1.5–2 hours total on RTX 3060.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer
cd ~/TXST/Thesis/cattle-vision-framework/one_day

python src/behavior/evaluate.py --config configs/behavior/videomae_combined.yaml \
    --checkpoint runs/behavior/videomae_combined_v1/checkpoint_best.pt --split val

python src/behavior/evaluate.py --config configs/behavior/videomae_cvb.yaml \
    --checkpoint runs/behavior/videomae_cvb_v1/checkpoint_best.pt --split val

python src/behavior/evaluate.py --config configs/behavior/videomae_cbvd5_to_cvb.yaml \
    --checkpoint runs/behavior/videomae_cbvd5_to_cvb_v1/checkpoint_best.pt --split val

python src/behavior/evaluate.py --config configs/behavior/videomae_cbvd5.yaml \
    --checkpoint runs/behavior/videomae_cbvd5_v1/checkpoint_best.pt --split val

python src/behavior/evaluate.py --config configs/behavior/videomae_cvb_to_cbvd5.yaml \
    --checkpoint runs/behavior/videomae_cvb_to_cbvd5_v1/checkpoint_best.pt --split val
```

**Estimated total time on RTX 3060 12 GB:** ~1.5–2 hours.

### 12.3 Outputs

Each `evaluate.py` run produces:

| Output | Path |
| ------ | ---- |
| Confusion matrix PNG | `results/behavior/confusion_matrices/{exp_name}_val.png` |
| Per-class F1 CSV (appended) | `results/behavior/f1_per_class.csv` |
| Per-tubelet predictions | `results/behavior/predictions/{exp_name}_val.csv` |

The predictions CSV is required for behavior video rendering (Phase 7 / Section 13).

---

## 13. Full-Pipeline Behavior Video Rendering

Script: `src/tracking/render_behavior_video.py`

Combines OC-SORT tracking JSON (bboxes + SAM2 masks) with VideoMAE predictions to render an annotated MP4. Each tracked cow displays its predicted behavior label (color-coded) with confidence. Overlapping tubelets are resolved per-frame by averaging logits before argmax.

**Prerequisite:** Run `evaluate.py` for the target config first to generate the predictions CSV.

### 13.1 Color Legend

| Class      | Color  |
| ---------- | ------ |
| Standing   | Green  |
| Lying      | Blue   |
| Foraging   | Cyan   |
| Drinking   | Yellow |
| Ruminating | Purple |
| Grooming   | Orange |
| Other      | Grey   |

### 13.2 Usage

```bash
# Auto-select best CVB video, combined model predictions
python src/tracking/render_behavior_video.py \
    --dataset cvb --auto \
    --predictions results/behavior/predictions/videomae_combined_v1_val.csv

# Specific video ID
python src/tracking/render_behavior_video.py \
    --dataset cvb \
    --video_id 0089_arm01_gopro1_20200323_002948_beh7_ani2_ins1_cut_2 \
    --predictions results/behavior/predictions/videomae_combined_v1_val.csv

# Use CVB in-domain model (best performing)
python src/tracking/render_behavior_video.py \
    --dataset cvb --auto \
    --predictions results/behavior/predictions/videomae_cvb_v1_val.csv
```

Output: `results/tracking/behavior_videos/{video_id}_behavior.mp4`

---

## 14. Task Status

| Task | Description                                 | Status                          |
| ---- | ------------------------------------------- | ------------------------------- |
| 6.1  | `TubeletDataset` class                      | Done                            |
| 6.2  | `train.py` — fine-tuning loop               | Done                            |
| 6.3  | `evaluate.py` — metrics + confusion matrix  | Done                            |
| 6.4  | 5 YAML configs                              | Done                            |
| 6.5  | `Dockerfile.videomae`                       | Done                            |
| 6.6  | Local sanity check                          | Done                            |
| 6.7  | HiPE1 deployment (Docker image, data rsync) | Done                            |
| 6.8  | Config 5 Combined — retrain with fixed data | **Done — val_macro_f1=0.7537**  |
| 6.9  | Configs 1–4                                 | **Done — all 5 complete**       |
| 6.10 | rsync checkpoints + logs from HiPE1         | **Done — 5 × 330 MB checkpoints, 5 CSVs, 5 PNGs, logs** |
| 6.11 | Evaluate all 5 locally, compile predictions | **Done — `results/behavior/f1_per_class.csv` confirmed** |
| 6.12 | Render behavior videos                      | **Pending — script ready (`render_behavior_video.py`), predictions CSVs exist** |
