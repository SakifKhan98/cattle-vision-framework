# HiPE1 v2 Behavior Training Guide

**RF-DETR-Seg pipeline — VideoMAE v2 training, evaluation, and retrieval**

This guide covers running all 5 v2 VideoMAE behavior configs on HiPE1 using the
RF-DETR-Seg tubelets (`data/processed/tubelets_rfdetr/`). The Docker image and src
code are unchanged from v1 — only new tubelets and v2 config files need to be synced.

Run all local commands from the project root (`~/TXST/Thesis/cattle-vision-framework`).

---

## Prerequisites

**Local (dubuntu):**

- Step 09b complete: `data/processed/tubelets_rfdetr/labels.csv` exists with ~137k rows
- SSH alias `hipe1` configured in `~/.ssh/config` (see `docs/hipe_ops.md §2`)

**HiPE1 (already verified):**

- Docker image `cattle-videomae:v1` loaded
- Both Tesla V100 GPUs free (16 GB each)
- ~604 GB disk available at `/home/zxs12/`

---

## Step 1 — Sync Data and Configs to HiPE1

Run from local machine. The tubelets sync (~33–40 GB) is the slow step.

```bash
# 1a. Create target directories on HiPE1
ssh hipe1 "mkdir -p ~/cattle_behavior/data/processed/tubelets_rfdetr \
                      ~/cattle_behavior/logs \
                      ~/cattle_behavior/runs/behavior"

# 1b. Sync tubelets_rfdetr (~33–40 GB, resumable)
rsync -avz --progress \
    data/processed/tubelets_rfdetr/ \
    hipe1:~/cattle_behavior/data/processed/tubelets_rfdetr/

# 1c. Sync v2 configs (fast, just 5 yaml files added)
rsync -avz configs/behavior/ hipe1:~/cattle_behavior/configs/behavior/

# 1d. Verify
ssh hipe1 "wc -l ~/cattle_behavior/data/processed/tubelets_rfdetr/labels.csv"
# expect: 137596 (137595 rows + 1 header)
```

> **If rsync drops:** re-run the exact same command — rsync resumes from where it left off.

---

## Step 2 — Training on HiPE1

### 2.1 The 5 v2 Configs

| Config file                     | Train dataset | Val dataset | Labels         | Run dir                    |
| ------------------------------- | ------------- | ----------- | -------------- | -------------------------- |
| `videomae_combined_v2.yaml`     | both          | both        | 0–6            | `videomae_combined_v2`     |
| `videomae_cvb_v2.yaml`          | CVB           | CVB         | 0–6            | `videomae_cvb_v2`          |
| `videomae_cbvd5_v2.yaml`        | CBVD-5        | CBVD-5      | 0–6            | `videomae_cbvd5_v2`        |
| `videomae_cbvd5_to_cvb_v2.yaml` | CBVD-5        | CVB         | 0–4 (val only) | `videomae_cbvd5_to_cvb_v2` |
| `videomae_cvb_to_cbvd5_v2.yaml` | CVB           | CBVD-5      | 0–4 (val only) | `videomae_cvb_to_cbvd5_v2` |

Run `combined_v2` first — it is the primary config used in Step 12b analytics.

### 2.2 GPU Assignment Strategy

HiPE1 has 2 V100s. Run 2 configs in parallel, then the remaining 3 in sequence as
GPUs free up. Recommended order:

| Batch  | GPU 0             | GPU 1             |
| ------ | ----------------- | ----------------- |
| First  | `combined_v2`     | `cvb_v2`          |
| Second | `cbvd5_v2`        | `cbvd5_to_cvb_v2` |
| Third  | `cvb_to_cbvd5_v2` | (idle)            |

### 2.3 Setting Up tmux on HiPE1

```bash
ssh hipe1
cd ~/cattle_behavior
tmux new-session -s behavior_v2
```

tmux cheat sheet:

- `Ctrl+B c` — new window
- `Ctrl+B 0` / `Ctrl+B 1` — switch windows
- `Ctrl+B d` — detach (session keeps running)
- `tmux attach -t behavior_v2` — reattach later

### 2.4 Docker Run Command Template

```bash
docker run --rm \
    --gpus '"device=DEVICE_ID"' \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    python src/behavior/train.py --config configs/behavior/CONFIG_FILE \
    2>&1 | tee logs/LOG_FILE
```

> **`--shm-size=16g` is required.** The VideoMAE DataLoader fails with
> `RuntimeError: No space left on device` without it.

### 2.5 Batch 1 — combined_v2 (GPU 0) + cvb_v2 (GPU 1)

**Window 0 — combined_v2 on GPU 0:**

```bash
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    python src/behavior/train.py --config configs/behavior/videomae_combined_v2.yaml \
    2>&1 | tee logs/combined_v2.log
```

`Ctrl+B c` — new window.

**Window 1 — cvb_v2 on GPU 1:**

```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    python src/behavior/train.py --config configs/behavior/videomae_cvb_v2.yaml \
    2>&1 | tee logs/cvb_v2.log
```

`Ctrl+B d` — detach. Training continues in the background.

### 2.6 Batch 2 — cbvd5_v2 + cbvd5_to_cvb_v2

When a Batch 1 window finishes (returns to prompt), switch to it and launch the next config
on that same GPU. Replace `device=0` or `device=1` to match which GPU just freed up.

**cbvd5_v2:**

```bash
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    python src/behavior/train.py --config configs/behavior/videomae_cbvd5_v2.yaml \
    2>&1 | tee logs/cbvd5_v2.log
```

**cbvd5_to_cvb_v2:**

```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    python src/behavior/train.py --config configs/behavior/videomae_cbvd5_to_cvb_v2.yaml \
    2>&1 | tee logs/cbvd5_to_cvb_v2.log
```

### 2.7 Batch 3 — cvb_to_cbvd5_v2

```bash
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    -v $(pwd)/data:/workspace/data:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/configs:/workspace/configs:ro \
    -v $(pwd)/src:/workspace/src:ro \
    cattle-videomae:v1 \
    python src/behavior/train.py --config configs/behavior/videomae_cvb_to_cbvd5_v2.yaml \
    2>&1 | tee logs/cvb_to_cbvd5_v2.log
```

---

## Step 3 — Monitoring

From your local machine (no need to SSH in interactively):

```bash
# Check latest epoch for any config
ssh hipe1 "tail -3 ~/cattle_behavior/logs/combined_v2.log"
ssh hipe1 "tail -3 ~/cattle_behavior/logs/cvb_v2.log"
ssh hipe1 "tail -3 ~/cattle_behavior/logs/cbvd5_v2.log"
ssh hipe1 "tail -3 ~/cattle_behavior/logs/cbvd5_to_cvb_v2.log"
ssh hipe1 "tail -3 ~/cattle_behavior/logs/cvb_to_cbvd5_v2.log"

# GPU utilization
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"

# Active containers
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'"

# Reattach to tmux
ssh hipe1
tmux attach -t behavior_v2
# Ctrl+B 0 / Ctrl+B 1 to switch between config windows
```

**What to look for in logs:**

```
Epoch   4 | lr=5.00e-05 | train_loss=0.1565 train_f1=0.9407 | val_loss=0.9726 val_macro_f1=0.7537
  -> new best val_macro_f1=0.7537
...
Early stopping at epoch 12 (patience=8)
Done. Best val_macro_f1=0.XXXX. Outputs in runs/behavior/videomae_combined_v2
```

Training stops when val_macro_f1 hasn't improved for 8 epochs (early stopping patience).
Expect 10–20 epochs per config, roughly 1–3 hours per config on V100.

---

## Step 4 — Retrieving Results from HiPE1

Run from local machine after all 5 configs complete:

```bash
# 4a. Fetch all v2 checkpoints and run metadata
mkdir -p runs/behavior/
rsync -avz \
    hipe1:~/cattle_behavior/runs/behavior/ \
    runs/behavior/

# 4b. Fetch logs
mkdir -p logs/
rsync -avz \
    hipe1:~/cattle_behavior/logs/ \
    logs/

# 4c. Verify all 5 checkpoints arrived
for cfg in combined cvb cbvd5 cbvd5_to_cvb cvb_to_cbvd5; do
    ckpt="runs/behavior/videomae_${cfg}_v2/checkpoint_best.pt"
    if [ -f "$ckpt" ]; then
        echo "OK  $ckpt"
    else
        echo "MISSING  $ckpt"
    fi
done
```

---

## Step 5 — Evaluation (local, Step 11b)

Run locally after checkpoints are fetched. Uses the `cattletransformer` conda env.

```bash
# All 5 v2 configs at once (skips any missing checkpoints)
bash scripts/11b_evaluate_rfdetr.sh

# Or single config
bash scripts/11b_evaluate_rfdetr.sh \
    configs/behavior/videomae_combined_v2.yaml \
    runs/behavior/videomae_combined_v2/checkpoint_best.pt
```

**Outputs written:**

- `results/behavior/predictions/videomae_*_v2_val.csv` — per-tubelet predictions
- `results/behavior/predictions_rfdetr/` — same CSVs, copied here for analytics separation
- `results/behavior/confusion_matrices/*_v2_*.png` — confusion matrix PNGs
- `results/behavior/f1_per_class.csv` — v2 rows appended (v1 rows preserved)

**Verify:**

```bash
ls results/behavior/predictions_rfdetr/
cat results/behavior/f1_per_class.csv | grep v2
```

---

## Step 6 — After Evaluation

Once all 5 v2 evals are done, run Step 12b analytics (uses `combined_v2` predictions):

```bash
bash scripts/12b_generate_analytics_rfdetr.sh
```

Then update the runbook status table in `docs/design/rfdetr_seg_pipeline_runbook.md`
and commit results:

```bash
git add results/behavior/ logs/
git commit -m "Add v2 behavior training results (RF-DETR-Seg tubelets)"
```

---

## Troubleshooting

### SSH drops during rsync

Re-run the exact same rsync command — it skips already-transferred bytes.

### "No space left on device" during training

The `--shm-size=16g` flag is missing from the docker run command.

### Container exits immediately / no GPU

```bash
ssh hipe1 "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
# If this fails:
ssh hipe1 "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
```

### Training diverges (loss goes to NaN)

Reduce learning rate in the config (`lr: 1.0e-5`) and restart. Do not modify the config
on disk during a run — kill the container, edit, then relaunch.

### Wrong image name error

The image on HiPE1 is `cattle-videomae:v1`. Do not use `cattle-behavior` (that name
is referenced in some older scripts but was never the actual tag on HiPE1).

### tmux session missing (HiPE1 rebooted)

```bash
ssh hipe1 "docker ps"   # check if container still running (survives tmux loss)
ssh hipe1 "tail -5 ~/cattle_behavior/logs/combined_v2.log"   # find last epoch
# If container died, relaunch with same docker run command (training restarts from epoch 1)
```

---

## Quick Reference — All Docker Commands

```bash
# combined_v2   (GPU 0) — PRIMARY, run first
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro -v $(pwd)/src:/workspace/src:ro \
  cattle-videomae:v1 python src/behavior/train.py \
  --config configs/behavior/videomae_combined_v2.yaml 2>&1 | tee logs/combined_v2.log

# cvb_v2   (GPU 1)
docker run --rm --gpus '"device=1"' --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro -v $(pwd)/src:/workspace/src:ro \
  cattle-videomae:v1 python src/behavior/train.py \
  --config configs/behavior/videomae_cvb_v2.yaml 2>&1 | tee logs/cvb_v2.log

# cbvd5_v2   (GPU 0 or 1)
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro -v $(pwd)/src:/workspace/src:ro \
  cattle-videomae:v1 python src/behavior/train.py \
  --config configs/behavior/videomae_cbvd5_v2.yaml 2>&1 | tee logs/cbvd5_v2.log

# cbvd5_to_cvb_v2   (GPU 0 or 1)
docker run --rm --gpus '"device=1"' --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro -v $(pwd)/src:/workspace/src:ro \
  cattle-videomae:v1 python src/behavior/train.py \
  --config configs/behavior/videomae_cbvd5_to_cvb_v2.yaml 2>&1 | tee logs/cbvd5_to_cvb_v2.log

# cvb_to_cbvd5_v2   (GPU 0 or 1)
docker run --rm --gpus '"device=0"' --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro -v $(pwd)/src:/workspace/src:ro \
  cattle-videomae:v1 python src/behavior/train.py \
  --config configs/behavior/videomae_cvb_to_cbvd5_v2.yaml 2>&1 | tee logs/cvb_to_cbvd5_v2.log
```
