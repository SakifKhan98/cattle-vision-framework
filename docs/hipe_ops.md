# HiPE Server Operations Guide

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

HiPE1 and HiPE2 are Texas State University MITTE cluster GPU servers used for
RF-DETR-Seg training (Phase 3b) and VideoMAE behavior training (Phase 6).

---

## 1. Hardware

| Machine         | GPU           | VRAM       | CUDA | Role                                          |
| --------------- | ------------- | ---------- | ---- | --------------------------------------------- |
| Local (dubuntu) | RTX 3060      | 12 GB      | 13.1 | Smoke testing                                 |
| HiPE1           | 2× Tesla V100 | 16 GB each | 12.2 | RF-DETR-Seg (Configs A+B) + VideoMAE training |
| HiPE2           | 2× Tesla V100 | 16 GB each | 12.2 | RF-DETR-Seg (Configs C+D)                     |

Access path: `Local → leap2.txstate.edu → hipe1/hipe2.mitte.txstate.edu`

HiPE1 home directory: `/home/zxs12/`

---

## 2. SSH Setup (one-time)

Add to `~/.ssh/config`:

```
Host leap2
    HostName leap2.txstate.edu
    User zxs12
    ServerAliveInterval 60
    ServerAliveCountMax 60

Host hipe1
    HostName hipe1.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    ServerAliveInterval 60
    ServerAliveCountMax 60

Host hipe2
    HostName hipe2.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    ServerAliveInterval 60
    ServerAliveCountMax 60
```

**Test:**

```bash
ssh hipe1 "nvidia-smi"
```

---

## 3. Conda on HiPE1

The system `conda` is not in PATH by default. Activate it with:

```bash
source /home/zxs12/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer
```

Add to `~/.bashrc` on hipe1 for persistence.

---

## 4. Docker on HiPE1 / HiPE2

All training runs on HiPE use Docker containers to avoid environment conflicts.

**Verify GPU access:**

```bash
ssh hipe1 "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
```

**If GPU unavailable:**

```bash
sudo ldconfig
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4.1 Standard VideoMAE Eval Pattern

Run from the repo root on hipe1 after syncing files:

```bash
docker run --gpus all --shm-size=16g \
  -v $(pwd):/workspace \
  cattle-behavior \
  python src/behavior/train.py --config configs/behavior/videomae_combined.yaml
```

Always include `--shm-size=16g` — the VideoMAE DataLoader fails with
`RuntimeError: No space left on device` without it.

---

## 5. RF-DETR-Seg Training (Phase 3b)

### 5.1 One-Time Server Setup

```bash
# Create directories on each server
ssh hipe1 "mkdir -p ~/cattle_seg/{data/rfdetr_seg/cattle,runs,logs,scripts}"
ssh hipe2 "mkdir -p ~/cattle_seg/{data/rfdetr_seg/cattle,runs,logs,scripts}"

# Transfer dataset (~3.2 GB, real files — no symlinks)
rsync -avz --progress data/rfdetr_seg/cattle/ hipe1:~/cattle_seg/data/rfdetr_seg/cattle/
rsync -avz --progress data/rfdetr_seg/cattle/ hipe2:~/cattle_seg/data/rfdetr_seg/cattle/

# Transfer Docker image (~4-5 GB compressed)
rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe1:~/cattle_seg/
rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe2:~/cattle_seg/

# Load image on each server
ssh hipe1 "cd ~/cattle_seg && docker load < cattle-rfdetr-seg-v1.tar.gz"
ssh hipe2 "cd ~/cattle_seg && docker load < cattle-rfdetr-seg-v1.tar.gz"
```

rsync is resumable — if the connection drops, re-run the exact same command.

### 5.2 Dataset Preparation

RF-DETR expects images at `train/<filename>.jpg` (flat). Flatten once if needed:

```bash
cd data/rfdetr_seg/cattle
mv train/images/* train/ && rmdir train/images
mv valid/images/* valid/ && rmdir valid/images
```

### 5.3 4-Config Hyperparameter Grid

| Config | Server | GPU | Model  | LR   | Effective Batch   |
| ------ | ------ | --- | ------ | ---- | ----------------- |
| A      | HiPE1  | 0   | medium | 1e-4 | 8 (bs=4, accum=2) |
| B      | HiPE1  | 1   | medium | 5e-5 | 4 (bs=4, accum=1) |
| C      | HiPE2  | 0   | large  | 1e-4 | 4 (bs=4, accum=1) |
| D      | HiPE2  | 1   | large  | 5e-5 | 4 (bs=4, accum=1) |

**Best result:** Config B (medium, lr=5e-5), epoch 59. Checkpoint: `weights/rf-detr-seg-medium.pt`.

### 5.4 Launching Training (tmux)

```bash
ssh hipe1
mkdir -p ~/cattle_seg/logs
tmux new-session -s thesis
```

**Window 0 — Config A (GPU 0):**

```bash
docker run --rm \
    --gpus all \
    --shm-size=8g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/scripts/hipe/train_rfdetr_seg_medium.py:/workspace/train_rfdetr_seg_medium.py:ro \
    cattle-rfdetr-seg:v1 \
    train_rfdetr_seg_medium.py \
    2>&1 | tee ~/cattle_seg/logs/config_A.log
```

`Ctrl+B c` — new window.

**Window 1 — Config B (GPU 1):**

```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=8g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/scripts/hipe/train_rfdetr_seg_parameterized.py:/workspace/train_rfdetr_seg_parameterized.py:ro \
    -e MODEL=medium -e LR=5e-5 -e RUN_NAME=seg_medium_lr5e5 \
    cattle-rfdetr-seg:v1 \
    train_rfdetr_seg_parameterized.py \
    2>&1 | tee ~/cattle_seg/logs/config_B.log
```

`Ctrl+B d` to detach.

**HiPE2 — Config C (GPU 0):**

```bash
ssh hipe2
tmux new-session -s thesis
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=8g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/scripts/hipe/train_rfdetr_seg_parameterized.py:/workspace/train_rfdetr_seg_parameterized.py:ro \
    -e MODEL=large -e LR=1e-4 -e RUN_NAME=seg_large_lr1e4 \
    cattle-rfdetr-seg:v1 \
    train_rfdetr_seg_parameterized.py \
    2>&1 | tee ~/cattle_seg/logs/config_C.log
```

### 5.5 Environment Variables (train_rfdetr_seg_parameterized.py)

| Variable     | Default  | Description                 |
| ------------ | -------- | --------------------------- |
| `MODEL`      | `medium` | `medium` or `large`         |
| `LR`         | `1e-4`   | Learning rate               |
| `EPOCHS`     | `100`    | Training epochs             |
| `BATCH_SIZE` | `4`      | Per-GPU batch size          |
| `GRAD_ACCUM` | `1`      | Gradient accumulation steps |
| `RUN_NAME`   | auto     | Output dir under `runs/`    |

### 5.6 Retrieving RF-DETR-Seg Results

```bash
rsync -avz hipe1:~/cattle_seg/runs/seg_medium_lr1e4/ _archive/runs/seg_medium_lr1e4/
rsync -avz hipe1:~/cattle_seg/runs/seg_medium_lr5e5/ _archive/runs/seg_medium_lr5e5/
```

Metadata (log.txt, results.json, run_config.json) already committed in `results/segmentation/`.

---

## 6. VideoMAE Behavior Training (Phase 6)

Training is run on HiPE1 from the cattle_behavior directory.

### 6.1 One-Time Setup on HiPE1

```bash
# The full repo is synced to HiPE1
ssh hipe1 "mkdir -p ~/cattle_behavior"
rsync -avz --progress \
  src/ configs/ scripts/ data/processed/tubelets/ \
  hipe1:~/cattle_behavior/

# Sync Docker image (if not already loaded)
rsync -avz --progress cattle-videomae-v1.tar.gz hipe1:~/cattle_behavior/
ssh hipe1 "cd ~/cattle_behavior && docker load < cattle-videomae-v1.tar.gz"
```

### 6.2 Training Command (one config)

```bash
ssh hipe1
cd ~/cattle_behavior
tmux new-session -s behavior
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro \
  -v $(pwd)/src:/workspace/src:ro \
  cattle-behavior \
  python src/behavior/train.py --config configs/behavior/videomae_combined.yaml \
  2>&1 | tee logs/combined.log
```

### 6.3 Evaluation Command

```bash
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd)/data:/workspace/data:ro \
  -v $(pwd)/runs:/workspace/runs:ro \
  -v $(pwd)/configs:/workspace/configs:ro \
  -v $(pwd)/src:/workspace/src:ro \
  -v $(pwd)/results:/workspace/results \
  cattle-behavior \
  python src/behavior/evaluate.py \
    --config configs/behavior/videomae_combined.yaml \
    --checkpoint runs/behavior/videomae_combined_v1/checkpoint_best.pt \
    --split val
```

### 6.4 Retrieving VideoMAE Results

```bash
mkdir -p runs/behavior/

for CONFIG in combined cvb cbvd5 cbvd5_to_cvb cvb_to_cbvd5; do
  rsync -avz hipe1:~/cattle_behavior/runs/behavior/videomae_${CONFIG}_v1/ \
    runs/behavior/videomae_${CONFIG}_v1/
done

rsync -avz hipe1:~/cattle_behavior/results/behavior/ results/behavior/
```

---

## 7. Monitoring

```bash
# Latest epoch from RF-DETR-Seg configs
ssh hipe1 "tail -2 ~/cattle_seg/logs/config_A.log"
ssh hipe1 "tail -2 ~/cattle_seg/logs/config_B.log"
ssh hipe2 "tail -2 ~/cattle_seg/logs/config_C.log"
ssh hipe2 "tail -2 ~/cattle_seg/logs/config_D.log"

# GPU utilization
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# Active containers
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}'"

# Reattach to tmux
ssh hipe1
tmux attach -t thesis   # Ctrl+B 0 / Ctrl+B 1 to switch windows
```

---

## 8. Troubleshooting

### SSH connection dropped during rsync

rsync is resumable. Re-run the exact same command — it skips already-transferred bytes.

### tmux session missing (server rebooted)

```bash
ssh hipe1 "docker ps"              # check if container still running
ssh hipe1 "tail -20 ~/cattle_seg/logs/config_A.log"  # check last epoch
# Restart with same docker run command if needed
```

### Updating a training script without rebuilding image

Docker images contain only the environment. Scripts are volume-mounted at runtime.
Only transfer the changed `.py` file and restart the container:

```bash
rsync scripts/hipe/train_rfdetr_seg_parameterized.py hipe1:~/cattle_seg/scripts/hipe/
# Kill old container in tmux, re-run docker run command
```

Only rebuild the image when pip dependencies change.

### CUDA OOM on V100 16 GB

Reduce batch size via env var: `-e BATCH_SIZE=2 -e GRAD_ACCUM=4` (keeps effective batch=8).

### cudnn warnings in logs

`UserWarning: TF32 ... cudnn` warnings in all VideoMAE eval logs are harmless.
PyTorch uses a fallback convolution plan. Training and evaluation results are unaffected.
