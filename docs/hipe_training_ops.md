# HiPE Server Training Operations Guide
**Cattle Vision Framework — RF-DETR-Seg Hyperparameter Tuning**
Texas State University, Master's Thesis — Sakif Khan

---

## Hardware Overview

| Machine | GPU | VRAM | CUDA | Role |
|---------|-----|------|------|------|
| Local (dubuntu) | RTX 3060 | 12GB | 13.1 | Smoke testing |
| HiPE1 | 2× Tesla V100 | 16GB each | 12.2 | Config A + B |
| HiPE2 | 2× Tesla V100 | 16GB each | 12.2 | Config C + D |

Access path: `Local → leap2.txstate.edu → hipe1/hipe2.mitte.txstate.edu`

---

## 1. One-Time SSH Setup

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

Test: `ssh hipe1 "nvidia-smi"`

---

## 2. Local Docker Setup

### Fix Docker context (Linux + Docker Desktop)

Docker Desktop uses its own daemon and ignores `/etc/docker/daemon.json`.
Always use the system Docker context for GPU work:

```bash
docker context use default
```

Verify the nvidia runtime is visible:
```bash
docker info | grep -i runtime
# Must show: nvidia
```

If nvidia runtime is missing from system dockerd:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Local smoke test

Before transferring anything to servers, validate the image locally:

```bash
cd ~/TXST/Thesis/cattle-vision-framework/one_day

docker run --rm --gpus all \
    --shm-size=8g \
    -v $(pwd)/data/rfdetr_seg/cattle:/workspace/data/rfdetr_seg/cattle:ro \
    -v $(pwd)/runs:/workspace/runs \
    -v $(pwd)/train_hipe1.py:/workspace/train_hipe1.py:ro \
    -e EPOCHS=2 -e BATCH_SIZE=2 \
    cattle-rfdetr-seg:v1 \
    train_hipe1.py
```

`BATCH_SIZE=2` because the RTX 3060 (11GB) cannot fit batch_size=4+ with multi-scale
resolution 552. The V100s (16GB) use the default of 4.

**If this completes without error, the image is valid. Do not rebuild it.**

---

## 3. Dataset Preparation

### Important: flatten the images directory

RF-DETR expects images at `train/<filename>.jpg` but the dataset was exported
with images in `train/images/<filename>.jpg`. Flatten it once:

```bash
cd data/rfdetr_seg/cattle
mv train/images/* train/ && rmdir train/images
mv valid/images/* valid/ && rmdir valid/images
```

### Symlink note

Symlinks exist in `data/processed/detection/` but those are NOT needed on the
servers. Only `data/rfdetr_seg/cattle/` is used for Docker training, and it
contains real files (no symlinks).

---

## 4. Key Script Fixes Applied

These issues were found during bring-up and are already fixed in the scripts:

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `FileNotFoundError: test/_annotations.coco.json` | RF-DETR auto-sets `dataset_file="roboflow"` when `dataset_dir` is passed, requiring a `test/` split | Added `run_test=False` to all train kwargs |
| `TypeError: multiple values for 'callbacks'` | `detr.py` wrapper passes `callbacks` internally; user code also passed it | Removed callbacks entirely |
| `BATCH_SIZE` not configurable | Hardcoded value, couldn't override for local testing | Now reads `os.environ.get("BATCH_SIZE", 4)` |
| `EPOCHS` not configurable | Hardcoded value | Now reads `os.environ.get("EPOCHS", 100)` |
| `validate_dataset` failing after flatten | Was checking for `train/images/` subdir | Removed `img_dir` check, annotation JSON existence is sufficient |
| OOM on V100 16GB at batch_size=8 | Multi-scale bumps resolution 432→552, memory spikes | Reduced default `BATCH_SIZE` to 4, added `GRAD_ACCUM=2` to keep effective batch=8 |

---

## 5. Training Scripts

### Script inventory

| Script | Purpose |
|--------|---------|
| `train_hipe1.py` | Seg-Medium, lr=1e-4, uses `--gpus all` (Config A) |
| `train_hipe2.py` | Seg-Large, lr=5e-5, uses `--gpus all` (Config D legacy) |
| `scripts/train_server.py` | Parameterized via env vars — use for all 4-config HP grid |

### train_server.py environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `medium` | `medium` or `large` |
| `LR` | `1e-4` | Learning rate |
| `EPOCHS` | `100` | Number of epochs |
| `BATCH_SIZE` | `4` | Per-GPU batch size |
| `GRAD_ACCUM` | `1` | Gradient accumulation steps |
| `RUN_NAME` | auto | Output directory name under `runs/` |

### 4-Config HP grid

| Config | Server | GPU | MODEL | LR | Effective Batch |
|--------|--------|-----|-------|----|----------------|
| A | HiPE1 | 0 | medium | 1e-4 | 8 (bs=4, accum=2) |
| B | HiPE1 | 1 | medium | 5e-5 | 4 (bs=4, accum=1) |
| C | HiPE2 | 0 | large  | 1e-4 | 4 (bs=4, accum=1) |
| D | HiPE2 | 1 | large  | 5e-5 | 4 (bs=4, accum=1) |

For uniform effective batch=8 across all configs, add `-e GRAD_ACCUM=2` to B, C, D.

---

## 6. One-Time Server Setup (per server)

```bash
# Create directories
ssh hipe1 "mkdir -p ~/cattle_seg/{data/rfdetr_seg/cattle,runs,logs,scripts}"
ssh hipe2 "mkdir -p ~/cattle_seg/{data/rfdetr_seg/cattle,runs,logs,scripts}"

# Transfer dataset (~3.2GB, real files, no symlinks)
rsync -avz --progress data/rfdetr_seg/cattle/ hipe1:~/cattle_seg/data/rfdetr_seg/cattle/
rsync -avz --progress data/rfdetr_seg/cattle/ hipe2:~/cattle_seg/data/rfdetr_seg/cattle/

# Transfer Docker image (~4-5GB compressed)
rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe1:~/cattle_seg/
rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe2:~/cattle_seg/

# Load image on each server
ssh hipe1 "cd ~/cattle_seg && docker load < cattle-rfdetr-seg-v1.tar.gz"
ssh hipe2 "cd ~/cattle_seg && docker load < cattle-rfdetr-seg-v1.tar.gz"
```

rsync is resumable — if the connection drops, re-run the exact same command.

### Verify GPU access on servers

```bash
ssh hipe1 "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
ssh hipe2 "docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
```

---

## 7. Launching Training

### Transfer scripts (do this before every launch if scripts changed)

```bash
rsync train_hipe1.py hipe1:~/cattle_seg/
rsync scripts/train_server.py hipe1:~/cattle_seg/scripts/
rsync scripts/train_server.py hipe2:~/cattle_seg/scripts/
```

### HiPE1 — 2 windows in tmux

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
    -v ~/cattle_seg/train_hipe1.py:/workspace/train_hipe1.py:ro \
    cattle-rfdetr-seg:v1 \
    train_hipe1.py \
    2>&1 | tee ~/cattle_seg/logs/config_A.log
```

`Ctrl+B c` to open Window 1.

**Window 1 — Config B (GPU 1):**
```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=8g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/scripts/train_server.py:/workspace/train_server.py:ro \
    -e MODEL=medium -e LR=5e-5 -e RUN_NAME=seg_medium_lr5e5 \
    cattle-rfdetr-seg:v1 \
    train_server.py \
    2>&1 | tee ~/cattle_seg/logs/config_B.log
```

`Ctrl+B d` to detach (training keeps running).

### HiPE2 — 2 windows in tmux

```bash
ssh hipe2
mkdir -p ~/cattle_seg/logs
tmux new-session -s thesis
```

**Window 0 — Config C (GPU 0):**
```bash
docker run --rm \
    --gpus '"device=0"' \
    --shm-size=8g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/scripts/train_server.py:/workspace/train_server.py:ro \
    -e MODEL=large -e LR=1e-4 -e RUN_NAME=seg_large_lr1e4 \
    cattle-rfdetr-seg:v1 \
    train_server.py \
    2>&1 | tee ~/cattle_seg/logs/config_C.log
```

`Ctrl+B c` to open Window 1.

**Window 1 — Config D (GPU 1):**
```bash
docker run --rm \
    --gpus '"device=1"' \
    --shm-size=8g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/scripts/train_server.py:/workspace/train_server.py:ro \
    -e MODEL=large -e LR=5e-5 -e RUN_NAME=seg_large_lr5e5 \
    cattle-rfdetr-seg:v1 \
    train_server.py \
    2>&1 | tee ~/cattle_seg/logs/config_D.log
```

`Ctrl+B d` to detach.

---

## 8. Monitoring

```bash
# Latest epoch from all 4 configs
ssh hipe1 "tail -2 ~/cattle_seg/logs/config_A.log"
ssh hipe1 "tail -2 ~/cattle_seg/logs/config_B.log"
ssh hipe2 "tail -2 ~/cattle_seg/logs/config_C.log"
ssh hipe2 "tail -2 ~/cattle_seg/logs/config_D.log"

# GPU utilization
ssh hipe1 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
ssh hipe2 "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"

# Are containers still running?
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}'"
ssh hipe2 "docker ps --format 'table {{.Names}}\t{{.Status}}'"

# Reattach to tmux
ssh hipe1
tmux attach -t thesis
# Ctrl+B 0 / Ctrl+B 1 to switch windows
```

---

## 9. Retrieving Results

```bash
mkdir -p runs/segmentation/{seg_medium_lr1e4,seg_medium_lr5e5,seg_large_lr1e4,seg_large_lr5e5}

rsync -avz hipe1:~/cattle_seg/runs/seg_medium_lr1e4_baseline/ runs/segmentation/seg_medium_lr1e4/
rsync -avz hipe1:~/cattle_seg/runs/seg_medium_lr5e5/          runs/segmentation/seg_medium_lr5e5/
rsync -avz hipe2:~/cattle_seg/runs/seg_large_lr1e4/           runs/segmentation/seg_large_lr1e4/
rsync -avz hipe2:~/cattle_seg/runs/seg_large_lr5e5/           runs/segmentation/seg_large_lr5e5/
```

Each run directory contains:
- `run_config.json` — hyperparameters (cite in thesis)
- `metrics_log.json` — per-epoch loss + mAP (if callbacks work)
- `tensorboard/` — TensorBoard event files
- `checkpoint_best_total.pth` — best model weights

---

## 10. Troubleshooting

### "could not select device driver" on local machine
Docker Desktop is the active context. Switch to system Docker:
```bash
docker context use default
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### "libnvidia-ml.so.1: cannot open shared object file" on server
ldconfig cache is stale. Rebuild it:
```bash
sudo ldconfig
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
If `ldconfig -p | grep nvidia-ml` returns nothing, find the library:
```bash
find /usr /lib /lib64 -name 'libnvidia-ml.so*' 2>/dev/null
```
Then add that directory to `/etc/ld.so.conf.d/` and re-run `sudo ldconfig`.

### CUDA out of memory on V100 16GB
Multi-scale training bumps resolution from 432 to 552, causing memory spikes.
Reduce batch size:
```bash
# Add to docker run command:
-e BATCH_SIZE=4   # or 2 if still OOMing
```
Default is already 4 in all current scripts. If OOM persists at batch_size=4,
try batch_size=2 with GRAD_ACCUM=4 to keep effective batch=8.

### SSH connection dropped during rsync
rsync is resumable. Re-run the exact same command — it skips already-transferred
bytes and continues from where it stopped. Never abort and restart from scratch.

### tmux session missing (server rebooted)
```bash
ssh hipe1 "docker ps"   # check if container still running
# If no: check last epoch in log, restart with same docker run command
ssh hipe1 "tail -20 ~/cattle_seg/logs/config_A.log"
```

### Updating a training script (fast iteration)
The Docker image contains only the environment (PyTorch + rfdetr). Scripts are
volume-mounted at runtime. A code change only requires:
```bash
rsync scripts/train_server.py hipe1:~/cattle_seg/scripts/
# Restart the container (kill old one in tmux, re-run docker run command)
```
Only rebuild the image when pip dependencies change.

---

## 11. Fast Iteration Loop

```
Code change?
    → rsync the .py file only (seconds)
    → restart the container
    → DO NOT rebuild the Docker image

Pip dependency change?
    → Rebuild image locally
    → Smoke test with EPOCHS=2 BATCH_SIZE=2
    → Re-export tar.gz, re-rsync, re-load on servers
    → This is the slow path — keep deps stable

Error on server but not locally?
    → Run exact same Docker command locally first
    → If passes locally: issue is data/permissions on server, not code
```
