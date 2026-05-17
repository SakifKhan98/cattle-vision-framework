#!/bin/bash
# =============================================================================
# CATTLE VISION FRAMEWORK — HiPE SERVER DEPLOYMENT GUIDE (UPDATED)
# Texas State University — Master's Thesis
# Author: Sakif Khan
#
# ACCESS PATH (from home):
#   Your laptop → leap2.txstate.edu → hipe1.mitte.txstate.edu (or hipe2)
#
# WHAT THIS ACHIEVES:
#   - One-time SSH config so rsync/scp work transparently through leap2
#   - Builds a Docker image locally, transfers it to both HiPE servers
#   - Transfers the cattle COCO dataset to both servers
#   - Launches training in tmux (survives SSH disconnect after 1 hour)
#   - HiPE1: RF-DETR-Seg-Medium, lr=1e-4  (baseline)
#   - HiPE2: RF-DETR-Seg-Large,  lr=5e-5  (larger model, conservative LR)
#   - Full thesis-grade logging: loss curves, mAP, LR, timing, GPU metrics
#
# TIME ESTIMATES:
#   SSH config setup:              5 min
#   Docker build (local):          ~10 min
#   Image transfer local→leap2:    ~30-40 min
#   Image transfer leap2→hipe:     ~5 min (internal university network, fast)
#   Dataset transfer:              ~20-30 min
#   Training per server:           ~4-6 hours (2× V100 DDP)
# =============================================================================


# =============================================================================
# STEP 0 — ONE-TIME SSH CONFIG (LOCAL machine, do this first)
# =============================================================================
# The ProxyJump directive routes SSH through leap2 automatically.
# After this, "ssh hipe1" and "rsync ... hipe1:..." just work.
# ServerAliveInterval keeps connections alive during long transfers.
#
# Open: nano ~/.ssh/config  (or vim, or any text editor)
# Paste the block below, save, and close.

: '
# ---- Paste into ~/.ssh/config ----

Host leap2
    HostName leap2.txstate.edu
    User zxs12
    ServerAliveInterval 60
    ServerAliveCountMax 30

Host hipe1
    HostName hipe1.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    ServerAliveInterval 60
    ServerAliveCountMax 30

Host hipe2
    HostName hipe2.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    ServerAliveInterval 60
    ServerAliveCountMax 30

# ---- End of block ----
'

# Test it works:
: '
ssh hipe1 "hostname && echo GPU: && nvidia-smi --query-gpu=name --format=csv,noheader"
ssh hipe2 "hostname && echo GPU: && nvidia-smi --query-gpu=name --format=csv,noheader"
'
# Expected:
#   hipe1.mitte.txstate.edu
#   GPU:
#   Tesla V100-PCIE-16GB
#   Tesla V100-PCIE-16GB

# Check disk space before proceeding (need ~25GB free per server):
: '
ssh hipe1 "df -h ~"
ssh hipe2 "df -h ~"
'


# =============================================================================
# STEP 1 — BUILD DOCKER IMAGE (LOCAL machine)
# =============================================================================
# Run from your project root (where Dockerfile lives).
# Creates a self-contained environment: PyTorch + RF-DETR + all deps.

: '
cd ~/TXST/Thesis/cattle-vision-framework/one_day
docker build -t cattle-rfdetr-seg:v1 .
'

# Verify:
: '
docker images | grep cattle-rfdetr-seg
'
# Expected: cattle-rfdetr-seg   v1   ...   ~8.0GB


# =============================================================================
# STEP 2 — EXPORT IMAGE TO COMPRESSED FILE (LOCAL machine)
# =============================================================================
# Compresses ~8GB image down to ~4-5GB for faster transfer.

: '
docker save cattle-rfdetr-seg:v1 | gzip > cattle-rfdetr-seg-v1.tar.gz
ls -lh cattle-rfdetr-seg-v1.tar.gz
'


# =============================================================================
# STEP 3 — TRANSFER TO SERVERS (LOCAL machine, open TWO terminals)
# =============================================================================
# ProxyJump in ~/.ssh/config means rsync goes:
#   local → leap2 → hipe1  (automatically, one command)
#
# IMPORTANT: rsync is resumable. If it disconnects, re-run the same
# command — it skips already-transferred files and continues from where
# it stopped. Never abort and restart from scratch.
#
# Run Terminal 1 (→ HiPE1) and Terminal 2 (→ HiPE2) simultaneously
# to transfer to both servers in parallel.

# ── Terminal 1: Transfer to HiPE1 ────────────────────────────────────────────
: '
# Create remote directories
ssh hipe1 "mkdir -p ~/cattle_seg/runs ~/cattle_seg/data ~/cattle_seg/logs"

# Docker image (~4-5GB compressed)
rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe1:~/cattle_seg/

# Cattle COCO dataset (~9GB, 17,111 images + 2 JSON files)
rsync -avz --progress \
    data/rfdetr_seg/cattle/ \
    hipe1:~/cattle_seg/data/rfdetr_seg/cattle/

# Training and evaluation scripts
rsync -avz \
    train_hipe1.py \
    src/tools/eval_rfdetr_seg.py \
    src/tools/plot_training_curves.py \
    hipe1:~/cattle_seg/
'

# ── Terminal 2: Transfer to HiPE2 (simultaneously) ───────────────────────────
: '
ssh hipe2 "mkdir -p ~/cattle_seg/runs ~/cattle_seg/data ~/cattle_seg/logs"

rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe2:~/cattle_seg/

rsync -avz --progress \
    data/rfdetr_seg/cattle/ \
    hipe2:~/cattle_seg/data/rfdetr_seg/cattle/

rsync -avz \
    train_hipe2.py \
    src/tools/eval_rfdetr_seg.py \
    src/tools/plot_training_curves.py \
    hipe2:~/cattle_seg/
'


# =============================================================================
# STEP 4 — LOAD DOCKER IMAGE ON EACH SERVER
# =============================================================================

# On HiPE1:
: '
ssh hipe1
cd ~/cattle_seg
docker load < cattle-rfdetr-seg-v1.tar.gz
# Takes ~3-5 minutes to decompress and load
docker images | grep cattle
exit
'

# On HiPE2 (separate terminal):
: '
ssh hipe2
cd ~/cattle_seg
docker load < cattle-rfdetr-seg-v1.tar.gz
docker images | grep cattle
exit
'


# =============================================================================
# STEP 5 — LAUNCH TRAINING IN TMUX
# =============================================================================
# CRITICAL: Run everything inside tmux. When your SSH session is killed
# after 1 hour, tmux keeps the process alive.
#
# tmux survival guide:
#   Start:    tmux new-session -s thesis
#   Detach:   Ctrl+B, then D         ← safe, training keeps running
#   Reattach: tmux attach -t thesis
#   List:     tmux ls
#   Scroll:   Ctrl+B then [          (arrow keys to scroll, Q to exit)

# ── HiPE1: RF-DETR-Seg-Medium, lr=1e-4 ───────────────────────────────────────
: '
ssh hipe1
cd ~/cattle_seg
tmux new-session -s thesis

# Verify both GPUs are free before launching
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv

# Launch container
# --gpus all       → expose both V100s to the container (DDP uses both)
# --shm-size=16g   → shared memory for DataLoader workers (needed for multi-GPU)
# -v mounts        → data/runs/logs persist after container exits
# tee              → writes to log file AND shows on screen simultaneously
docker run --rm \
    --gpus all \
    --shm-size=16g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/logs:/workspace/logs \
    -v ~/cattle_seg/train_hipe1.py:/workspace/train_hipe1.py:ro \
    cattle-rfdetr-seg:v1 \
    train_hipe1.py \
    2>&1 | tee ~/cattle_seg/logs/hipe1_training.log

# After launching: Ctrl+B then D to detach
# Training continues even after SSH disconnects
'

# ── HiPE2: RF-DETR-Seg-Large, lr=5e-5 ────────────────────────────────────────
: '
ssh hipe2
cd ~/cattle_seg
tmux new-session -s thesis

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv

docker run --rm \
    --gpus all \
    --shm-size=16g \
    -v ~/cattle_seg/data:/workspace/data:ro \
    -v ~/cattle_seg/runs:/workspace/runs \
    -v ~/cattle_seg/logs:/workspace/logs \
    -v ~/cattle_seg/train_hipe2.py:/workspace/train_hipe2.py:ro \
    cattle-rfdetr-seg:v1 \
    train_hipe2.py \
    2>&1 | tee ~/cattle_seg/logs/hipe2_training.log

# Ctrl+B then D to detach
'


# =============================================================================
# STEP 6 — MONITOR FROM HOME (LOCAL machine, no SSH reattach needed)
# =============================================================================
# Run any of these from your local terminal at any time.

# Latest epoch metrics (last 30 lines of log):
: '
ssh hipe1 "tail -30 ~/cattle_seg/logs/hipe1_training.log"
ssh hipe2 "tail -30 ~/cattle_seg/logs/hipe2_training.log"
'

# GPU temperature, utilization, memory:
: '
ssh hipe1 "nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv"
ssh hipe2 "nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv"
'

# Is Docker container still running?
: '
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'"
ssh hipe2 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'"
'

# Live tail both logs in split terminals:
: '
# Terminal 1:
ssh hipe1 "tail -f ~/cattle_seg/logs/hipe1_training.log"
# Terminal 2:
ssh hipe2 "tail -f ~/cattle_seg/logs/hipe2_training.log"
'

# Extract just mAP lines for a quick progress check:
: '
ssh hipe1 "grep -i 'map\|ap50\|epoch' ~/cattle_seg/logs/hipe1_training.log | tail -20"
ssh hipe2 "grep -i 'map\|ap50\|epoch' ~/cattle_seg/logs/hipe2_training.log | tail -20"
'

# Check how many epochs completed so far:
: '
ssh hipe1 "grep -c 'Epoch' ~/cattle_seg/logs/hipe1_training.log"
'


# =============================================================================
# STEP 7 — RETRIEVE RESULTS (LOCAL machine, after training completes)
# =============================================================================

: '
mkdir -p runs/segmentation/seg_medium_lr1e4_baseline
mkdir -p runs/segmentation/seg_large_lr5e5_conservative

# Pull HiPE1 results (checkpoints + metrics + log)
rsync -avz --progress \
    hipe1:~/cattle_seg/runs/seg_medium_lr1e4_baseline/ \
    runs/segmentation/seg_medium_lr1e4_baseline/
rsync -avz hipe1:~/cattle_seg/logs/hipe1_training.log \
    runs/segmentation/seg_medium_lr1e4_baseline/

# Pull HiPE2 results
rsync -avz --progress \
    hipe2:~/cattle_seg/runs/seg_large_lr5e5_conservative/ \
    runs/segmentation/seg_large_lr5e5_conservative/
rsync -avz hipe2:~/cattle_seg/logs/hipe2_training.log \
    runs/segmentation/seg_large_lr5e5_conservative/
'

# You will have locally:
#
#   runs/segmentation/seg_medium_lr1e4_baseline/
#       checkpoint_best_total.pth       ← production model weights
#       run_config.json                 ← hyperparameters (cite in thesis)
#       metrics_log.json                ← per-epoch loss + mAP (for plots)
#       tensorboard/                    ← TensorBoard event files
#       hipe1_training.log              ← full stdout (appendix material)
#
#   runs/segmentation/seg_large_lr5e5_conservative/
#       checkpoint_best_total.pth
#       run_config.json
#       metrics_log.json
#       tensorboard/
#       hipe2_training.log


# =============================================================================
# STEP 8 — GENERATE THESIS PLOTS AND EVALUATION (LOCAL machine)
# =============================================================================

: '
# 1. Training curves (loss + mAP per epoch, both runs on same plot)
python src/tools/plot_training_curves.py \
    --runs \
        runs/segmentation/seg_medium_lr1e4_baseline \
        runs/segmentation/seg_large_lr5e5_conservative \
    --output_dir results/rfdetr_seg/comparison

# 2. Quantitative evaluation on validation set
python src/tools/eval_rfdetr_seg.py \
    --checkpoint runs/segmentation/seg_medium_lr1e4_baseline/checkpoint_best_total.pth \
    --dataset_dir data/rfdetr_seg/cattle \
    --model medium \
    --output_dir results/rfdetr_seg/medium_baseline

python src/tools/eval_rfdetr_seg.py \
    --checkpoint runs/segmentation/seg_large_lr5e5_conservative/checkpoint_best_total.pth \
    --dataset_dir data/rfdetr_seg/cattle \
    --model large \
    --output_dir results/rfdetr_seg/large_conservative

# 3. Inference speed benchmark
python src/tools/eval_rfdetr_seg.py \
    --checkpoint runs/segmentation/seg_medium_lr1e4_baseline/checkpoint_best_total.pth \
    --model medium \
    --benchmark_only \
    --n_trials 100
'

# Thesis table you can fill in after Step 8:
#
# | Config              | mAP@50 | mAP@50:95 | Latency (ms) | Params (M) |
# |---------------------|--------|-----------|--------------|------------|
# | RF-DETR-Seg-M COCO  |  68.4  |   45.3    |     5.9      |   35.7     |  ← pretrained baseline (no fine-tune)
# | Seg-Medium lr=1e-4  |   ?    |    ?      |     5.9      |   35.7     |  ← HiPE1 result
# | Seg-Large  lr=5e-5  |   ?    |    ?      |     8.8      |   36.2     |  ← HiPE2 result
# | SAM2 (teacher)      |   —    |    —      |    ~500      |   856      |  ← reference


# =============================================================================
# TROUBLESHOOTING
# =============================================================================
#
# "ssh: Could not resolve hostname hipe1"
#   → Check ~/.ssh/config was saved correctly
#   → Test with explicit jump: ssh -J zxs12@leap2.txstate.edu zxs12@hipe1.mitte.txstate.edu
#
# "Connection closed" during rsync after 1 hour
#   → ServerAliveInterval in ~/.ssh/config should prevent this
#   → If it still happens: just re-run the rsync command, it resumes
#   → Alternative: run rsync from inside leap2 to avoid the double hop:
#       ssh leap2
#       rsync -avz zxs12@your-local-ip:~/one_day/data/ ~/cattle_seg/data/
#
# "docker: permission denied" on HiPE server
#   → sudo usermod -aG docker $USER && newgrp docker
#   → If no sudo: email professor's sysadmin
#
# "CUDA out of memory" at startup
#   → Reduce BATCH_SIZE in train_hipe1.py (8 → 4), re-rsync, restart
#
# Training loss stuck / not improving after epoch 20
#   → ssh hipe1 "grep 'loss' ~/cattle_seg/logs/hipe1_training.log | tail -40"
#   → If flat, stop container, change LR to 2e-5, restart
#
# tmux session missing (server rebooted while you were away)
#   → ssh hipe1 "docker ps"  — check if container still running
#   → If yes: docker attach $(docker ps -q)  OR  tail -f the log file
#   → If no (training was interrupted): check last epoch in log, then
#     restart with resume_checkpoint pointing to latest checkpoint.pth

echo "Guide ready. Begin at STEP 0: configure ~/.ssh/config"
