# Cattle Vision Framework — Master's Thesis Continuation Guide
**Author:** Sakif Khan, Texas State University  
**Thesis:** Automated Multi-Behavior Recognition in Cattle Using a Transformer-Based Video Pipeline  
**Date:** March 2026  
**Purpose:** This document is a complete context handoff for continuing work in a new chat session.

---

## 1. Project Overview

The thesis builds an automated pipeline that takes cattle surveillance video and produces per-animal behavior reports. The pipeline has 7 phases. The goal is a tool where someone uploads a video and receives a Gantt-style timeline of each cow's behavior, activity budgets, and welfare flags.

**Local machine:** Ubuntu, RTX 3060 12GB, conda env `cattletransformer`  
**Project root:** `~/TXST/Thesis/cattle-vision-framework/one_day/`  
**University HPC servers:**
- `leap2.txstate.edu` — jump host only, do not run jobs here
- `hipe1.mitte.txstate.edu` — 2× Tesla V100 16GB, training server
- `hipe2.mitte.txstate.edu` — 2× Tesla V100 16GB, training server
- SSH access: local → leap2 → hipe1/hipe2 (ProxyJump configured in `~/.ssh/config`)
- SSH keys set up, passwordless access working

---

## 2. Full Pipeline — Current Status

```
Phase 0  🔲  Video ingestion (not yet built — post-thesis)
Phase 2  ✅  RF-DETR cattle detection — COMPLETE
Phase 3  ✅  SAM2 segmentation — COMPLETE
Phase 3b 🔄  RF-DETR-Seg fine-tuning — IN PROGRESS (blocked on Docker issue)
Phase 4  🔲  OC-SORT tracking — NOT STARTED
Phase 5  🔲  Tubelet generation — NOT STARTED
Phase 6  🔲  VideoMAE behavior classification — NOT STARTED
Phase 7  🔲  Analytics — NOT STARTED
Phase 8  🔲  Report generation — NOT STARTED (post-thesis)
```

---

## 3. Completed Work — Detailed

### Phase 2 — RF-DETR Detection ✅
- Model: RF-DETR Medium, fine-tuned on CBVD-5 + CVB combined
- Result: 70.4% mAP@50 on combined validation set
- Outputs: `data/processed/tracking/{cbvd5,cvb}/*.json`
- Format: `{video_id: {frame_id: [{bbox:[x,y,w,h], score:float}]}}`
- Script: `src/detection/detect.py`
- Shell: `scripts/06_run_detection.sh`

### Phase 3 — SAM2 Segmentation ✅
- Model: SAM2.1 Hiera Large (frozen, NOT trained — used as pretrained tool)
- Checkpoint: `models/sam2/sam2.1_hiera_large.pt` (856MB)
- SAM2 source: `~/TXST/Thesis/cattle-vision-framework/sam2/`
- Strategy: RF-DETR boxes → SAM2 prompts → pixel masks
  - CBVD-5: keyframe mode (one prompt per frame, no propagation)
  - CVB: propagation mode with K=15 re-prompting interval
- CBVD-5 results: 684/687 videos, 15,900 masks, 100.0% coverage, 27.8 min runtime
- CVB results: 502/502 videos, 226,789 masks, 100.3% coverage, 329 min runtime
- Total masks: 242,689 across both datasets
- Outputs: `data/processed/segmentation/{cbvd5,cvb}/*.json`
- Output format per file:
  ```json
  {
    "video_id": "10",
    "dataset": "cbvd5",
    "frames": {
      "2": [{"bbox":[x,y,w,h], "score":0.93, "mask_rle":{"size":[H,W],"counts":"..."}, "mask_area":45066}]
    }
  }
  ```
- Coverage >100% explanation: SAM2 propagation can split one tracked object into
  two mask blobs; this is expected behavior, not an error. Use overall_coverage_rate
  (1.003) not mean_coverage_per_video (1.368) in the thesis — the latter is
  inflated by session-resumption artifact.
- Script: `src/segmentation/segment.py`
- Shell: `scripts/07_run_segmentation.sh`
- Reports: `results/segmentation/{cbvd5,cvb}_summary.json` and `_segmentation_stats.csv`

### Phase 3b — RF-DETR-Seg Fine-Tuning 🔄 IN PROGRESS

**Concept:** Use SAM2's 242,689 masks as pseudo-labels to train RF-DETR-Seg.
This replaces the slow SAM2 (500ms/frame) with a fast RF-DETR-Seg (6ms/frame)
in the production pipeline — an 83x speedup. This is called knowledge distillation
and is a strong thesis contribution.

**Thesis argument:** SAM2 ran for 6 hours generating supervision signal that
would have taken weeks to hand-label. RF-DETR-Seg trained on that pseudo-label
data becomes deployable at 6ms/frame. The comparison is:
- SAM2 teacher: 500ms/frame, 856MB, not real-time
- RF-DETR-Seg student: 6ms/frame, 35MB, real-time
- Quality gap: measured by mask IoU on held-out frames (eval script ready)

**Dataset built:** `data/rfdetr_seg/cattle/`
- Conversion script: `src/tools/sam2_to_coco_seg.py`
- Parameters used: `--min_area 500 --cvb_stride 15 --val_ratio 0.2`
- Train split: 13,718 images, 85,182 annotations, avg 6.2 instances/image
- Val split: 3,393 images, 21,526 annotations
- CBVD-5 contribution: 3,394 frames (indoor barn, mean area 23,222px)
- CVB contribution: 13,717 frames (outdoor GoPro, mean area ~7,200px — smaller cows)
- Both datasets included for cross-domain robustness

**Hyperparameter plan — two experiments in parallel:**
| Run | Server | Model | LR | Batch | Notes |
|-----|--------|-------|----|-------|-------|
| seg_medium_lr1e4_baseline | HiPE1 | Seg-Medium | 1e-4 | 8×2GPU | Baseline |
| seg_large_lr5e5_conservative | HiPE2 | Seg-Large | 5e-5 | 6×2GPU | Best expected |

**Scripts written (in project root and src/tools/):**
- `train_hipe1.py` — HiPE1 training script (Seg-Medium)
- `train_hipe2.py` — HiPE2 training script (Seg-Large)
- `src/tools/sam2_to_coco_seg.py` — SAM2 → COCO dataset conversion
- `src/tools/eval_rfdetr_seg.py` — post-training evaluation (COCO AP, speed, qualitative)
- `src/tools/plot_training_curves.py` — thesis figures from metrics_log.json
- `scripts/08_train_rfdetr_seg.sh` — local training wrapper
- `Dockerfile` — container for HiPE training
- `hipe_deployment_guide.sh` — step-by-step deployment reference

**Thesis metrics this produces:**
- mAP@50 and mAP@50:95 on cattle val set (both box and mask AP)
- Inference speed: mean ± std latency in ms, FPS, SAM2 speedup ratio
- Training curves: loss + mAP per epoch for both runs
- Qualitative: 16 side-by-side overlay images (CBVD-5 and CVB samples)
- Comparison table: COCO pretrained baseline vs fine-tuned Medium vs fine-tuned Large

**CURRENT BLOCKER — Docker issue on HiPE servers:**

The Docker container on HiPE1 is importing Python from the server's
`/opt/conda` instead of the container's Python. This is because the
official `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` base image
itself uses `/opt/conda` internally — this is normal and NOT a leak.

The actual error is:
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```
Root cause: `transformers` library inside the container is too new (>=4.38)
and expects a PyTorch 2.1+ pytree API that was renamed.

Fix already applied in Dockerfile: pinned `transformers==4.37.2`
Status: Dockerfile updated but new image NOT yet built and transferred.

**Next steps to unblock Phase 3b:**
1. Rebuild Docker image locally:
   ```bash
   docker build --no-cache -t cattle-rfdetr-seg:v1 .
   ```
2. Test fix locally BEFORE transferring:
   ```bash
   docker run --rm cattle-rfdetr-seg:v1 -c \
     "import torch; import transformers; print('OK:', torch.__version__, transformers.__version__)"
   ```
   Expected output: `OK: 2.1.0 4.37.2`
3. If OK, export and transfer:
   ```bash
   docker save cattle-rfdetr-seg:v1 | gzip > cattle-rfdetr-seg-v1.tar.gz
   rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe1:~/cattle_seg/
   rsync -avz --progress cattle-rfdetr-seg-v1.tar.gz hipe2:~/cattle_seg/
   ```
4. Reload on servers:
   ```bash
   ssh hipe1 "docker rmi cattle-rfdetr-seg:v1 2>/dev/null; docker load < ~/cattle_seg/cattle-rfdetr-seg-v1.tar.gz"
   ssh hipe2 "docker rmi cattle-rfdetr-seg:v1 2>/dev/null; docker load < ~/cattle_seg/cattle-rfdetr-seg-v1.tar.gz"
   ```
5. Verify fix:
   ```bash
   ssh hipe1 "docker run --rm cattle-rfdetr-seg:v1 -c \
     'import torch; import transformers; print(torch.__version__, transformers.__version__)'"
   ```
6. If Docker continues to have issues, fallback plan is conda on HiPE:
   ```bash
   ssh hipe1 "conda create -n rfdetr python=3.10 -y && \
     conda run -n rfdetr pip install 'transformers==4.37.2' 'rfdetr>=1.4.0' \
     supervision pycocotools opencv-python-headless albumentations tensorboard tqdm"
   ```
7. Launch training inside tmux (see deployment guide for full docker run command)
8. Monitor: `ssh hipe1 "tail -30 ~/cattle_seg/logs/hipe1_training.log"`

**After training completes, retrieve results:**
```bash
rsync -avz --progress hipe1:~/cattle_seg/runs/seg_medium_lr1e4_baseline/ \
    runs/segmentation/seg_medium_lr1e4_baseline/
rsync -avz --progress hipe2:~/cattle_seg/runs/seg_large_lr5e5_conservative/ \
    runs/segmentation/seg_large_lr5e5_conservative/
```

**Then run evaluation:**
```bash
python src/tools/eval_rfdetr_seg.py \
    --checkpoint runs/segmentation/seg_medium_lr1e4_baseline/checkpoint_best_total.pth \
    --dataset_dir data/rfdetr_seg/cattle \
    --model medium \
    --output_dir results/rfdetr_seg/medium_baseline \
    --n_qualitative 16

python src/tools/plot_training_curves.py \
    --runs \
        runs/segmentation/seg_medium_lr1e4_baseline \
        runs/segmentation/seg_large_lr5e5_conservative \
    --output_dir results/rfdetr_seg/comparison
```

---

## 4. Phase 4 — OC-SORT Tracking (NEXT MAJOR PHASE)

Not started. This is the next phase after Phase 3b training completes.

**What it does:** Links the same cow across frames with a persistent track ID.
Input: segmentation mask JSONs from Phase 3.
Output: `{frame_id: [{track_id, bbox, mask_rle}]}`

**Key design decision:** Use mask IoU as the cost function instead of box IoU.
This is better for cattle because cows are wide and boxy — their bounding
boxes overlap heavily when they're side by side, but their masks don't.

**Evaluation:**
- CVB has ground truth track IDs → quantitative eval: MOTA, MOTP, IDF1, ID Switches
- CBVD-5 has no ground truth track IDs → qualitative eval only

**Implementation:** OC-SORT is available as a Python package.
Key hyperparameter to tune: `det_thresh`, `max_age`, `min_hits`, `iou_threshold`

---

## 5. Data — What Exists

### Datasets
| Dataset | Type | Location | Notes |
|---------|------|----------|-------|
| CBVD-5 | Indoor dairy barn, China | `data/raw/cbvd5/` | 687 labeled videos, 6 frames each |
| CVB | Outdoor GoPro, Australia | `data/raw/cvb/` | 502 clips, 450 frames each |

### CBVD-5 Frame Naming
`data/raw/cbvd5/labelframes/labelframes/{video_id}_{frame_idx:05d}.jpg`
Example: `10_00002.jpg`

### CVB Frame Naming
`data/raw/cvb/raw_frames/{clip_id}/img_{frame_idx:05d}.jpg`
Example: `0002_arm01_gopro1_.../img_00001.jpg`

### Processed Data
| Path | Contents |
|------|----------|
| `data/processed/tracking/cbvd5/` | RF-DETR detection JSONs |
| `data/processed/tracking/cvb/` | RF-DETR detection JSONs |
| `data/processed/segmentation/cbvd5/` | SAM2 mask JSONs (687 files) |
| `data/processed/segmentation/cvb/` | SAM2 mask JSONs (502 files) |
| `data/rfdetr_seg/cattle/` | COCO segmentation dataset for RF-DETR-Seg training |
| `results/segmentation/` | Summary JSONs, CSVs, viz images |
| `models/sam2/sam2.1_hiera_large.pt` | SAM2 checkpoint (856MB) |

### RF-DETR-Seg COCO Dataset (ready for training)
```
data/rfdetr_seg/cattle/
    train/
        images/         13,718 JPEGs
        _annotations.coco.json   85,182 annotations
    valid/
        images/         3,393 JPEGs
        _annotations.coco.json   21,526 annotations
```
Annotation format: COCO instance segmentation with RLE masks.
Category: `{"id": 1, "name": "cattle", "supercategory": "animal"}`

---

## 6. Key Technical Notes

### SAM2 API (important for Phase 4 integration)
- SAM2 returns `(masks, scores, low_res_masks)` — shape `(1, H, W)` when `multimask_output=False`
- For propagation, `mask_input` must be SAM2's low-res logits `(1, 1, 256, 256) float32` NOT full-res binary mask
- The `low_res_masks[0]` from each `predict()` call must be saved and passed as `mask_input` next frame
- Mask indexing: use `masks[0]` not `masks[0,0]`

### RF-DETR-Seg API
- `from rfdetr import RFDETRSegMedium` (or Large, Small, etc.)
- `model = RFDETRSegMedium(pretrain_weights="path/to/checkpoint.pth")`
- `detections = model.predict(pil_image, threshold=0.5)`
- `detections.xyxy` — bounding boxes
- `detections.mask` — binary masks, shape `(N, H, W)`
- `detections.confidence` — scores
- Requires rfdetr>=1.4.0, transformers==4.37.2, PyTorch 2.1

### Coverage >100% in CVB
Not an error. SAM2 mask propagation occasionally splits one tracked object
into two separate blobs, both of which pass the area threshold. Each becomes
a separate mask entry. Use `overall_coverage_rate` (1.003) in the thesis,
not `mean_coverage_per_video` (1.368).

### GPU Temperature
- RTX 3060: safe limit 93°C, sustained 80°C is acceptable
- V100: runs much cooler, no temperature concerns
- Monitor: `watch -n 60 nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader`

---

## 7. SSH Access Pattern

```bash
# ~/.ssh/config (already configured on local machine)
Host leap2
    HostName leap2.txstate.edu
    User zxs12
    IdentityFile ~/.ssh/id_thesis
    ServerAliveInterval 60
    ServerAliveCountMax 30

Host hipe1
    HostName hipe1.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    IdentityFile ~/.ssh/id_thesis
    ServerAliveInterval 60
    ServerAliveCountMax 30

Host hipe2
    HostName hipe2.mitte.txstate.edu
    User zxs12
    ProxyJump leap2
    IdentityFile ~/.ssh/id_thesis
    ServerAliveInterval 60
    ServerAliveCountMax 30
```

SSH keys set up and passwordless. Direct `ssh hipe1` works from local machine.
Session disconnects after ~1 hour — always run jobs inside `tmux`.

**tmux commands:**
```bash
tmux new-session -s thesis   # start
Ctrl+B then D                 # detach (job keeps running)
tmux attach -t thesis         # reattach
tmux ls                       # list sessions
```

**Remote monitoring without attaching:**
```bash
ssh hipe1 "tail -30 ~/cattle_seg/logs/hipe1_training.log"
ssh hipe1 "nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader"
ssh hipe1 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'"
```

---

## 8. HiPE Server State

### What's already on HiPE1 and HiPE2
```
~/cattle_seg/
    data/rfdetr_seg/cattle/         ✅ transferred (13,718 train + 3,393 val images)
    cattle-rfdetr-seg-v1.tar.gz     ✅ transferred (OLD image with bug)
    train_hipe1.py                  ✅ transferred
    train_hipe2.py                  ✅ transferred
    plot_training_curves.py         ✅ transferred
    eval_rfdetr_seg.py              ✅ transferred
    logs/                           ✅ directory exists
    runs/                           ✅ directory exists
```

### What needs to happen next on HiPE servers
1. Rebuild Docker image locally with `transformers==4.37.2` fix
2. Re-export and re-transfer the tar.gz (replaces the buggy one)
3. `docker rmi` old image, `docker load` new image
4. Verify with: `docker run --rm cattle-rfdetr-seg:v1 -c 'import torch; import transformers; print(torch.__version__, transformers.__version__)'`
5. Expected output: `2.1.0 4.37.2`
6. Launch training inside tmux with docker run command (see Section 3 above)

### Docker run command for HiPE1 (copy-paste ready)
```bash
# SSH into HiPE1, then:
cd ~/cattle_seg
tmux new-session -s thesis
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
# Ctrl+B then D to detach
```

### Docker run command for HiPE2 (copy-paste ready)
```bash
cd ~/cattle_seg
tmux new-session -s thesis
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
```

---

## 9. All Scripts Written (complete inventory)

| Script | Purpose | Status |
|--------|---------|--------|
| `src/detection/detect.py` | RF-DETR detection | ✅ done |
| `src/segmentation/segment.py` | SAM2 segmentation | ✅ done |
| `src/tools/sam2_to_coco_seg.py` | Convert SAM2 masks → COCO dataset | ✅ done |
| `src/tools/eval_rfdetr_seg.py` | Evaluate RF-DETR-Seg checkpoint | ✅ done |
| `src/tools/plot_training_curves.py` | Generate thesis training figures | ✅ done |
| `scripts/06_run_detection.sh` | Run detection pipeline | ✅ done |
| `scripts/07_run_segmentation.sh` | Run segmentation pipeline | ✅ done |
| `scripts/08_train_rfdetr_seg.sh` | Train RF-DETR-Seg locally | ✅ done |
| `train_hipe1.py` | HiPE1 training (Seg-Medium) | ✅ done |
| `train_hipe2.py` | HiPE2 training (Seg-Large) | ✅ done |
| `Dockerfile` | Container for HiPE training | ✅ done (fix applied) |
| `hipe_deployment_guide.sh` | Step-by-step HiPE deployment | ✅ done |

---

## 10. Thesis Evaluation Plan

### Phase 3b RF-DETR-Seg (when training completes)
Compare these in the thesis:
| Model | mAP@50 | mAP@50:95 | Latency | Notes |
|-------|--------|-----------|---------|-------|
| RF-DETR-Seg-M COCO pretrained | 68.4 | 45.3 | 5.9ms | Zero-shot baseline (from paper) |
| RF-DETR-Seg-M fine-tuned | ? | ? | 5.9ms | HiPE1 result |
| RF-DETR-Seg-L fine-tuned | ? | ? | 8.8ms | HiPE2 result |
| SAM2.1 Hiera Large (teacher) | — | — | ~500ms | Reference only |

### Phase 4 OC-SORT (upcoming)
| Metric | Dataset | Notes |
|--------|---------|-------|
| MOTA | CVB | Ground truth track IDs available |
| MOTP | CVB | |
| IDF1 | CVB | Primary tracking metric |
| ID Switches | CVB | Lower is better |
| Qualitative | CBVD-5 | No ground truth, visual inspection |

---

## 11. What the Next Chat Should Focus On

**Immediate priority (unblock Phase 3b):**
1. Fix the Docker transformers version issue (see Section 3 blocker details)
2. Successfully launch training on HiPE1 and HiPE2
3. Wait for training (~4-6 hours), retrieve results
4. Run `eval_rfdetr_seg.py` and `plot_training_curves.py`
5. Interpret results — which config wins, what's the SAM2 quality gap

**After Phase 3b is done:**
1. Begin Phase 4 — OC-SORT tracking
2. Design the tracking pipeline: load segmentation JSONs → run OC-SORT → save track JSONs
3. Evaluate on CVB ground truth (MOTA, IDF1, ID Switches)

**Important context for the assistant:**
- This is a master's thesis — every design decision needs a written justification
- All experiments need: motivation, implementation details, quantitative results, discussion
- The RF-DETR-Seg distillation is a novel contribution (SAM2 pseudo-labels → fast deployable model)
- The assistant has been deeply involved in all phases and knows the codebase well
- The conda environment on local machine is `cattletransformer`
- Python is 3.10, PyTorch 2.1, CUDA 12.x

---

## 12. Pipeline Architecture (Updated)

```
INPUT: Raw Video
    │
    ▼
🔲 Phase 0: Video Ingestion (post-thesis)
    │
    ▼
✅ Phase 2: RF-DETR Detection (70.4% mAP@50)
    │  Output: bbox + confidence per frame
    ▼
✅ Phase 3: SAM2 Segmentation (242,689 masks, 100.3% coverage)
    │  Output: pixel masks per detection
    │
    ├──→ 🔄 Phase 3b: RF-DETR-Seg Distillation
    │         SAM2 masks → train RF-DETR-Seg → 83x faster production model
    │         HiPE1: Seg-Medium lr=1e-4 | HiPE2: Seg-Large lr=5e-5
    │
    ▼
🔲 Phase 4: OC-SORT Tracking
    │  Output: track_id per cow per frame
    ▼
🔲 Phase 5: Tubelet Generation
    │  Output: 16-frame clips per cow
    ▼
🔲 Phase 6: VideoMAE Behavior Classification
    │  Output: behavior label per clip
    ▼
🔲 Phase 7: Analytics
    │  Output: Gantt charts, activity budgets
    ▼
🔲 Phase 8: Report Generation (post-thesis)
```

---

*Cattle Vision Framework — Masters Thesis, Texas State University — 2026*  
*Context guide for chat continuation — generated March 7, 2026*
