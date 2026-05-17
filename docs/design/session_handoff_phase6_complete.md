# Session Handoff — Phase 6 Complete

**Date:** 2026-05-06
**Working directory:** `~/TXST/Thesis/cattle-vision-framework/one_day/`

---

## What Was Completed This Session

### 1. Phase 6 Training — All 5 Configs Done (on HiPE1)

All 5 VideoMAE configs trained to completion on HiPE1 (`/home/zxs12/cattle_behavior/`).

| Config | YAML | Best Macro-F1 | Best Epoch |
|--------|------|---------------|------------|
| 5 — Combined | `videomae_combined.yaml` | 0.7537 | 4 |
| 2 — CVB in-domain | `videomae_cvb.yaml` | 0.7607 | 4 |
| 1 — CBVD-5 in-domain | `videomae_cbvd5.yaml` | 0.3149 | 4 |
| 3 — OOD CBVD-5→CVB | `videomae_cbvd5_to_cvb.yaml` | 0.1690 | 4 |
| 4 — OOD CVB→CBVD-5 | `videomae_cvb_to_cbvd5.yaml` | 0.1789 | 7 |

### 2. Phase 6 Evaluation — All 5 Configs Done (on HiPE1)

Ran `evaluate.py` for all 5 configs via Docker on HiPE1. Key fix: `--shm-size=16g` required or DataLoader throws `RuntimeError: No space left on device`.

All outputs retrieved via rsync. Everything is now local.

### 3. All Results Now Local

| File/Dir | Status |
|----------|--------|
| `runs/behavior/videomae_*/checkpoint_best.pt` | ✓ 5 × 330 MB |
| `results/behavior/predictions/*_val.csv` | ✓ 5 CSVs |
| `results/behavior/confusion_matrices/*.png` | ✓ 5 PNGs |
| `results/behavior/f1_per_class.csv` | ✓ All 5 rows |
| `logs/eval_*.log` | ✓ 5 logs |

### 4. New Script Written

`src/tracking/render_behavior_video.py` — full pipeline behavior video renderer.
Combines `tracking_v2/` JSONs (bboxes + SAM2 masks) with predictions CSV to render MP4 with per-cow behavior labels + confidence. Overlapping tubelets resolved by averaging logits per frame.

### 5. Docs Updated

`docs/phase6_report.md` — updated with:
- All 5 config training curves and per-class F1 breakdowns
- Interpretation of each result
- Section 11: rsync retrieval commands (HiPE1 home = `/home/zxs12/`)
- Section 12: eval commands for HiPE1 (with `--shm-size=16g`) and locally
- Section 13: behavior video rendering usage
- Section 14: updated task status table

---

## Key Numbers for Thesis

From `results/behavior/f1_per_class.csv`:

| Config | Macro-F1 | Standing | Lying | Foraging | Drinking | Ruminating | Grooming | Other |
|--------|----------|----------|-------|----------|----------|------------|----------|-------|
| CVB in-domain | **0.7607** | 0.860 | 0.845 | 0.979 | 0.881 | 0.801 | 0.715 | 0.243 |
| Combined | 0.7537 | 0.870 | 0.823 | 0.980 | 0.876 | 0.772 | 0.722 | 0.233 |
| CBVD-5 in-domain | 0.3149 | 0.906 | 0.303 | 0.896 | 0.000 | 0.100 | 0.000 | 0.000 |
| CVB→CBVD-5 OOD | 0.1789 | 0.675 | 0.145 | 0.432 | 0.000 | 0.000 | 0.000 | 0.000 |
| CBVD-5→CVB OOD | 0.1690 | 0.215 | 0.281 | 0.006 | 0.188 | 0.493 | 0.000 | 0.000 |

**Thesis notes:**
- Report Config 1 (CBVD-5 in-domain) as 5-class macro-F1 — 7-class denominator is unfair (Drinking=0 val samples, Grooming/Other absent from CBVD-5)
- OOD results (~0.17) are a key finding: large domain gap between indoor CBVD-5 and outdoor CVB
- `Other` class F1 ~0.23 is a dataset limitation (catch-all class), not model failure
- cudnn warnings in all logs are harmless (PyTorch fallback conv plan)

---

## What To Do Next Session

### Priority 1 — Update phase6_report.md

The doc already has training results and structure. Add the final confirmed per-class F1 numbers from `results/behavior/f1_per_class.csv` into each config's section. Numbers are now ground truth (from evaluate.py), not just from training logs.

### Priority 2 — Render Behavior Videos

Prerequisites met (predictions CSVs exist). Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate cattletransformer
cd ~/TXST/Thesis/cattle-vision-framework/one_day

python src/tracking/render_behavior_video.py \
    --dataset cvb --auto \
    --predictions results/behavior/predictions/videomae_combined_v1_val.csv
```

Output → `results/tracking/behavior_videos/{video_id}_behavior.mp4`

### Priority 3 — Phase 7 Analytics

`src/analytics/timeline.py` and `src/analytics/budget.py` are stubs. Implement per the plan in `docs/phase5_7_plan.md` §7.

Inputs needed:
- `results/behavior/predictions/videomae_combined_v1_val.csv` (per-tubelet predictions with logits) ✓
- `data/processed/tracking_v2/{cbvd5,cvb}/` (track JSONs) ✓

Outputs:
- `results/analytics/timelines/{video_id}_{track_id}.csv`
- `results/analytics/activity_budget.csv`
- `results/analytics/transition_matrix.csv`
- `results/analytics/welfare_flags.csv`

---

## Conda Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cattletransformer
```

`conda` is not in PATH by default — must source first or run `~/miniconda3/bin/conda init bash` once.

---

## HiPE1 Notes

- SSH alias: `hipe1`
- Home dir on server: `/home/zxs12/cattle_behavior/`  (NOT `/home/sakif/`)
- All training/eval complete — server no longer needed until Phase 7 (analytics runs locally)
- Docker eval commands require `--shm-size=16g` (64 MB default causes OOM on DataLoader workers)
