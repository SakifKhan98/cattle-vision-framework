# Training and Evaluation Logs

HiPE1 logs from VideoMAE Phase 6 training and evaluation. All runs used
`docker run --gpus all --shm-size=16g cattle-behavior ...` on HiPE1 Tesla V100.

---

## Training Logs

| File | Config | Best Macro-F1 | Best Epoch | Notes |
|------|--------|---------------|------------|-------|
| `combined.log` | Config 5: CBVD-5 + CVB joint training | 0.7537 | 4 | Primary analytics model |
| `cvb.log` | Config 2: CVB in-domain | 0.7607 | 4 | Highest per-dataset F1 |
| `cbvd5.log` | Config 1: CBVD-5 in-domain | 0.3149 | 4 | 5-class macro more appropriate |
| `cbvd5_to_cvb.log` | Config 3: CBVD-5 → CVB OOD | 0.1690 | — | Large domain gap finding |
| `cvb_to_cbvd5.log` | Config 4: CVB → CBVD-5 OOD | 0.1789 | 7 | Large domain gap finding |
| `combined_v2.log` | Config 5 re-run for verification | — | — | Sanity check run |

## Evaluation Logs

| File | Config | Notes |
|------|--------|-------|
| `eval_combined.log` | Config 5 evaluation | Wrote `results/behavior/predictions/combined_val.csv` |
| `eval_cvb.log` | Config 2 evaluation | Wrote `results/behavior/predictions/cvb_val.csv` |
| `eval_cbvd5.log` | Config 1 evaluation | Wrote `results/behavior/predictions/cbvd5_val.csv` |
| `eval_cbvd5_to_cvb.log` | Config 3 evaluation | Wrote `results/behavior/predictions/cbvd5_to_cvb_val.csv` |
| `eval_cvb_to_cbvd5.log` | Config 4 evaluation | Wrote `results/behavior/predictions/cvb_to_cbvd5_val.csv` |

## Other

| File | Notes |
|------|-------|
| `cvb_reexport.log` | CVB tubelet re-export run (2026-04-29) after relaxing frame presence threshold to 12/16 |

---

## What to Look For

- **Training:** epoch-by-epoch macro-F1 is the key metric. Loss curves can be plotted with
  `notebooks/04_behavior_results.ipynb` or `scripts/plot_training_results.py`.
- **Evaluation:** final per-class F1 summary is printed at the end of each eval log.
  Ground truth numbers are in `results/behavior/f1_per_class.csv`.
- **cudnn warnings** (`UserWarning: TF32 ... cudnn`) appear in all logs — harmless.
  PyTorch uses a fallback convolution plan; results are unaffected.
