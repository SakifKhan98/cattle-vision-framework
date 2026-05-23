# Dataset Analysis Notebooks — PRD

> GitHub Issue: https://github.com/SakifKhan98/cattle-vision-framework/issues/9

## Problem Statement

The thesis requires a rigorous, independently reproducible characterization of every dataset used in training and evaluation. Currently there are no analysis notebooks — the existing `01_dataset_exploration.ipynb` is empty. Without documented dataset provenance, class distributions, annotation format walkthroughs, and label mapping tables, the thesis methodology chapter cannot be written and reviewers cannot reproduce dataset statistics.

## Solution

Produce seven Jupyter notebooks that collectively characterize all six datasets from raw data alone (never from pipeline outputs), save machine-readable summary statistics, and generate cross-dataset comparison figures suitable for direct inclusion in the thesis.

All notebooks live in `notebooks/`. Each per-dataset notebook is independently runnable and thesis-citeable. A final comparison notebook reads the summary JSONs and produces three cross-dataset thesis figures.

## User Stories

1. As a thesis author, I want a notebook for CBVD-5 that parses the VIA CSV and AVA annotation files from raw, so that I can cite exact video counts, annotation counts, and class distributions in the methodology chapter.
2. As a thesis author, I want the CBVD-5 notebook to display sample frames from the raw videos, so that I have a visual reference showing the indoor surveillance setting.
3. As a thesis author, I want the CBVD-5 notebook to show the 5-class label distribution as a bar chart, so that I can identify class imbalance before discussing it in the thesis.
4. As a thesis author, I want the CBVD-5 notebook to document the AVA annotation format by parsing and displaying one sample row, so that the raw data contract is captured in a citeable artifact.
5. As a thesis author, I want the CBVD-5 notebook to produce a `results/analysis/cbvd5_summary.json`, so that the comparison notebook can ingest it without re-running analysis.
6. As a thesis author, I want a notebook for CVB that parses the per-clip COCO JSON annotations from raw, so that I can cite the 502-clip structure, frame counts, and behavior label distribution.
7. As a thesis author, I want the CVB notebook to show the Supervisely-style string behavior labels and their mapping to the 7-class taxonomy, so that the label alignment decision is documented inline.
8. As a thesis author, I want the CVB notebook to surface any frames present in annotation files but missing on disk (a known CVB gotcha), so that the preprocessing gap is quantified.
9. As a thesis author, I want the CVB notebook to display sample frames from multiple camera angles (GoPro at 4 corners), so that the outdoor pasture setting is visually documented.
10. As a thesis author, I want the CVB notebook to produce `results/analysis/cvb_summary.json`.
11. As a thesis author, I want a notebook for the Freeman Center dataset that reads the YOLO-format labels from `CMB_dataset/train/labels/`, `val/labels/`, and `test/labels/`, so that I can cite frame counts per split and the 9-class behavior distribution.
12. As a thesis author, I want the Freeman Center notebook to display the 9-class behavior taxonomy (from `CMB_config.yml`) alongside its tentative mapping to the 7-class canonical taxonomy, so that the label alignment is documented before zero-shot OOD evaluation.
13. As a thesis author, I want the Freeman Center notebook to show the class frequency distribution so that the "normal" class volume is visible, because the decision to map or drop it is deferred to this analysis.
14. As a thesis author, I want the Freeman Center notebook to flag the open label-mapping decisions (normal, dominance assertion, fear response, vocalizing, sniffing) explicitly as a notebook section, so that a future reader can see what was confirmed vs. tentative.
15. As a thesis author, I want the Freeman Center notebook to display sample frames from the trail camera footage, so that the pasture and lighting conditions are documented.
16. As a thesis author, I want the Freeman Center notebook to produce `results/analysis/freeman_summary.json`.
17. As a thesis author, I want a notebook for Cows2021 that parses the DatasetNinja/Supervisely JSON annotations from the train, val, and test splits, so that I can cite image counts (7248 / 1023 / 2131) and bounding box counts.
18. As a thesis author, I want the Cows2021 notebook to have a clearly labelled "Evaluation Scope" section stating that this dataset has no behavior labels and is used only for detection mAP and tracking IDF1 evaluation, so that its role in Phase 8 is unambiguous.
19. As a thesis author, I want the Cows2021 notebook to show the detection annotation class distribution (`cattle torso` box counts per split), so that the detection-only role is quantified.
20. As a thesis author, I want the Cows2021 notebook to display sample images showing the top-down Holstein-Friesian coat pattern, so that the distinctive black-and-white appearance is documented.
21. As a thesis author, I want the Cows2021 notebook to produce `results/analysis/cows2021_summary.json`.
22. As a thesis author, I want a notebook for OpenCows2020 that parses the DatasetNinja JSON annotations, so that I can cite total image and annotation counts (7043 images) with no train/val/test split in the source data.
23. As a thesis author, I want the OpenCows2020 notebook to have a "Evaluation Scope" section stating it is used for detection mAP evaluation only, so that its Phase 8 role is unambiguous.
24. As a thesis author, I want the OpenCows2020 notebook to show detection box counts per image as a histogram, so that annotation density is documented.
25. As a thesis author, I want the OpenCows2020 notebook to display sample frames, so that the top-down in-barn overhead view is visually documented.
26. As a thesis author, I want the OpenCows2020 notebook to produce `results/analysis/opencows2020_summary.json`.
27. As a thesis author, I want a notebook for CattleEyeView that parses the YOLO-format labels from the `detect/labels/{train,val,test}` subdirectory structure (frames are nested inside per-video subfolders), so that I can cite the 14-video, 30,703-frame, 753-instance dataset scale.
28. As a thesis author, I want the CattleEyeView notebook to have an "Evaluation Scope" section stating it is used for detection mAP, Mask IoU, and IDF1 evaluation, so that its Phase 8 role is unambiguous.
29. As a thesis author, I want the CattleEyeView notebook to show the train/test split distribution (75%/25%), so that the evaluation setup mirrors the paper's benchmark.
30. As a thesis author, I want the CattleEyeView notebook to display sample frames showing the top-down loading-ramp view across multiple cattle breeds and coat colors, so that the challenging setting is visually documented.
31. As a thesis author, I want the CattleEyeView notebook to produce `results/analysis/cattleeyeview_summary.json`.
32. As a thesis author, I want a `dataset_comparison.ipynb` that reads all six `results/analysis/{dataset}_summary.json` files, so that cross-dataset figures can be produced without re-running individual analysis notebooks.
33. As a thesis author, I want the comparison notebook to produce a side-by-side behavior class distribution figure across the datasets that have behavior labels (CBVD-5, CVB, Freeman Center), so that label imbalance patterns are comparable at a glance.
34. As a thesis author, I want the comparison notebook to produce a dataset scale comparison figure (total annotation counts and image/frame counts per dataset), so that the relative sizes of training vs. evaluation datasets are immediately visible.
35. As a thesis author, I want the comparison notebook to produce an evaluation scope summary table (one row per dataset: format, resolution, behavior labels yes/no, evaluation role), so that the thesis has a single reference table for all datasets.

## Implementation Decisions

### Notebook naming and location

All analysis notebooks live in `notebooks/dataset_analysis/` (separate from the numbered pipeline notebooks in `notebooks/`).

| Notebook | Dataset |
| --- | --- |
| `notebooks/dataset_analysis/analysis_cbvd5.ipynb` | CBVD-5 |
| `notebooks/dataset_analysis/analysis_cvb.ipynb` | CVB |
| `notebooks/dataset_analysis/analysis_freeman.ipynb` | Freeman Center |
| `notebooks/dataset_analysis/analysis_cows2021.ipynb` | Cows2021 |
| `notebooks/dataset_analysis/analysis_opencows2020.ipynb` | OpenCows2020 |
| `notebooks/dataset_analysis/analysis_cattleeyeview.ipynb` | CattleEyeView |
| `notebooks/dataset_analysis/dataset_comparison.ipynb` | Cross-dataset comparison |

### Standard 8-section structure (all per-dataset notebooks)

Every notebook follows this section order, with substitutions for detection-only datasets:

| #   | Section                                                                                                | Detection-only substitution                                     |
| --- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| 1   | Dataset Overview — origin, paper citation, license, intended use, Phase 8 evaluation role              | (same)                                                          |
| 2   | Data Inventory — file counts, split sizes (train/val/test), video vs. image format, total size on disk | (same)                                                          |
| 3   | Sample Image Visualization — 6–9 representative frames in a grid                                       | (same)                                                          |
| 4   | Annotation Format — parse one sample annotation file/row and display its structure                     | (same)                                                          |
| 5   | Class Distribution — behavior label counts and bar chart                                               | Detection box counts per split                                  |
| 6   | Resolution & Quality — frame dimensions, fps for video datasets, missing/corrupt file check            | (same)                                                          |
| 7   | Label Mapping to 7-class Taxonomy — explicit mapping table                                             | "Evaluation Scope" statement — no behavior labels, Phase 8 role |
| 8   | Preprocessing Gap Analysis — what must happen before this dataset enters the pipeline                  | (same)                                                          |

### Re-derive from raw data only

All statistics (counts, distributions, resolutions) are computed by reading raw annotation files and image/video files directly from `data/raw/`. No pipeline output files (`data/processed/`, `results/`) are read inside per-dataset notebooks.

### Raw annotation formats by dataset

| Dataset        | Format                        | Key files                                                                       |
| -------------- | ----------------------------- | ------------------------------------------------------------------------------- |
| CBVD-5         | VIA CSV + AVA CSV             | `data/raw/cbvd5/CBVD-5.csv`, `annotations/ava_{train,val}_v2.1.csv`             |
| CVB            | Per-clip COCO JSON            | `data/raw/cvb/annotations/{clip_id}/annotations/instances_default.json`         |
| Freeman Center | YOLO `.txt` per frame         | `data/raw/freeman-cmb-2024/CMB_dataset/CMB_dataset/{train,val,test}/labels/`    |
| Cows2021       | DatasetNinja/Supervisely JSON | `data/raw/cows2021/{split}/ann/{image}.jpg.json`                                |
| OpenCows2020   | DatasetNinja/Supervisely JSON | `data/raw/opencow2020/detection_and_localisation/ann/`                          |
| CattleEyeView  | YOLO `.txt` per frame         | `data/raw/cattle-eye-view/dataset/detect/labels/{split}/{video_id}/{frame}.txt` |

**CBVD-5 AVA format** — columns: `video_id, timestamp, x1, y1, x2, y2, action_id, entity_id`. Action IDs 1–5 map to canonical IDs via `_CBVD5_ACTION_MAP` in `src/data/label_utils.py`.

**CVB behavior label** — encoded in the clip folder name as `beh{N}`. String→ID mapping via `_CVB_BEHAVIOR_MAP` in `src/data/label_utils.py`.

**Freeman Center class definitions** — 9 classes (IDs 0–8) defined in `data/raw/freeman-cmb-2024/CMB_config.yml`:

```
0: hay feeding  → Foraging/Grazing (2)   [resolved]
1: grazing      → Foraging/Grazing (2)   [resolved]
2: normal       → TBD                    [OPEN DECISION]
3: dominance assertion → Other (6)       [tentative]
4: ruminating   → Ruminating (4)         [resolved]
5: fear response → Other (6)            [tentative]
6: grooming     → Grooming (5)           [resolved]
7: vocalizing   → Other (6)             [tentative]
8: sniffing     → Other (6)             [tentative]
```

**Cows2021 / OpenCows2020 JSON structure** — each file has a `size` dict (`height`, `width`) and an `objects` list; each object has `classTitle` and `points.exterior` (two corner coordinates).

**CattleEyeView** — YOLO format, one class (`cow`, ID 0). Label files are nested one level deeper than flat YOLO: `labels/{split}/{video_id}/{frame}.txt`. Split config in `detect/cattleeyeview_detect.yaml`.

### Reuse existing label_utils

Notebooks import `src/data/label_utils.py` for CBVD-5 and CVB label parsing rather than duplicating the mapping tables inline.

### Summary JSON schema

Each per-dataset notebook writes `results/analysis/{dataset}_summary.json`:

```json
{
  "dataset": "cbvd5",
  "total_images_or_frames": 206100,
  "total_annotations": 38200,
  "splits": { "train": 0, "val": 0, "test": 0 },
  "class_distribution": {
    "0": 4200,
    "1": 3100,
    "2": 8000,
    "3": 1500,
    "4": 7000
  },
  "resolution": { "width": 1920, "height": 1080 },
  "has_behavior_labels": true,
  "evaluation_scope": ["behavior_f1", "in_domain"]
}
```

Integer counts only. Class keys are string canonical IDs (`"0"`–`"6"`) for behavior datasets, or raw class names for detection-only datasets.

### Freeman Center open decisions (flagged inline)

The `analysis_freeman.ipynb` Label Mapping section must mark decisions explicitly:

- **OPEN DECISION** — `normal` → Standing (0) OR drop from evaluation. Decide after reviewing class frequency distribution in this notebook.
- **TENTATIVE** — `dominance assertion`, `fear response`, `vocalizing`, `sniffing` → Other (6). Confirm against paper.

### Creation order

CBVD-5 → CVB → Freeman Center → Cows2021 → OpenCows2020 → CattleEyeView → dataset_comparison

### results/analysis/ directory

Create `results/analysis/` if it does not exist. This directory is committed (consistent with the project rule that `results/` is committed).

### Runtime requirements

No GPU required. All notebooks must run on CPU with standard packages: `numpy`, `pandas`, `matplotlib`, `Pillow`, `opencv-python`.

## Testing Decisions

Notebooks are not unit-tested. Correctness is verified by:

- **Self-consistency check** — each notebook prints a summary table at the end comparing computed counts against numbers stated in the dataset paper (sourced from `docs/dataset-papers/*.md`). Mismatches are flagged with a `⚠️` cell.
- **Summary JSON schema check** — the last cell of every per-dataset notebook validates the written JSON against required schema keys and asserts all counts are non-zero integers before saving.
- **Comparison notebook smoke test** — `dataset_comparison.ipynb` asserts all six summary JSONs exist and are readable before producing figures, and raises a clear error if one is missing.

Label mapping logic reused by notebooks is already tested in `tests/` via `src/data/label_utils.py`.

## Out of Scope

- Reprocessing or re-running any pipeline stage (detection, tracking, tubelet generation)
- Training or evaluating models inside notebooks
- Committing the dataset paper markdown files (`docs/dataset-papers/*.md`) to git
- Creating analysis notebooks for any dataset not listed (AVA, Kinetics, etc.)
- Resolving the Freeman Center "normal" class mapping — flagged as open, deferred to human review of the class distribution output
- Downloading or fetching datasets not already on disk at `data/raw/`

## Further Notes

- **CBVD-5 test = val** — no separate test labels were released. Document in Preprocessing Gap Analysis and reflect in `splits` field of the summary JSON.
- **CVB missing frames** — not all annotated frames have corresponding image files on disk. The Resolution & Quality section must count and report the number of missing frames. This is a known issue in `src/data/convert_cvb.py`.
- **CattleEyeView nested labels** — label files are at `detect/labels/{split}/{video_id}/{frame}.txt`, not flat. File traversal must recurse one level deeper than a standard YOLO dataset.
- **Dataset paper markdown files** — co-located at `docs/dataset-papers/{name}.md`, gitignored. Use as reading reference when writing Dataset Overview and Annotation Format sections. Do not embed or reference them in notebook cells.
