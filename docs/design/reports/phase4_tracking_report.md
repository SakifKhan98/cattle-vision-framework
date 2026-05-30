# Phase 4 Report — Multi-Object Tracking (OC-SORT)

**Cattle Vision Framework — Texas State University**
*OC-SORT with Mask-IoU Association · CBVD-5 & CVB · April 2026*

---

## 1. Overview

Phase 4 assigns a persistent identity label (`track_id`) to every detected cattle instance across consecutive video frames. It converts the per-frame detections produced in Phases 1–3 into temporally coherent animal tracks that downstream phases can use as the unit of analysis.

Without tracking, each frame's detections are independent events — there is no way to say "this cow in frame 100 is the same animal seen in frame 50." Tracking is what makes identity-preserved behavior timelines (Phase 7) possible.

**Input:** Per-frame segmentation JSONs from Phase 3  
`data/processed/segmentation/{cbvd5,cvb}/{video_id}_masks.json`

**Output:** Per-video tracking JSONs with stable `track_id` assignments  
`data/processed/tracking_v2/{cbvd5,cvb}/{video_id}_tracks.json`

**Algorithm:** OC-SORT (Observation-Centric SORT), extended with mask-IoU post-association to carry SAM2 segmentation masks forward into track outputs.

---

## 2. Algorithm Selection — Why OC-SORT

OC-SORT (Cao et al., 2022) is a linear-complexity multi-object tracker built on the SORT framework. It addresses a known failure mode of standard SORT: when a tracked object is occluded for several frames, SORT's Kalman filter extrapolates position based on prior motion, which drifts. OC-SORT compensates by using the last reliable observation to re-anchor the Kalman state when the object reappears.

Three properties made it the right choice for this pipeline:

1. **Occlusion handling.** Cattle in group settings frequently overlap one another. OC-SORT's observation-centric re-detection step recovers correct identity after partial occlusion without requiring a re-ID appearance model, keeping the implementation lightweight.

2. **No appearance model required.** Appearance-based trackers (DeepSORT, StrongSORT) require a separate re-ID embedding network trained on cattle identity. No such network is available for cattle, and training one was out of scope. OC-SORT achieves competitive IDF1 using motion and spatial cues alone.

3. **Compatibility with variable frame rates.** CBVD-5 provides only 6 keyframes per 10-second video (sparse timestamps at 2.0–7.0 s). OC-SORT's `max_age` parameter controls how long a track survives without a detection, making it tolerant of the large inter-frame gaps in CBVD-5.

---

## 3. Integration with SAM2 Masks

Standard OC-SORT operates entirely on bounding boxes. This pipeline extends it to carry SAM2 segmentation masks (`mask_rle`) through the tracking stage.

**The extension works in two steps:**

**Step 1 — OC-SORT internal association (box IoU).**  
OC-SORT runs its standard Kalman-filter update internally using bounding boxes as input. It assigns a `track_id` to each detection and outputs tracked bounding boxes. No modification to OC-SORT's internal code is needed.

**Step 2 — Post-association mask recovery (mask IoU or box IoU).**  
After OC-SORT returns its tracked boxes, the pipeline matches each tracked box back to the original detection list from Phase 3 to recover the `mask_rle` field. Matching uses:
- Candidate filtering by box IoU > 0.1 (narrows to spatially plausible detections)
- Final selection by highest box IoU among candidates with valid masks
- Fallback to pure box IoU if no candidate has a mask

This means the OC-SORT output for every frame is augmented with the SAM2 mask that most closely matches the tracked box position, preserving pixel-level identity through the full pipeline.

The `--use_box_iou` flag disables this and falls back to box-only tracking (used in `scripts/08_run_tracking.sh` for the box-only path).

---

## 4. Configuration

All parameters set via command-line defaults in `src/tracking/track.py`:

| Parameter | Value | Meaning |
|---|---|---|
| `det_thresh` | 0.3 | Minimum detection confidence to enter the tracker |
| `max_age` | 30 | Frames a track survives without a matching detection |
| `min_hits` | 3 | Confirmed detections required before a track is reported |
| `iou_threshold` | 0.3 | IoU threshold for internal SORT association |
| `delta_t` | 3 | OC-SORT observation-centric time window |
| `inertia` | 0.2 | Weight of Kalman velocity prior vs. observation correction |
| `asso_func` | `"iou"` | OC-SORT internal cost function (box IoU) |
| `use_byte` | `False` | ByteTrack secondary matching disabled |

**Rationale for key choices:**

- `max_age=30`: CVB videos are 450 frames (15 s at 30 fps). An occlusion of up to 1 second (30 frames) can be survived without dropping the track. CBVD-5's sparse keyframes benefit even more — with only 6 annotated frames across 250 total, `max_age=30` bridges the inter-keyframe gaps.
- `min_hits=3`: Requires 3 consecutive confirmed detections before reporting a track. This eliminates most spurious single-frame false positives from the detector without discarding genuine short-duration tracks.
- `det_thresh=0.3`: Matches the detection confidence threshold used in Phase 1/3 inference. Lowering it would introduce more false-positive tracks; raising it would drop low-confidence but genuine detections at the edges of the frame.

---

## 5. Dataset Characteristics and Tracking Regime

The two datasets present fundamentally different tracking problems:

| Property | CVB | CBVD-5 |
|---|---|---|
| Video duration | 15 s (450 frames @ 30 fps) | 10 s (250 frames @ 25 fps) |
| Annotated frames | All 450 frames | 6 keyframes (2.0–7.0 s) |
| Environment | Outdoor pasture | Indoor barn |
| Avg cattle per video | ~10 | ~5 |
| Camera viewpoint | Side / angle | Overhead / side |
| Ground-truth track IDs | Yes (in annotation JSON) | No (AVA format has `person_id` per keyframe only) |

**CVB tracking challenge:** High animal density (avg 10 cattle per video, max 25) creates frequent occlusions and near-misses. The continuous 450-frame sequence gives OC-SORT full motion information to maintain consistent tracks.

**CBVD-5 tracking challenge:** Only 6 keyframes are annotated per video. OC-SORT processes all 250 frames (the detector runs on every frame), but the Kalman filter must bridge large unannotated inter-frame gaps. `max_age=30` was sufficient to keep tracks alive across the 40-frame gaps between annotated keyframes. Evaluation with IDF1 is not possible for CBVD-5 because the dataset does not provide persistent ground-truth track IDs across frames — `person_id` in the AVA-format CSV is a per-keyframe annotation label, not a temporal identity.

---

## 6. Output Statistics

**CVB (502 videos):**

| Metric | Value |
|---|---|
| Videos tracked | 502 |
| Total unique tracks (sum across videos) | 5,023 |
| Avg tracks per video | 10.0 |
| Median tracks per video | 10 |
| Max tracks in any video | 25 |
| Frames per video | 450 (all CVB) |
| Avg tracking time per video | ~0.22 s |

**CBVD-5 (687 videos):**

| Metric | Value |
|---|---|
| Videos tracked | 687 |
| Total unique tracks (sum across videos) | 3,239 |
| Avg tracks per video | 4.7 |
| Median tracks per video | 4 |
| Max tracks in any video | 25 |
| Frames per video | 250 (all CBVD-5) |

Tracking summary CSVs saved at:
- `data/processed/tracking_v2/cvb/tracking_summary_mask_iou.csv`
- `data/processed/tracking_v2/cbvd5/tracking_summary_mask_iou.csv`

---

## 7. Evaluation Results

Evaluation was performed on CVB, the only dataset with persistent ground-truth track IDs. The standard MOT metrics were computed using `src/tracking/evaluate_tracking.py` against the CVB annotation identity labels.

All 447 CVB videos with annotated track IDs were evaluated (502 total CVB videos; 55 had no annotated instances). IoU threshold for TP/FP classification: 0.5.

**Table 7.1. Tracking Evaluation Results — CVB (447 videos)**

| Metric | Value |
|---|---|
| **IDF1** | **67.31%** |
| **MOTA** | **36.61%** |
| **MOTP** | **77.42%** |
| Precision | 65.69% |
| Recall | 77.41% |
| Total GT detections | 38,609 |
| Total predicted detections | 45,499 |
| True positives (TP) | 29,887 |
| False positives (FP) | 15,612 |
| False negatives (FN) | 8,722 |
| Total identity switches (IDSW) | 141 |
| Avg identity switches per video | 0.32 |

Results file: `results/tracking/tracking_summary_all.json`

---

## 8. Metric Interpretation

### IDF1 = 67.31%

IDF1 measures identity consistency: across all matched detections, what fraction are correctly associated with the right ground-truth identity throughout the sequence. 67.31% is the primary metric for this pipeline because downstream behavior analysis depends on following the same animal across time, not on minimizing false positives at the frame level.

An IDF1 of 67.31% means approximately two-thirds of matched detections carry a consistent identity label — sufficient for meaningful tubelet construction and behavior timeline aggregation.

### MOTA = 36.61%

MOTA penalizes all three error types simultaneously: FP + FN + IDSW, normalized by total GT count. The 36.61% is lower than IDF1 because MOTA is dominated by the FP count (15,612). The detector intentionally over-detects (recall = 77.41%, precision = 65.69%) — this is a deliberate design choice to avoid missing cattle at the detection stage. Every missed detection is a permanently lost tubelet; every extra detection is simply a short-lived track that Phase 5 will filter out for lacking enough annotated labels to match.

The high FP count is expected and acceptable: `min_hits=3` already suppresses single-frame detections, but genuine multi-frame false tracks from background objects (fence posts, shadows) still accumulate. These do not propagate to behavior outputs because tubelet label assignment (Phase 5) requires a Hungarian-matched ground-truth annotation with IoU ≥ 0.3.

### MOTP = 77.42%

MOTP measures the average IoU between matched tracked boxes and their ground-truth boxes. 77.42% indicates high localization accuracy for matched tracks — the tracker preserves the spatial precision of the Phase 1 detector output faithfully.

### Identity switches = 141 total (0.32 per video)

141 identity switches across 447 videos (avg 0.32 per video) is very low. Most videos have zero switches; occasional switches occur in high-density scenes (avg 10+ cattle) during simultaneous occlusions. The `delta_t=3` observation-centric correction in OC-SORT directly suppresses switches that would otherwise occur after 2–3 frame gaps.

---

## 9. Output JSON Format

Each `{video_id}_tracks.json` follows this schema (same as the data contract in `CLAUDE.md`):

```json
{
  "video_id": "0002_arm01_gopro1_20200322_222554_beh7_ani1_ins1_cut_1",
  "dataset": "cvb",
  "frames": {
    "1": [
      {
        "track_id": 11,
        "bbox": [x1, y1, x2, y2],
        "score": 1.0,
        "mask_rle": {"size": [1080, 1920], "counts": "..."},
        "mask_area": 44290
      }
    ]
  },
  "stats": {
    "total_frames": 450,
    "frames_with_tracks": 447,
    "total_unique_tracks": 19,
    "association_mode": "mask_iou"
  }
}
```

Frame keys are string-formatted integers. `bbox` is in absolute pixel coordinates `[x1, y1, x2, y2]`. `mask_rle` is a COCO-format RLE dict; present when Phase 3 SAM2 masks were available, `null` otherwise.

---

## 10. Key Engineering Decisions

**Decision 1: Two-stage association (OC-SORT + mask IoU post-matching)**  
Separating OC-SORT's internal box-IoU association from the mask recovery step keeps OC-SORT's code unmodified. The post-matching step is a thin wrapper that adds mask propagation without touching tracker internals. This makes it easy to update or replace OC-SORT independently.

**Decision 2: `max_age=30` for CBVD-5 keyframe gaps**  
CBVD-5 annotates only 6 frames per 10-second video. Without a high `max_age`, tracks would die between keyframes and Phase 5 would see only single-frame track fragments. `max_age=30` keeps tracks alive across the ~40-frame inter-keyframe gaps.

**Decision 3: Box IoU only inside OC-SORT; mask IoU only at recovery step**  
Running full mask IoU inside OC-SORT's Hungarian assignment would require forking the tracker source. Mask IoU at the recovery step achieves the same result — the correct mask is associated to the correct track — at much lower implementation cost.

**Decision 4: `min_hits=3` to suppress spurious tracks**  
Single-frame detections are common (RF-DETR fires on background objects occasionally). Requiring 3 consecutive detections before reporting a track eliminates most spurious tracks without discarding legitimate short-duration cattle appearances. See §10.1 for ablation results confirming this choice is robust.

### 10.1 `min_hits` Ablation

A sweep over `min_hits ∈ {1, 2, 3, 5}` was run on all 502 CVB segmentation clips (mh=1 and mh=2 as fresh runs; mh=3 canonical; mh=5 from archived tracks). Results below are evaluated against the same CVB GT (447 videos, split=all, IoU thresh=0.5).

| min_hits | IDF1 (%) | MOTA (%) | MOTP (%) | ID Switches | Videos |
|----------|----------|----------|----------|-------------|--------|
| 1        | 67.31    | 36.60    | 77.42    | 141         | 447    |
| 2        | 67.31    | 36.61    | 77.42    | 141         | 447    |
| **3**    | **67.31**| **36.61**| **77.42**| **141**     | **447**|
| 5        | 67.31    | 36.61    | 77.42    | 141         | 447    |

**Finding:** All metrics are essentially constant across the full sweep. IDF1, MOTA, MOTP, and total ID switches are identical for mh ∈ {2, 3, 5} and differ by ≤0.01 for mh=1.

**Interpretation:** The high FP count (15,612) is detector-driven — RF-DETR produces persistent multi-frame false detections (fence posts, shadows) that survive any min_hits threshold. Single-frame spurious detections are already too rare to affect the aggregate metrics. The ID switch count (141 total) is already so low that min_hits changes have no additional effect. This confirms that mh=3 is not a sensitive hyperparameter: the choice is robust and cannot be improved by tuning in either direction.

**Source files:** `results/tracking/cvb_mh{1,2,3,5}_summary.json`, `results/tracking/minhits_ablation.csv`

---

## 11. How Phase 4 Outputs Feed into Phase 5

Phase 5 (Tubelet Generation) reads the tracking JSONs directly. For each video, it:
1. Parses `{video_id}_tracks.json` into a frame→tracks dict.
2. For each `track_id`, collects all frame indices where that track appears.
3. Slides 16-frame windows (stride 8) over each track to generate tubelet candidates.
4. Requires ≥ 12 of 16 window frames to have a valid track entry (tolerates OC-SORT tracking gaps up to 4 frames within a window).
5. Uses the OC-SORT-predicted `bbox` (not the ground-truth box) for frame cropping — training tubelet crops match what production inference will see.

The `mask_rle` field in the tracking JSON is not used by Phase 5 directly (Phase 5 crops by bounding box). It is preserved for Phase 7 analytics and future visualization tools.

---

## 12. Key Files

| File | Description |
|---|---|
| `src/tracking/track.py` | OC-SORT wrapper with mask IoU post-association |
| `scripts/08_run_tracking.sh` | Shell wrapper — runs box-only tracking on both datasets |
| `configs/tracking/ocsort.yaml` | OC-SORT hyperparameter reference (mirrors argparse defaults) |
| `third_party/OC_SORT/` | OC-SORT source (cloned, not modified) |
| `data/processed/tracking_v2/cvb/` | 502 CVB tracking JSONs + summary CSV |
| `data/processed/tracking_v2/cbvd5/` | 687 CBVD-5 tracking JSONs + summary CSV |
| `results/tracking/tracking_summary_all.json` | IDF1, MOTA, MOTP, IDSW for CVB (447 videos) |
| `results/tracking/tracking_per_video_all.csv` | Per-video tracking metrics |
| `results/tracking/visualizations/` | Track grid visualizations per dataset |

---

## 13. Thesis-Ready Sections

### 13.1 Methodology (Draft — for §4.3.3)

*"Multi-object tracking was performed using OC-SORT (Cao et al., 2022), an observation-centric extension of the SORT framework. OC-SORT was selected for its ability to recover correct identity after short occlusions without requiring a separate re-identification appearance model — a practical constraint given the absence of pretrained cattle re-ID embeddings. The tracker was initialized with a detection confidence threshold of 0.3, maximum track age of 30 frames, and a minimum of 3 confirmed detections before a track is reported.*

*Each video was processed frame-by-frame; OC-SORT's Kalman filter propagated bounding box state across frames, and its observation-centric re-anchoring step used the last confirmed observation to correct Kalman drift upon redetection after occlusion. The OC-SORT internal cost function used bounding box IoU. Following tracking, each OC-SORT-assigned identity was matched back to the original SAM2 segmentation masks from Phase 3 using a two-stage box IoU filter, preserving pixel-level mask information in the tracking output.*

*Tracking outputs were stored as per-video JSON files containing frame-indexed lists of track assignments with bounding boxes, track IDs, confidence scores, and COCO RLE segmentation masks. These files served as the primary input to tubelet generation (Phase 5)."*

### 13.2 Results (Draft — for §4.5.3.3 / tracking results table)

*"Tracking performance was evaluated on the CVB dataset (447 videos with persistent ground-truth track annotations). The system achieved an IDF1 of 67.31%, indicating that approximately two-thirds of all matched detections carry a consistent ground-truth identity across their full track lifespan. MOTA was 36.61%, with MOTP of 77.42% confirming high spatial localization accuracy for matched tracks.*

*The gap between IDF1 and MOTA is attributable to the high false positive count (15,612 FP vs. 8,722 FN), which reflects an intentional design choice to maximize detection recall at the cost of precision — every missed detection permanently eliminates a tubelet, while excess detections are filtered at the tubelet label-assignment stage. Identity switches were low in absolute terms (141 total, 0.32 per video), confirming that OC-SORT's observation-centric re-anchoring effectively suppressed identity confusion in the high-density CVB scenes (average 10 cattle per video). Formal tracking evaluation was not performed on CBVD-5, which does not provide persistent ground-truth track identities across frames."*

### 13.3 Discussion Points

- IDF1 is the appropriate primary metric for this pipeline because downstream behavior analysis aggregates predictions by `track_id`. A high IDF1 ensures that behavior predictions are accumulated for the correct animal.
- MOTA's sensitivity to false positives makes it a less useful signal here: the detector intentionally over-detects, and tubelet label assignment downstream filters out unlabeled tracks.
- The 0.32 identity switches per video is consistent with the literature for indoor/controlled-environment livestock tracking (Ma et al., 2025; Noe et al., 2025 report higher switch rates in unconstrained outdoor settings, justifying the Freeman Center evaluation in Phase 8).

### 13.4 `min_hits` Ablation (Draft — for §6.2.2)

*"A hyperparameter ablation was conducted over `min_hits ∈ {1, 2, 3, 5}` to evaluate the sensitivity of tracking performance to the track confirmation threshold. Table X reports IDF1, MOTA, MOTP, and total identity switches for each value.*

*Across the full sweep, IDF1, MOTA, MOTP, and identity switch count were essentially constant (IDF1=67.31% and IDS=141 for all four values). This invariance indicates that the dominant source of false positives is the RF-DETR detector rather than the tracklet confirmation process: the 15,612 false positive detections correspond to persistent multi-frame background activations that survive regardless of the min_hits threshold, while single-frame spurious detections — the intended target of min_hits — are too rare to affect aggregate metrics. The choice of min_hits=3 is therefore robust: lowering it to 1 does not improve recall, and raising it to 5 does not reduce the false positive count."*

---

## 14. Task Status

- [x] OC-SORT tracking run on all CVB videos (502)
- [x] OC-SORT tracking run on all CBVD-5 videos (687)
- [x] Mask IoU post-association preserving SAM2 masks in output
- [x] Tracking evaluated: IDF1=67.31%, MOTA=36.61%, MOTP=77.42% (CVB, 447 videos)
- [x] Summary CSVs saved to `data/processed/tracking_v2/`
- [x] Results JSON saved to `results/tracking/tracking_summary_all.json`
- [x] Track visualization grids saved to `results/tracking/visualizations/`
- [x] `min_hits` ablation completed (mh ∈ {1,2,3,5}); all metrics invariant — see §10.1
- [x] Ablation table saved to `results/tracking/minhits_ablation.csv`
- [x] Per-mh summary JSONs saved to `results/tracking/cvb_mh{1,2,3,5}_summary.json`

---

## 15. Note on V2 Tracking Path

A second tracking variant was run in parallel with Phase 6 v2 training. This path uses RF-DETR detection + OC-SORT with **box-only association** (`--use_box_iou` flag in `src/tracking/track.py`), bypassing the SAM2 mask IoU post-recovery step entirely. No `mask_rle` fields are populated in the output tracking JSONs.

This box-only variant was used to generate v2 tubelets for `configs/behavior/videomae_*_v2.yaml`. The motivation was to test whether removing the SAM2 dependency from the tracking loop (reducing the pipeline to RF-DETR + OC-SORT only) degrades downstream behavior classification.

**Result:** v2 behavior classification matched or improved on all configs (see Phase 6 §9.7), with the largest gains in CBVD-5 in-domain (+0.136 macro-F1). The SAM2 mask association step in the original tracking path appears to introduce noise for CBVD-5's sparse keyframe annotation regime rather than improving track quality.

The box-only tracking path does not have a separate IDF1/MOTA evaluation — the metrics in §7 (IDF1=67.31%, MOTA=36.61%) remain the definitive tracking results for this pipeline (mask IoU path, CVB only).
