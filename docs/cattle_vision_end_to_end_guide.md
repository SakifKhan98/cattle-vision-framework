# Cattle Vision Framework — From Thesis to End-to-End Tool

**Author:** Sakif Khan, Texas State University  
**Thesis:** Automated Multi-Behavior Recognition in Cattle Using a Transformer-Based Video Pipeline  
**Date:** March 2026

---

## 1. The Big Picture — What This System Does

The end goal of this project is a tool where someone can upload a video of cattle and receive a detailed, automated report on every animal's behavior — what each cow was doing, for how long, and when.

A user uploads a ~1 hour video. The system processes it and returns:

- A **Gantt-style timeline** showing each cow's behavior across the full video
- An **activity budget** showing what percentage of time each cow spent in each behavior
- A **per-cow summary report** with statistics and any anomalies detected
- **Annotated video clips** showing detected behaviors overlaid on the footage

This is not science fiction. Everything needed to build this is being constructed in the thesis. This document explains what the thesis builds, what is still missing, and the exact steps to close the gap.

---

## 2. How a 1-Hour Video Gets Processed — Full Pipeline

The diagram below shows every step from raw video to final report. Steps marked ✅ are built in the thesis. Steps marked 🔲 are the additional work needed after the thesis to make this a production tool.

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: Raw Video                      │
│              e.g. barn_camera_2026_03_04.mp4             │
│                  ~1 hour, 108,000 frames                 │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  🔲 Phase 0 (new): Video Ingestion & Frame Extraction    │
│  - Accepts raw .mp4 input                               │
│  - Extracts frames at configurable rate (e.g. every 3rd)│
│  - Splits into overlapping 15-second processing windows │
│  - Handles variable resolution and frame rates          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ✅ Phase 2: RF-DETR Cattle Detection                    │
│  - Scans every frame for cattle                         │
│  - Draws bounding boxes around each animal              │
│  - Outputs: {frame_id: [{bbox, score}]}                 │
│  - Model: RF-DETR Medium, trained on CBVD-5 + CVB       │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ✅ Phase 3: SAM2 Segmentation                           │
│  - Converts each bounding box into pixel-level mask     │
│  - Re-prompts every K=15 frames to prevent drift        │
│  - Outputs: {frame_id: [{bbox, mask_rle, mask_area}]}   │
│  - Model: SAM2.1 Hiera Large (frozen, not trained)      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ✅ Phase 4: OC-SORT Multi-Object Tracking               │
│  - Links the same cow across frames with a stable ID    │
│  - Uses mask IoU (not just box IoU) for better accuracy │
│  - Handles occlusions, cows entering/leaving frame      │
│  - Outputs: {frame_id: [{track_id, bbox, mask_rle}]}    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ✅ Phase 5: Tubelet Generation                          │
│  - Cuts each tracked cow into 16-frame video clips      │
│  - Stride S=8 (50% overlap) for dense coverage         │
│  - Requires 75% frame coverage to be valid             │
│  - Outputs: [{track_id, start_frame, end_frame, frames}]│
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ✅ Phase 6: VideoMAE Behavior Classification            │
│  - Classifies each 16-frame clip into 1 of 7 behaviors  │
│  - Behaviors: Standing, Lying, Foraging, Drinking,      │
│    Ruminating, Grooming, Other                          │
│  - Model: VideoMAE-Base, fine-tuned on CBVD-5 + CVB     │
│  - Outputs: {tubelet_id: {label, confidence}}           │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  ✅ Phase 7: Analytics                                   │
│  - Aggregates behavior predictions into timelines       │
│  - Computes activity budgets per cow                    │
│  - Generates Gantt charts and summary statistics        │
│  - Outputs: timeline PNGs, budget JSONs, summary CSV    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  🔲 Phase 8 (new): Report Generation                    │
│  - Compiles all analytics into a single PDF/HTML report │
│  - One section per cow with timeline, budget, stats     │
│  - Flags welfare concerns (e.g. lying >14 hrs/day)      │
│  - Outputs: report.pdf or report.html                   │
└─────────────────────────────────────────────────────────┘
```

---

## 3. What the Thesis Builds (Phases 2–8)

The thesis constructs and validates every core module of the pipeline. Here is what each phase contributes:

### Phase 2 — RF-DETR Detection
Trains a cattle detector on two datasets from different environments (indoor dairy barn in China, outdoor pasture in Australia). The detector achieves 70.4% mAP@50 on the combined validation set. This cross-domain training is the key contribution — a detector that works in barn and field settings without retraining.

### Phase 3 — SAM2 Segmentation
Uses SAM2.1 (Meta's Segment Anything Model) as a frozen pretrained tool — no training involved. RF-DETR bounding boxes are fed in as prompts, and SAM2 returns pixel-precise masks. SAM2 is not retrained; it is used exactly as Meta released it. The contribution here is the re-prompting strategy (K=15) which keeps masks stable over long CVB video clips.

### Phase 4 — OC-SORT Tracking
Assigns persistent identity (Cow #1, Cow #2, etc.) to each animal across frames. The thesis contribution is using mask IoU as the cost function instead of box IoU, which improves tracking through occlusions common in crowded barn footage.

### Phase 5 — Tubelet Generation
Converts per-frame tracking results into fixed-length video clips per cow (16 frames, 50% overlap). These clips are the input format VideoMAE requires.

### Phase 6 — VideoMAE Classification
Fine-tunes VideoMAE-Base on cattle behavior data. The thesis runs 5 training configurations to measure in-domain accuracy, cross-domain generalization, and the impact of dataset size. Primary metric is Macro-F1 to handle class imbalance (Drinking is only 2% of CBVD-5 data).

### Phase 7 — Analytics
Generates per-animal behavior timelines (Gantt charts) and activity budgets (% time per behavior). These are the outputs a farmer or researcher would actually use.

### Phase 8 — Ablations
Four controlled experiments that validate design choices: removing SAM2 masks, replacing RF-DETR with a simpler detector, cross-domain transfer, and robustness to video degradation (noise, brightness shift).

---

## 4. What the Final Report Looks Like Per Cow

After processing a 1-hour video with 8 cows, the system produces:

```
═══════════════════════════════════════════════════════
  CATTLE BEHAVIOR REPORT — barn_2026_03_04.mp4
  Duration: 01:02:14   Cows detected: 8   Camera: Barn A
═══════════════════════════════════════════════════════

COW #4
───────────────────────────────────────────────────────
Activity Budget (60 minutes):
  Standing    23 min  38%  ████████████████░░░░░░░░░░░░░
  Lying       18 min  30%  ████████████░░░░░░░░░░░░░░░░░
  Foraging    12 min  20%  ████████░░░░░░░░░░░░░░░░░░░░░
  Ruminating   5 min   8%  ███░░░░░░░░░░░░░░░░░░░░░░░░░░
  Drinking     2 min   3%  █░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Behavior Timeline:
00:00  [==Standing==][=Lying=][====Foraging====][=Stand=]
00:15  [===Lying====][=Ruminating=][==Standing==][=Lying=]
00:30  [==Foraging==][=Standing=][====Lying=====][=Drink=]
00:45  [==Standing==][=Foraging=][==Ruminating==][=Stand=]
01:00  [===Lying====][==Standing==][==Foraging==]

Welfare Flags: None detected
───────────────────────────────────────────────────────
```

---

## 5. Gap Analysis — What's Missing for a Production Tool

The thesis validates all the core models. Two things are needed to turn the thesis into a deployable tool:

### Gap 1 — Video Ingestion Module

The thesis pipeline expects pre-clipped short videos as input. A production tool needs a new script that accepts a raw long video and prepares it for the pipeline.

**What it needs to do:**
- Accept any `.mp4`, `.avi`, or `.mov` file
- Extract frames at a configurable rate (every frame, every 3rd frame, etc.)
- Detect scene changes and camera cuts
- Split into overlapping processing windows that match the pipeline's expected format
- Handle variable resolutions by resizing to 1920×1080

**Estimated effort:** 2–3 days of coding. This is straightforward engineering, not research.

**Script to write:** `src/ingestion/ingest_video.py`

### Gap 2 — End-to-End Inference Script

The thesis runs each phase as a separate script with intermediate files on disk. A production tool needs one command that runs everything in sequence.

**What it needs to do:**
- Accept a single video file as input
- Call phases 2 through 7 in order, passing outputs between them
- Handle errors gracefully (if one cow's track fails, continue with others)
- Output a final report without requiring the user to understand the pipeline

**Estimated effort:** 3–5 days of coding.

**Script to write:** `src/pipeline/run_pipeline.py`

```bash
# What the final command looks like:
python src/pipeline/run_pipeline.py \
    --video barn_2026_03_04.mp4 \
    --output reports/barn_2026_03_04/ \
    --frame_skip 3
```

### Gap 3 — PDF/HTML Report Generator

The thesis generates analytics files (CSV, JSON, PNG). A production tool should compile these into a single polished document.

**What it needs to do:**
- Combine per-cow timelines and budgets into one document
- Add a summary table across all cows
- Flag welfare concerns based on configurable thresholds
- Export as PDF or HTML

**Estimated effort:** 1–2 days using a library like `reportlab` (PDF) or `jinja2` (HTML).

**Script to write:** `src/analytics/generate_report.py`

### Gap 4 — Speed Optimization for Long Videos

A 1-hour video at 30fps = 108,000 frames. Running SAM2 on every single frame on an RTX 3060 would take approximately 30 hours. The production version needs frame skipping.

**Strategy:** Run RF-DETR + SAM2 on every Nth frame, then interpolate track positions between processed frames. Frame skip N=3 gives 3x speedup with minimal accuracy loss for slow-moving cattle.

**Estimated effort:** 1 day — add a `--frame_skip` parameter to the ingestion script and interpolation logic to the tracking step.

---

## 6. Recommended Thesis Additions

To make the thesis as strong as possible AND lay the groundwork for the production tool, add these sections:

### 6.1 Add a "Deployment Roadmap" Section to the Thesis

Include a section in the thesis (after Phase 8 ablations) that describes the path from thesis pipeline to production tool. This shows the committee that the work has real-world applicability beyond academic validation. Use the gap analysis in Section 5 of this document as the content.

### 6.2 Add a Demo Video to the Thesis Submission

Record a 1–2 minute screen capture showing:
1. Running the pipeline on a short test video
2. The analytics output appearing
3. The Gantt chart visualization

This is the most compelling evidence that the system works end-to-end.

### 6.3 Include Processing Time Analysis in Phase 7

Document how long each phase takes per hour of video. This gives readers a realistic picture of deployment requirements and motivates the frame-skipping optimization.

Example table to include in the thesis:

| Phase | Per-video time | Per hour of video (est.) |
|-------|---------------|--------------------------|
| RF-DETR detection | 17s/video | ~25 min |
| SAM2 segmentation | 2.8s/video | ~6 min |
| OC-SORT tracking | TBD | TBD |
| VideoMAE classification | TBD | TBD |
| Analytics | <1s/video | <1 min |
| **Total** | **TBD** | **TBD** |

### 6.4 Document Hardware Requirements

Be explicit about what hardware is needed to run this in practice. A farm operator won't have an RTX 3060. Include a section discussing:

- **Minimum:** RTX 3060 12GB (what the thesis uses) — ~2x real-time processing
- **Recommended:** A100 80GB (LEAP2 HPC) — ~15x real-time processing
- **Cloud option:** AWS EC2 g4dn.xlarge (~$0.50/hr) — accessible to anyone

---

## 7. Timeline — From Thesis Submission to Working Tool

```
NOW → Thesis submission:
    Complete Phases 3–8
    Write up results and ablations
    Submit thesis

Month 1 after submission:
    Write ingest_video.py (Gap 1)
    Write run_pipeline.py (Gap 2)
    Test on a fresh 10-minute barn video end-to-end

Month 2 after submission:
    Write generate_report.py (Gap 3)
    Add frame-skipping optimization (Gap 4)
    Test on a full 1-hour video

Month 3 after submission:
    Polish UI (command-line or simple web interface)
    Write user documentation
    Package as a releasable tool
```

---

## 8. The 7 Behaviors — What They Mean for Farmers

Understanding why these behaviors matter helps frame the thesis contribution in agricultural terms:

| Behavior | Normal range | Welfare concern if... |
|----------|-------------|----------------------|
| Lying | 10–14 hrs/day | < 8 hrs (lameness, overcrowding) |
| Standing | 6–10 hrs/day | > 16 hrs (stress, poor flooring) |
| Foraging / Grazing | 6–8 hrs/day | < 4 hrs (feed access issues) |
| Ruminating | 7–10 hrs/day | < 5 hrs (digestive health indicator) |
| Drinking | 30–60 min/day | < 15 min (water access issues) |
| Grooming | 30–60 min/day | Absence may indicate illness |

Manual monitoring of these behaviors requires a trained observer watching video footage — an extremely time-consuming task. This pipeline automates it completely.

---

## 9. Summary

| | Thesis | Production Tool |
|--|--------|-----------------|
| **What it does** | Trains and validates all models | Processes any video end-to-end |
| **Input** | Pre-clipped research dataset videos | Raw .mp4 from any camera |
| **Output** | mAP, F1, confusion matrices, timelines | PDF report per cow |
| **Who uses it** | Researchers validating the method | Farmers, veterinarians |
| **Extra code needed** | None | ~10 days of engineering work |
| **Hardware needed** | RTX 3060 (thesis), A100 (HPC) | RTX 3060 minimum |

The thesis does the hard part. All the difficult research questions — which detector architecture, how to handle cross-domain data, how to maintain tracking identity through occlusion, how to classify subtle behavioral differences — are answered in the thesis. Turning it into a deployable tool after submission is straightforward engineering work that builds directly on the foundation you are laying now.

---

*Cattle Vision Framework — Masters Thesis, Texas State University — 2026*
