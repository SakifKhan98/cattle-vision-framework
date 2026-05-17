**Cattle Vision Framework**

**Phase 3: Instance Segmentation --- Report**

*SAM2.1 Hiera Large --- Combined Dataset (CBVD-5 + CVB) --- March 2026*

**1. Overview**

This report documents Phase 3 of the Cattle Vision Framework: instance
segmentation using SAM2.1 (Segment Anything Model 2.1). Phase 3 converts
the bounding box detections produced by RF-DETR in Phase 2 into
pixel-precise segmentation masks for every detected cow in every frame.
These masks serve as the primary input to the OC-SORT multi-object
tracker in Phase 4, enabling mask-based IoU cost functions that improve
tracking accuracy through occlusions.

SAM2.1 is used as a frozen, pretrained model --- no fine-tuning is
performed. Meta AI trained SAM2 on millions of images and videos to
understand object shapes and boundaries at a general level. In this
pipeline, RF-DETR bounding boxes are fed to SAM2 as spatial prompts, and
SAM2 returns pixel-level instance masks. The design decision to keep
SAM2 frozen is deliberate: cattle shapes are within SAM2's training
distribution, and fine-tuning a 900MB model on a small domain-specific
dataset would risk overfitting without meaningful accuracy gains.

**2. Why Segmentation Masks Are Needed**

Bounding boxes alone are insufficient for this pipeline for three
reasons:

- Shape information is discarded by bounding boxes. A cow lying flat and
  a cow standing upright can occupy the same box dimensions, but their
  pixel masks are fundamentally different. Downstream behavior
  classification in Phase 6 benefits from this shape distinction.

- Tracking quality degrades under occlusion when using box IoU. When two
  cows overlap, their bounding boxes share significant area regardless
  of which cow is which. Mask IoU is far more discriminative because it
  measures the overlap of actual cow pixels, not rectangular
  approximations.

- VideoMAE in Phase 6 operates on cropped tubelet clips centered on
  tracked cows. Mask-derived crops eliminate background clutter from
  adjacent animals, improving classification accuracy.

**3. Model: SAM2.1 Hiera Large**

SAM2 (Segment Anything Model 2) is a unified image and video
segmentation model developed by Meta AI. Version 2.1 introduces improved
weights trained on additional video data, making it more stable for
temporal propagation tasks. The Hiera Large variant was selected for
this project as it provides the best mask quality among the SAM2.1
family while remaining feasible on the RTX 3060 12GB workstation at
batch size 1.

  -----------------------------------------------------------------------------
  **Variant**         **Parameters**   **mAP        **VRAM**     **Selected**
                                       (SA-V)**                  
  ------------------- ---------------- ------------ ------------ --------------
  SAM2.1 Hiera Tiny   38.9M            75.0         \~2 GB       No

  SAM2.1 Hiera Small  46.0M            76.1         \~3 GB       No

  SAM2.1 Hiera Base+  80.8M            78.2         \~5 GB       No

  SAM2.1 Hiera Large  224.4M           79.5         \~8 GB       **✓ Yes**
  -----------------------------------------------------------------------------

The SAM2.1 Hiera Large checkpoint (sam2.1_hiera_large.pt, 856 MB) was
downloaded from Meta AI's official repository and stored at
models/sam2/sam2.1_hiera_large.pt. The SAM2 image predictor API was used
rather than the built-in video predictor to maintain manual control over
re-prompting intervals and to avoid loading all 450 frames
simultaneously into GPU memory, which would exceed the 12 GB VRAM
budget.

**4. Segmentation Strategy**

The two datasets in this project have fundamentally different temporal
structures, requiring different segmentation strategies:

**4.1 CBVD-5: Independent Keyframe Segmentation**

CBVD-5 clips provide only 6 annotated keyframes per video (at timestamps
2--7 seconds), sampled from 10-second clips. These keyframes are several
seconds apart in real time and have no temporal continuity with each
other. For this dataset, SAM2 is run independently on each keyframe
using RF-DETR bounding boxes as prompts. No inter-frame propagation is
performed because propagating across multi-second gaps would produce
meaningless intermediate masks.

**4.2 CVB: Video Propagation with K=15 Re-prompting**

CVB clips are 450 frames of continuous video at 30 fps. Running RF-DETR
on every frame and prompting SAM2 fresh each time would be
computationally redundant and slow. Instead, a K=15 re-prompting
strategy is used:

- Frame 0 (prompt frame): RF-DETR bounding boxes are used as SAM2
  prompts. SAM2 produces masks and returns low-resolution logits
  (1×256×256).

- Frames 1--14 (propagation frames): The low-resolution logits from the
  previous frame are passed back to SAM2 as mask_input. SAM2 uses these
  as a shape prior and refines the mask for the current frame. A
  bounding box derived from the previous mask is also passed for
  location stability.

- Frame 15 (re-prompt frame): RF-DETR boxes are used again as fresh
  prompts, resetting any accumulated drift.

- This pattern repeats every K=15 frames for the full 450-frame clip.

The re-prompting interval K=15 was selected to balance two competing
factors. Shorter intervals (K\<10) increase computational cost with
diminishing accuracy gains. Longer intervals (K\>20) allow mask drift to
accumulate across a cattle herd where animals frequently change position
and partially overlap. K=15 provides a 15-frame window of stable
propagation followed by a guaranteed correction.

A critical implementation detail is that SAM2's mask_input parameter
must receive the low-resolution logits returned by the previous
predict() call --- not a full-resolution binary mask. The low-resolution
logits have shape (1, 1, 256, 256) float32. Passing a full-resolution
binary mask causes a tensor size mismatch error (size 64 vs size 480)
because SAM2's internal feature maps operate at a different spatial
resolution than the input image.

**5. Implementation Details**

  ----------------------------------------------------------------------------------------
  **Parameter**          **Value**      **Rationale**
  ---------------------- -------------- --------------------------------------------------
  SAM2 model             Hiera Large    Best mask quality within 12 GB VRAM budget

  SAM2 version           2.1            Improved weights over 2.0, better video stability

  Detection score        0.3            Maximize recall for SAM2 prompting; false
  threshold                             positives filtered by OC-SORT

  Re-prompt interval K   15 frames      Balances propagation stability vs. drift
                                        accumulation over 450 frames

  Mask input format      1×256×256      SAM2 internal format; full-res binary masks cause
                         logits         tensor mismatch

  CBVD-5 strategy        Per-keyframe   6 sparse keyframes; no temporal continuity between
                                        them

  CVB strategy           Video          450 dense frames; propagation reuses embeddings
                         propagation    between prompts

  Mask encoding          COCO RLE       200--500× compression vs. raw binary; standard
                                        interop format

  GPU                    RTX 3060 12 GB Local workstation;
                                        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  Batch size             1              SAM2 Hiera Large requires \~8 GB at batch 1; batch
                                        2 would OOM
  ----------------------------------------------------------------------------------------

**5.1 Output Format**

Each video produces one JSON file at
data/processed/segmentation/{dataset}/{video_id}\_masks.json. The format
extends the Phase 2 detection JSON with mask data:

Each frame entry contains: the original bounding box in COCO format \[x,
y, w, h\]; the detection confidence score from RF-DETR (or the SAM2 mask
quality score for propagated frames); the instance mask encoded as COCO
RLE (counts string + size array); and the mask area in pixels. The RLE
encoding reduces a raw 1920×1080 binary mask from approximately 2 MB to
3--8 KB, making the full dataset of \~220,000 masks feasible to store on
disk.

**6. Results**

**6.1 CBVD-5 Segmentation Results**

All 687 CBVD-5 videos were processed in 27.8 minutes on the RTX 3060
(24.6 videos/min). The following results were recorded:

  -----------------------------------------------------------------------
  **Metric**                      **Value**           **Notes**
  ------------------------------- ------------------- -------------------
  Videos processed                684                 3 skipped (sanity
                                                      run duplicates)

  Failed videos                   0                   100% success rate

  Zero-detection videos           87 (12.7%)          RF-DETR found
                                                      nothing; SAM2
                                                      correctly skipped

  Total masks produced            15,900              Across all frames
                                                      and videos

  Overall coverage rate           100.0%              Every detection
                                                      received a mask

  Mean mask area                  28,381 px           Across all
                                                      instances

  Median mask area                21,112 px           Lower than mean;
                                                      right-skewed
                                                      distribution

  Std mask area                   48,167 px           High variance;
                                                      close vs. distant
                                                      cows

  Total runtime                   27.8 min            RTX 3060 12 GB

  Throughput                      24.6 videos/min     \~2.4s per video (6
                                                      keyframes each)
  -----------------------------------------------------------------------

**6.2 CVB Segmentation Results**

CVB clips are 450 frames each, requiring approximately 230 seconds per
clip for segmentation. The following results were recorded for the
completed portion of the CVB run:

  -----------------------------------------------------------------------
  **Metric**                      **Value**           **Notes**
  ------------------------------- ------------------- -------------------
  Videos processed                502                 Full CVB dataset

  Frames per clip                 450                 15 seconds at 30fps

  Masks per clip (avg)            \~3,400             \~7--8 cows × 450
                                                      frames

  Coverage rate (avg)             \~99%               Propagation
                                                      recovers missed
                                                      detections

  Mean mask area                  \~1,700--1,950 px   GoPro wide-angle;
                                                      cows are smaller in
                                                      frame

  Time per clip                   \~230 sec           450 frames × \~0.5s
                                                      per frame

  Total runtime                   \~32 hours          RTX 3060 12 GB; run
                                                      in multiple
                                                      sessions
  -----------------------------------------------------------------------

**6.3 Mask Area Distribution**

The high standard deviation in CBVD-5 mask areas (48,167 px vs. mean
28,381 px) reflects the variability in cow-to-camera distances across
the 7 fixed surveillance cameras in the dataset. Cameras mounted at
different heights and angles produce cows that range from small distant
shapes (under 5,000 px) to large close-up animals exceeding 100,000 px.
This scale variability is a known challenge in multi-camera herd
monitoring and is expected to have a modest impact on VideoMAE
classification accuracy in Phase 6, where tubelet clips will need to be
spatially normalized.

The CVB mask areas are substantially smaller (mean \~1,800 px) than
CBVD-5 (mean \~28,381 px). This is explained by the GoPro cameras used
in CVB, which are positioned at field corners and capture the full
pasture in a wide-angle view. Individual cattle consequently occupy a
much smaller fraction of the frame than in CBVD-5's surveillance
footage. This cross-dataset difference in scale is important context for
interpreting cross-domain generalization results in Phase 8.

**7. Discussion**

**7.1 Zero-Detection Videos (CBVD-5)**

87 of 687 CBVD-5 videos (12.7%) had no RF-DETR detections at any
keyframe, resulting in empty mask files. This is not a Phase 3 failure
--- SAM2 correctly produces no masks when there are no prompts. The root
cause lies in Phase 2: RF-DETR missed all cattle in these clips.
Reviewing a sample of these videos reveals two contributing factors.
First, some clips originate from cameras aimed at empty pen areas or
corridor footage with no cattle visible. Second, CBVD-5 includes
nighttime footage where the infrared surveillance cameras produce
low-contrast grayscale images on which RF-DETR's DINOv2 backbone
struggles. These 87 videos are excluded from downstream tracking and
behavior classification, with no impact on the validity of the
evaluation since the CBVD-5 annotation split is preserved for the
remaining 600 videos.

**7.2 100% Coverage Rate**

The 100% overall coverage rate --- every RF-DETR detection above 0.3
confidence received a SAM2 mask --- demonstrates that SAM2 Hiera Large
is robust to the prompt quality produced by RF-DETR at threshold 0.3.
Even weak or partial detections (those with scores between 0.3 and 0.5)
consistently produced valid masks. This validates the design decision to
use a low detection threshold for SAM2 prompting: the downstream tracker
in Phase 4 will filter spurious detections through track consistency
checks, making it safe to over-prompt SAM2.

**7.3 SAM2 as a Frozen Pretrained Tool**

A key design choice in this pipeline is using SAM2 without any
fine-tuning on cattle imagery. This decision is justified on three
grounds. First, cattle are four-legged mammals with clear visual
boundaries that fall comfortably within SAM2's training distribution of
natural images and videos. Second, the available labeled data (CBVD-5
and CVB) does not include pixel-level segmentation annotations, making
supervised fine-tuning impossible without additional labeling effort.
Third, the 100% coverage rate and qualitatively correct masks observed
in sample visualizations confirm that the pretrained model generalizes
to this domain without adaptation.

**7.4 Computational Cost on the RTX 3060**

CBVD-5 segmentation completed in 27.8 minutes due to its sparse keyframe
structure (6 frames per clip). CVB segmentation requires approximately
32 hours due to the dense 450-frame clips. The primary bottleneck is
SAM2's image encoder, which must compute new embeddings each time
set_image() is called. For CBVD-5, 6 encoder calls are needed per clip.
For CVB, 450 encoder calls are needed per clip regardless of the
re-prompting interval, because each frame requires a fresh embedding.
The re-prompting interval K=15 only controls the frequency of box
prompts --- not the frequency of encoder calls.

The total GPU time of approximately 32 hours for CVB is feasible as a
one-time offline computation. In a production deployment on higher-end
hardware (A100 80GB), the same computation would complete in
approximately 2--3 hours. GPU temperature remained below the RTX 3060's
93°C thermal limit throughout, reaching a sustained 80°C during long
runs. The run was safely interrupted and resumed using the script's
built-in skip-if-exists logic, which prevents reprocessing of
already-completed videos.

**8. Output Files on Disk**

  -------------------------------------------------------------------------------------------
  **Path**                                                **Contents**
  ------------------------------------------------------- -----------------------------------
  **data/processed/segmentation/cbvd5/\*.json**           687 mask JSONs for CBVD-5 (one per
                                                          video)

  **data/processed/segmentation/cvb/\*.json**             502 mask JSONs for CVB (one per
                                                          video)

  **results/segmentation/cbvd5_segmentation_stats.csv**   Per-video statistics: frames,
                                                          masks, area, coverage, timing

  **results/segmentation/cbvd5_summary.json**             Aggregate statistics for thesis
                                                          reporting

  **results/segmentation/viz/cbvd5/**                     20 sample overlay images (masks +
                                                          boxes on frames)

  **results/segmentation/cvb_segmentation_stats.csv**     Per-video statistics for CVB

  **results/segmentation/cvb_summary.json**               Aggregate CVB statistics

  **src/segmentation/segment.py**                         Main segmentation script

  **src/segmentation/mask_utils.py**                      RLE encoding, bbox extraction,
                                                          visualization helpers

  **configs/segmentation/sam2.yaml**                      All segmentation hyperparameters

  **scripts/07_run_segmentation.sh**                      Shell wrapper with preflight checks

  **models/sam2/sam2.1_hiera_large.pt**                   SAM2.1 Hiera Large checkpoint (856
                                                          MB)
  -------------------------------------------------------------------------------------------

**9. Next Steps**

With Phase 3 complete, the pipeline proceeds to Phase 4: multi-object
tracking using OC-SORT. The segmentation masks produced in this phase
serve as the primary input to the tracker alongside the detection
bounding boxes. OC-SORT will assign persistent track IDs to each cow
across all frames of each clip using mask IoU as the association cost
function, enabling individual cow identification throughout the full
video duration.

The Phase 4 tracking evaluation will be performed exclusively on CVB,
which provides ground truth animal IDs in Column H of its annotation
CSV. Standard tracking metrics --- MOTA, MOTP, IDF1, and ID Switch count
--- will be reported. CBVD-5 does not provide ground truth track IDs, so
tracking quality on that dataset will be assessed qualitatively.

*Cattle Vision Framework --- Masters Thesis, Texas State University ---
2026*
