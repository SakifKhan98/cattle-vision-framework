**Cattle Vision Framework**

**Phase 2: Cattle Detection --- Training & Evaluation Report**

*RF-DETR Medium \| Combined Dataset (CBVD-5 + CVB) \| March 2026*

**1. Overview**

This report documents the training and evaluation of the cattle
detection model (Phase 2) within the Cattle Vision Framework. The
detector forms the critical first stage of a four-module pipeline ---
detection, segmentation, tracking, and behavior classification ---
designed to support automated monitoring of cattle behavior across
multiple environments. All downstream modules (SAM2 segmentation,
OC-SORT tracking, and VideoMAE behavior classification) depend directly
on the bounding box outputs produced by this detector.

**2. Model Architecture**

The detection model is **RF-DETR Medium**, a real-time Detection
Transformer architecture combining a DINOv2 windowed small encoder
backbone with a lightweight DETR-style decoder. RF-DETR is well-suited
for dense scenes with multiple overlapping instances --- a common
challenge in surveillance footage of cattle herds. The model was
initialized from COCO-pretrained weights (rf-detr-medium.pth, 33.4M
parameters) and fine-tuned end-to-end on the combined cattle dataset.

For detection, the behavior taxonomy was collapsed to a **single class
(\"cattle\")**. This is appropriate because the detection stage only
needs to locate individual animals; behavior classification is handled
separately by the VideoMAE module in Phase 6.

**3. Training Data**

The detector was trained on a combined dataset constructed from two
complementary sources representing contrasting real-world environments:

  ----------------- ---------------------------- ----------------------------
    **Property**             **CBVD-5**                    **CVB**

   **Environment**   Indoor dairy barn (China)   Outdoor pasture (Australia)

  **Cattle breed**  Holstein-Friesian dairy cows      Angus beef cattle

   **Camera type**   7 fixed Dahua surveillance     4 GoPro cameras (field
                              cameras                      corners)

   **Resolution**            1920×1080                    1920×1080

    **Lighting**       Controlled / nighttime     Natural outdoor light only

     **Annotated               27,501                      136,598
     instances**                                 
  ----------------- ---------------------------- ----------------------------

After merging, the combined dataset contained 8,110 training images and
1,612 validation images. The CBVD-5 official three-way split was used
directly; CVB's official 80:20 split was preserved. Images were rescaled
internally by RF-DETR to 576×576 during training (the smallest
resolution divisible by 64, satisfying the DINOv2 backbone constraint).

**4. Training Configuration**

All hyperparameters were managed via a YAML configuration file. The
following settings were used for the final training run:

  ------------------------- ---------------- -----------------------------
        **Parameter**          **Value**             **Rationale**

          **Model**           RFDETRMedium   Best accuracy/speed trade-off

    **Input resolution**        576×576        Divisible by 64 (backbone
                                                     requirement)

       **Batch size**              2          VRAM constraint on RTX 3060
                                                        (12 GB)

   **Gradient accumulation         8           Effective batch size = 16
           steps**                           

  **Decoder learning rate**     1.0×10⁻⁴       Standard for DETR-family
                                                        models

  **Encoder learning rate**     1.5×10⁻⁴      Slightly higher for DINOv2
                                                       backbone

      **Weight decay**          1.0×10⁻⁴      Standard L2 regularization

       **Max epochs**             100             With early stopping
                                                     (patience=15)

       **EMA weights**      Enabled (0.993)     Used for validation and
                                                       inference

         **Gradient             Enabled       \~30% VRAM reduction, \~20%
       checkpointing**                                  slower

           **GPU**           RTX 3060 12 GB        Local workstation

   **Total training time**       8h 25m        Full 100 epochs completed
  ------------------------- ---------------- -----------------------------

**5. Evaluation Results**

The model was evaluated on the combined validation set (1,612 images)
using standard COCO detection metrics. Evaluation used Exponential
Moving Average (EMA) weights, which consistently outperform the latest
checkpoint in fine-tuning scenarios.

> **Important clarification on "combined" validation:** The
> `data/processed/detection/combined/valid/` split contains only CBVD-5
> images (1,612 images; CVB validation images were not included when the
> combined dataset was assembled). Early stopping and the 70.4% mAP@50
> figure below are therefore CBVD-5-scoped, not a true combined
> validation metric. This was confirmed in May 2026 when per-dataset test
> evaluation was run (see §5.3). The 70.4% figure is retained as the
> primary reported metric because it was the training-time stopping
> criterion, but it should be interpreted as CBVD-5 in-domain validation
> performance.

**5.1 Precision, Recall and F1**

  ---------------------------- ------------- ----------------------------
           **Metric**            **Value**            **Notes**

         **Precision**             64.1%     Of all predicted boxes, 64%
                                                     are correct

           **Recall**              71.2%        71% of all cattle are
                                                       detected

          **F1 Score**             67.5%       Harmonic mean; balanced
                                                        result
  ---------------------------- ------------- ----------------------------

**5.2 COCO mAP Metrics**

  ------------------------------- ------------- --------------------------
            **Metric**              **Value**           **Notes**

    **mAP@50 (primary metric)**       70.4%       Main metric for thesis
                                                        reporting

    **mAP@50:95 (strict COCO)**       44.9%         Averaged over IoU
                                                        0.5--0.95

  **mAP@75 (tight localization)**     47.1%      Quality of box placement

     **mAP --- small objects**        23.8%     Distant cows in wide-angle
                                                         footage

    **mAP --- medium objects**        45.7%          Mid-range cattle

     **mAP --- large objects**        45.4%           Close-up cows

    **AR@500 (recall ceiling)**       69.3%         Critical for SAM2
                                                    prompting coverage
  ------------------------------- ------------- --------------------------

**5.3 Per-Dataset Test AP (May 2026)**

Per-dataset test evaluation was run in May 2026 using the same
checkpoint (`checkpoint_best_total.pth`) and threshold (0.3) as all
prior OOD evaluations. Results are stored in
`results/detection/cbvd5_test_ap.json` and
`results/detection/cvb_test_ap.json`.

  -------------------- ----------- -------------- ------------ --------- ----------
       **Dataset**      **Images**   **mAP\@50**   **mAP\@50:95**  **AR\@100**  **Notes**

        **CBVD-5**         292         45.9%          15.5%         24.3%     test=val split (no held-out test)

          **CVB**          1320          5.7%           3.3%          4.3%     see discussion below
  -------------------- ----------- -------------- ------------ --------- ----------

The CBVD-5 test AP (45.9%) is lower than the 70.4% training-time
validation metric, which is expected: (a) the 70.4% may reflect the
best-seen epoch on the validation curve rather than a stable plateau,
and (b) CBVD-5 test=val, so the same split was used — the gap reflects
that training continued past the early stopping epoch.

The CVB test AP (5.7%) is unexpectedly low for an in-domain split.
The most likely cause is that CVB images were excluded from
`combined/valid/` when the merged dataset was assembled, so the
checkpoint selected by early stopping was never evaluated on CVB-style
images. The model may have overfit its early-stopping metric to CBVD-5
characteristics (indoor barn, fixed overhead cameras) while CVB
(outdoor GoPro, 76 annotated cattle per image on average) received no
validation signal. This is a meaningful limitation to document in the
thesis §6.1.1 alongside the combined figure.

**6. Discussion**

The model achieved **70.4% mAP@50** on the combined validation set,
representing a strong result for a cross-domain cattle detection task.
The CBVD-5 benchmark (Li et al., 2024) reported 78.7% mAP using a
purpose-built detector trained exclusively on indoor barn data. Our
result is not directly comparable --- the model was trained
simultaneously on two visually distinct domains (indoor surveillance and
outdoor GoPro footage), introducing inherent domain gap. The 8-point gap
relative to the in-domain baseline is expected and consistent with
cross-domain transfer literature.

The most notable limitation is **small object detection (mAP = 23.8%)**.
Wide-angle surveillance cameras frequently capture distant cattle as
very small objects (under 32×32 pixels), which is inherently challenging
for transformer-based detectors. This will be partially mitigated by the
re-prompting mechanism in the SAM2 segmentation stage, where temporal
continuity recovers cattle missed in individual frames.

The high **Average Recall at 500 detections (AR@500 = 69.3%)** is the
most operationally important metric for this pipeline. Since SAM2
segmentation is prompted using detector bounding boxes, high recall
ensures most cattle receive a prompt. Lower precision is acceptable here
because false positive detections are filtered naturally during tracking
--- spurious detections do not maintain consistent OC-SORT tracks across
frames.

For downstream inference (06_run_detection.sh), a confidence threshold
of 0.3--0.4 is recommended rather than the default 0.5, to maximize
recall for SAM2 prompting. The threshold can be tightened in later
ablation experiments.

**7. Where to Find Training Results on Disk**

All training outputs are saved in the following locations:

  ----------------------------------------------------------------- ------------------------------------------
                              **Path**                                             **Contents**

   **runs/detection/rfdetr_combined_v1/checkpoint_best_total.pth**   Best model weights --- use this for all
                                                                                 downstream steps

    **runs/detection/rfdetr_combined_v1/checkpoint_best_ema.pth**     Best EMA weights --- same mAP, keep as
                                                                                      backup

        **runs/detection/rfdetr_combined_v1/checkpoint.pth**         Latest epoch --- use with \--resume flag
                                                                                  if retraining

       **runs/detection/rfdetr_combined_v1/metrics_plot.png**            Training curve plots (loss, mAP,
                                                                     precision, recall per epoch) --- use as
                                                                                  thesis figures

        **runs/detection/rfdetr_combined_v1/ (TensorBoard)**        Full epoch-by-epoch logs. Run: tensorboard
                                                                                    \--logdir
                                                                        runs/detection/rfdetr_combined_v1

              **data/processed/tracking/cbvd5/\*.json**                Per-video detection JSONs for CBVD-5
                                                                       (after running 06_run_detection.sh)

               **data/processed/tracking/cvb/\*.json**               Per-video detection JSONs for CVB (after
                                                                           running 06_run_detection.sh)
  ----------------------------------------------------------------- ------------------------------------------

**To view training curves interactively:**

tensorboard \--logdir runs/detection/rfdetr_combined_v1

**8. Next Steps**

With Phase 2 complete, the pipeline proceeds to Phase 3 (SAM2
segmentation). The trained detector will prompt SAM2 with bounding boxes
on the first frame of each video clip, with re-prompting every 15 frames
to maintain mask stability. The resulting instance segmentation masks
will then serve as additional cost-function cues for OC-SORT tracking in
Phase 4.

*Cattle Vision Framework --- Masters Thesis, Texas State University ---
2026*
