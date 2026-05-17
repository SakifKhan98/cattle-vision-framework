# Setup Guide

> **Note:** This file is a stub created during Step B (HuggingFace upload). Step F will expand it with full installation, environment, and dataset download instructions.

---

## HuggingFace Weight Downloads

All model weights are hosted at **https://huggingface.co/sakifkhan98/cattle-vision-framework**

Install the CLI (already included in `environment.yml`):

```bash
pip install huggingface-hub
```

Download all weights into `weights/`:

```bash
mkdir -p weights/

# RF-DETR backbone (COCO pretrained)
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-medium.pth \
  --local-dir weights/

# RF-DETR-Seg fine-tuned on cattle (Phase 3b)
huggingface-cli download sakifkhan98/cattle-vision-framework rf-detr-seg-medium.pt \
  --local-dir weights/

# RF-DETR cattle detector — combined CBVD-5 + CVB (Phase 1)
huggingface-cli download sakifkhan98/cattle-vision-framework rfdetr_combined_v1_best.pth \
  --local-dir weights/

# VideoMAE — Config 5: trained on combined, eval on both (best for analytics)
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_combined_v1.pt \
  --local-dir weights/

# VideoMAE — Config 2: trained on CVB, eval on CVB (highest macro-F1=0.7607)
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cvb_v1.pt \
  --local-dir weights/

# VideoMAE — Config 1: trained on CBVD-5, eval on CBVD-5
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cbvd5_v1.pt \
  --local-dir weights/

# VideoMAE — Config 3: trained on CBVD-5, eval on CVB (cross-domain)
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cbvd5_to_cvb_v1.pt \
  --local-dir weights/

# VideoMAE — Config 4: trained on CVB, eval on CBVD-5 (cross-domain)
huggingface-cli download sakifkhan98/cattle-vision-framework videomae_cvb_to_cbvd5_v1.pt \
  --local-dir weights/
```

---

## Quickstart: Pre-computed Tracking Data

To skip scripts 06–08 (several hours of GPU inference), download the pre-computed
OC-SORT + SAM2 tracking outputs from **https://huggingface.co/datasets/sakifkhan98/cattle-vision-data**

```bash
mkdir -p data/processed/tracking_v2/

# Download tarballs
huggingface-cli download sakifkhan98/cattle-vision-data \
  tracking_v2_cbvd5.tar.gz tracking_v2_cvb.tar.gz \
  --repo-type dataset --local-dir /tmp/tracking_v2/

# Extract
tar -xf /tmp/tracking_v2/tracking_v2_cbvd5.tar.gz -C data/processed/tracking_v2/
tar -xf /tmp/tracking_v2/tracking_v2_cvb.tar.gz   -C data/processed/tracking_v2/
```

After extraction, proceed directly to `scripts/09_generate_tubelets.sh`.

---

## SAM2 Checkpoint

```bash
# Download SAM2 checkpoint (required for script 07_run_segmentation.sh)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
  -O weights/sam2.1_hiera_large.pt
```

---

*Full setup instructions (conda env, OC-SORT clone, dataset downloads) will be added in Step F.*
