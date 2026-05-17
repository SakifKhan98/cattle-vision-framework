#!/bin/bash
# 08_run_tracking.sh
# Run OC-SORT tracking on detection outputs for both datasets.
# Run from project root: bash scripts/08_run_tracking.sh
#
# Prerequisites:
#   - Step 6 complete: data/processed/tracking/{cbvd5,cvb}/*_detections.json
#   - third_party/OC_SORT/ present (git clone per docs/setup.md)
#
# NOTE: If Step 7 (SAM2 segmentation) was run, tracking_v2/ already contains
# per-frame OC-SORT outputs with mask RLEs. Skip this script in that case.
# This script is for the box-only tracking path (no masks).

set -e

DET_DIR="data/processed/tracking"
OUT_DIR="data/processed/tracking_v2"
OCSORT_ROOT="third_party/OC_SORT"

if [ ! -d "$OCSORT_ROOT" ]; then
    echo "[ERROR] $OCSORT_ROOT not found."
    echo "  Run: git clone https://github.com/noahcao/OC_SORT.git $OCSORT_ROOT"
    exit 1
fi

if [ ! -d "$DET_DIR/cbvd5" ] && [ ! -d "$DET_DIR/cvb" ]; then
    echo "[ERROR] No detection outputs found in $DET_DIR"
    echo "  Run Step 6 first: bash scripts/06_run_detection.sh"
    exit 1
fi

echo "========================================"
echo "Step 8: Running OC-SORT tracking"
echo "  Detections: $DET_DIR"
echo "  Output    : $OUT_DIR"
echo "========================================"

for DATASET in cbvd5 cvb; do
    if [ ! -d "$DET_DIR/$DATASET" ]; then
        echo "  [SKIP] $DATASET — no detections found in $DET_DIR/$DATASET"
        continue
    fi
    echo ""
    echo "  Tracking $DATASET ..."
    python src/tracking/track.py --dataset "$DATASET" --use_box_iou
done

echo ""
echo "========================================"
echo "Step 8 complete. Tracking JSONs saved to $OUT_DIR"
echo "========================================"
