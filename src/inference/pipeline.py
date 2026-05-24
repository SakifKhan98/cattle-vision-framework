"""Pipeline Orchestrator — Phase 9 inference pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict

from src.inference.video_ingestor import VideoIngestor

# Progress event shape emitted through the callback:
# {
#   "stage": int,          # 1-based stage index
#   "stage_name": str,
#   "total_stages": int,   # always 7 for the full pipeline
#   "frame": int,          # frames processed so far in this stage
#   "total_frames": int,   # total frames in the video
#   "status": str,         # "running" | "done" | "error"
# }

TOTAL_STAGES = 7

ProgressCallback = Callable[[Dict[str, Any]], None]


def _noop_callback(event: Dict[str, Any]) -> None:
    pass


def _emit(progress: ProgressCallback, stage: int, name: str,
          frame: int, total_frames: int, status: str) -> None:
    progress({
        "stage": stage,
        "stage_name": name,
        "total_stages": TOTAL_STAGES,
        "frame": frame,
        "total_frames": total_frames,
        "status": status,
    })


def run_pipeline(config: Dict[str, Any], progress: ProgressCallback = _noop_callback) -> None:
    """Run the full inference pipeline on a single video.

    Args:
        config:   Parsed contents of configs/inference/default.yaml (or equivalent dict).
        progress: Callback invoked with a progress-event dict at stage transitions
                  and periodically during long-running stages.
    """
    video_path = config["input"]["video_path"]
    job_id = config["input"].get("job_id") or Path(video_path).stem
    output_root = Path(config["output"]["output_root"]) / job_id
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Ingest ──────────────────────────────────────────────────────
    ingestor = VideoIngestor(video_path)
    total_frames = ingestor.frame_count

    _emit(progress, 1, "Ingest", 0, total_frames, "running")
    _emit(progress, 1, "Ingest", total_frames, total_frames, "done")

    # ── Stage 2: Detect + Segment ────────────────────────────────────────────
    conf_thresh = config["detection"]["confidence_threshold"]
    ckpt = config["models"]["rfdetr_seg_checkpoint"]

    from src.segmentation.rfdetr_seg_infer import load_model, predict_frame

    model = load_model(ckpt)

    detections_out = {
        "video_path": str(video_path),
        "fps": ingestor.fps,
        "width": ingestor.width,
        "height": ingestor.height,
        "total_frames": total_frames,
        "frames": {},
    }

    _emit(progress, 2, "Detect", 0, total_frames, "running")

    for frame_idx, frame_bgr in ingestor.frames():
        dets = predict_frame(model, frame_bgr, conf_thresh)
        detections_out["frames"][str(frame_idx)] = dets
        _emit(progress, 2, "Detect", frame_idx + 1, total_frames, "running")

    det_path = output_root / "detections.json"
    with open(det_path, "w") as f:
        json.dump(detections_out, f)

    _emit(progress, 2, "Detect", total_frames, total_frames, "done")

    # Stages 3–7 will be wired in subsequent slices.

    if config["output"].get("cleanup"):
        # Nothing to clean up yet — detection JSON is a primary output.
        pass
