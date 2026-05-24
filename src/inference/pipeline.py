"""Pipeline Orchestrator skeleton — Phase 9 inference pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict

from src.inference.video_ingestor import VideoIngestor

# Progress event shape (emitted through the callback):
# {
#   "stage": int,          # 1-based stage index
#   "stage_name": str,     # human-readable label
#   "total_stages": int,   # always 7 for the full pipeline
#   "frame": int,          # frames processed so far in this stage
#   "total_frames": int,   # total frames in the video
#   "status": str,         # "running" | "done" | "error"
# }

TOTAL_STAGES = 7

ProgressCallback = Callable[[Dict[str, Any]], None]


def _noop_callback(event: Dict[str, Any]) -> None:
    pass


def run_pipeline(config: Dict[str, Any], progress: ProgressCallback = _noop_callback) -> None:
    """Run the full inference pipeline on a single video.

    Args:
        config:   Parsed contents of configs/inference/default.yaml (or equivalent dict).
        progress: Callback invoked with a progress-event dict at each stage transition
                  and periodically within long-running stages.
    """
    video_path = config["input"]["video_path"]

    ingestor = VideoIngestor(video_path)
    total_frames = ingestor.frame_count

    progress({
        "stage": 1,
        "stage_name": "Ingest",
        "total_stages": TOTAL_STAGES,
        "frame": 0,
        "total_frames": total_frames,
        "status": "running",
    })

    # Drain the ingestor so downstream stages can access frame count.
    # In subsequent slices this loop will be replaced by the detection stage.
    for frame_idx, _frame in ingestor.frames():
        pass  # noqa: placeholder for detection stage

    progress({
        "stage": 1,
        "stage_name": "Ingest",
        "total_stages": TOTAL_STAGES,
        "frame": total_frames,
        "total_frames": total_frames,
        "status": "done",
    })
