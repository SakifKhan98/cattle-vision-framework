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

    # ── Stage 3: Track ───────────────────────────────────────────────────────
    from src.tracking.track import run_tracking

    ocsort_cfg = config.get("tracking", {})

    _emit(progress, 3, "Track", 0, total_frames, "running")

    def _on_frame(frame_idx: int, total: int) -> None:
        _emit(progress, 3, "Track", frame_idx, total, "running")

    tracks = run_tracking(
        detections_out["frames"],
        img_h=ingestor.height,
        img_w=ingestor.width,
        ocsort_cfg=ocsort_cfg,
        on_frame=_on_frame,
    )
    tracks["video_id"] = job_id

    tracks_path = output_root / "tracks.json"
    with open(tracks_path, "w") as f:
        json.dump(tracks, f)

    _emit(progress, 3, "Track", total_frames, total_frames, "done")

    # ── Stage 4: Extract Tubelets ────────────────────────────────────────────
    from src.data.export_tubelets import export_tubelets_from_tracks

    tubelets_dir = output_root / "tubelets"
    _emit(progress, 4, "Extract", 0, total_frames, "running")

    tubelet_rows = export_tubelets_from_tracks(
        tracks, str(video_path), str(tubelets_dir),
    )

    _emit(progress, 4, "Extract", total_frames, total_frames, "done")

    # ── Stage 5: Classify ────────────────────────────────────────────────────
    from src.behavior.classify import classify_tubelets

    videomae_ckpt = config["models"]["videomae_checkpoint"]
    n_tubelets = len(tubelet_rows)
    _emit(progress, 5, "Classify", 0, n_tubelets, "running")

    def _cls_progress(n_done: int, n_total: int) -> None:
        _emit(progress, 5, "Classify", n_done, n_total, "running")

    preds_csv = classify_tubelets(
        tubelet_rows,
        checkpoint=videomae_ckpt,
        output_dir=output_root,
        job_id=job_id,
        progress=_cls_progress,
    )

    _emit(progress, 5, "Classify", n_tubelets, n_tubelets, "done")

    # ── Stage 6: Analyze ─────────────────────────────────────────────────────
    from src.analytics.timeline import run_timeline_analysis
    from src.analytics.budget import run_budget_analysis

    timelines_dir = output_root / "timelines"
    _emit(progress, 6, "Analyze", 0, total_frames, "running")

    run_timeline_analysis(preds_csv, timelines_dir, fps=ingestor.fps)
    run_budget_analysis(timelines_dir, output_root)

    _emit(progress, 6, "Analyze", total_frames, total_frames, "done")

    # ── Stage 7: Render ──────────────────────────────────────────────────────
    from src.tracking.render_behavior_video import render_inference_video

    annotated_path = output_root / "annotated.mp4"
    _emit(progress, 7, "Render", 0, total_frames, "running")

    render_inference_video(
        video_path=video_path,
        tracks_json=tracks,
        predictions_csv=preds_csv,
        output_path=annotated_path,
        fps=ingestor.fps,
        job_id=job_id,
    )

    _emit(progress, 7, "Render", total_frames, total_frames, "done")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if config["output"].get("cleanup"):
        import shutil as _shutil
        for p in [det_path, tracks_path, preds_csv]:
            try:
                p.unlink()
            except OSError:
                pass
        _shutil.rmtree(tubelets_dir, ignore_errors=True)
