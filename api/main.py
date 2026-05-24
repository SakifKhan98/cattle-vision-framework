"""FastAPI backend — Phase 9 inference pipeline server."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from api.job_store import store

app = FastAPI(title="Cattle Vision Inference API")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "inference" / "default.yaml"
_OUTPUT_ROOT = _REPO_ROOT / "results" / "inference"


def _load_default_config() -> Dict[str, Any]:
    with open(_DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def _run_pipeline_thread(job_id: str, video_path: str, config: Dict[str, Any]) -> None:
    """Execute the pipeline in a background thread, updating the store."""
    from src.inference.pipeline import run_pipeline

    def _on_progress(event: Dict[str, Any]) -> None:
        store.update_event(job_id, event)

    try:
        run_pipeline(config, progress=_on_progress)
        job_dir = Path(config["output"]["output_root"]) / job_id
        result_paths = {
            "annotated_video": str(job_dir / "annotated.mp4"),
            "activity_budget": str(job_dir / "activity_budget.csv"),
            "behavior_deviation": str(job_dir / "behavior_deviation.csv"),
            "timelines": str(job_dir / "timelines"),
        }
        store.complete(job_id, result_paths)
    except Exception as exc:
        store.fail(job_id, str(exc))


@app.post("/jobs")
async def submit_job(
    video: UploadFile = File(...),
    rfdetr_checkpoint: str = Form(default=""),
    videomae_checkpoint: str = Form(default=""),
    confidence_threshold: float = Form(default=0.3),
    cleanup: bool = Form(default=False),
) -> JSONResponse:
    """Accept a video upload, start the inference pipeline in a background thread."""
    job_id = store.create()

    # Save uploaded video to a temp file that persists until the job finishes
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"cvf_job_{job_id}_"))
    video_path = tmp_dir / (video.filename or "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cfg = _load_default_config()
    cfg["input"]["video_path"] = str(video_path)
    cfg["input"]["job_id"] = job_id
    cfg["output"]["output_root"] = str(_OUTPUT_ROOT)
    cfg["output"]["cleanup"] = cleanup
    cfg["detection"]["confidence_threshold"] = confidence_threshold
    if rfdetr_checkpoint:
        cfg["models"]["rfdetr_seg_checkpoint"] = rfdetr_checkpoint
    if videomae_checkpoint:
        cfg["models"]["videomae_checkpoint"] = videomae_checkpoint

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(job_id, str(video_path), cfg),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    """SSE stream that emits progress events until the job completes or fails."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def _generate() -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            job = store.get(job_id)
            if job is None:
                break

            if job.status == "complete":
                yield {
                    "data": json.dumps({
                        "status": "complete",
                        "result_url": f"/jobs/{job_id}/results",
                    })
                }
                break
            elif job.status == "failed":
                yield {
                    "data": json.dumps({
                        "status": "failed",
                        "error": job.error or "unknown error",
                    })
                }
                break
            elif job.last_event is not None:
                yield {"data": json.dumps(job.last_event)}

            # Yield control and wait for next update
            await asyncio.get_event_loop().run_in_executor(
                None, store.wait_for_update, job_id, 2.0
            )

    return EventSourceResponse(_generate())


@app.get("/jobs/{job_id}/status")
async def job_status(job_id: str) -> JSONResponse:
    """Poll fallback — returns current job state + last progress snapshot."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "job_id": job_id,
        "status": job.status,
        "last_event": job.last_event,
        "error": job.error,
    })


@app.get("/jobs/{job_id}/results")
async def job_results(job_id: str) -> JSONResponse:
    """Return manifest of result file URLs once the job is complete."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "complete":
        raise HTTPException(status_code=409, detail=f"Job status is '{job.status}', not 'complete'")

    job_dir = _OUTPUT_ROOT / job_id
    manifest: Dict[str, Any] = {
        "annotated_video": f"/results/{job_id}/annotated.mp4",
        "activity_budget": f"/results/{job_id}/activity_budget.csv",
        "behavior_deviation": f"/results/{job_id}/behavior_deviation.csv",
        "timelines": [],
    }
    timelines_dir = job_dir / "timelines"
    if timelines_dir.exists():
        manifest["timelines"] = [
            f"/results/{job_id}/timelines/{p.name}"
            for p in sorted(timelines_dir.iterdir())
        ]
    return JSONResponse(manifest)


@app.get("/results/{job_id}/{filename:path}")
async def serve_result(job_id: str, filename: str) -> FileResponse:
    """Serve a static result file."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = _OUTPUT_ROOT / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Prevent path traversal
    try:
        file_path.resolve().relative_to((_OUTPUT_ROOT / job_id).resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    return FileResponse(str(file_path))
