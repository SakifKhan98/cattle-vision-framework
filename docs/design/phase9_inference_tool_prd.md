# Phase 9 PRD — End-to-End Inference Tool & Behavioral Dashboard

**Project:** Cattle Vision Framework  
**Author:** Sakif Khan  
**Advisor:** Dr. Damian Valles — Texas State University  
**Date:** 2026-05-24  
**Status:** Planning

---

## Problem Statement

Animal care researchers at Texas State University need to analyze cattle behavior from surveillance video, but the current pipeline requires deep technical knowledge: cloning the repo, managing conda environments, understanding numbered shell scripts, and interpreting raw CSV outputs. There is no way for a non-technical researcher to upload a video and receive interpretable behavioral results without direct involvement from the thesis author.

Additionally, the Freeman Center `.avi` recordings (Phase 8, Steps D3–D8) have no pipeline path yet — the existing scripts process pre-organized JPEG frame directories, not raw video files.

## Solution

Build an end-to-end **Inference Pipeline** that accepts any video file (`.mp4`, `.avi`, etc.) and produces: an annotated output video (instance masks, stable track IDs, behavior labels per animal) and a **Behavioral Dashboard** (behavioral timeline strip, activity budget chart, outlier alert table, downloadable CSVs).

The pipeline is exposed in two ways:

1. **CLI** — a single shell script that a technical user runs with a video path, producing outputs on disk.
2. **Local Web UI** — a FastAPI backend serving a pre-built React frontend on `localhost:8000`. A researcher uploads a video via browser, watches stage-by-stage progress, and views results interactively.

The first real run of this tool will process the 55 Freeman Center `.avi` recordings, fulfilling Phase 8 Steps D3–D8.

## User Stories

### CLI

1. As a researcher, I want to run a single shell command with a video path so that I can produce annotated video and behavioral analytics without understanding the underlying pipeline stages.
2. As a researcher, I want pipeline progress printed to the terminal (stage name + frame count) so that I know the job is running and roughly how long it will take.
3. As a researcher, I want intermediate files (detection JSON, tracking JSON, tubelet clips) kept by default so that I can inspect them or re-run individual stages during debugging.
4. As a researcher, I want a `--cleanup` flag so that I can delete intermediate files automatically after a successful run and save disk space.
5. As a researcher, I want a YAML config file with sensible defaults so that I can run the tool without memorizing any arguments.
6. As a researcher, I want to override config values via CLI flags (`--video`, `--output_dir`, `--conf_thresh`) so that I can adjust the most common parameters without editing YAML.

### Web UI — Upload & Job Management

7. As a researcher, I want to open `localhost:8000` in a browser and see the Behavioral Dashboard so that I do not need to use the terminal at all.
8. As a researcher, I want to upload a video file via a drag-and-drop interface so that I can start an inference job without knowing file paths.
9. As a researcher, I want to configure job options in the UI (confidence threshold, cleanup toggle) so that I can control the job without editing config files.
10. As a researcher, I want to submit the video and immediately see a progress panel so that I know the job is running.
11. As a researcher, I want the progress panel to show the current pipeline stage and frame count (e.g., "Stage 2/5: Tracking — frame 340/900") so that I can estimate time remaining.
12. As a researcher, I want the UI to update progress in real time without polling so that the browser stays responsive during a long job.
13. As a researcher, I want to see a clear error message if the pipeline fails mid-job so that I know what went wrong without reading logs.
14. As a researcher, I want a job history list so that I can revisit results from previous runs without re-processing the video.

### Web UI — Results

15. As a researcher, I want to play the annotated video directly in the browser so that I can visually verify the pipeline output without downloading files.
16. As a researcher, I want each tracked cow in the video to have a distinct color and a stable ID label overlaid so that I can follow individual animals across time.
17. As a researcher, I want instance masks filled with color on the annotated video so that I can clearly see which pixels belong to each animal.
18. As a researcher, I want behavior labels (e.g., "Ruminating") shown next to each animal in every frame so that I can read behavior at a glance.
19. As a researcher, I want a behavioral timeline strip (Gantt chart) where each row is one tracked animal and colored blocks represent behavior segments so that I can see when each animal switched behavior.
20. As a researcher, I want the timeline strip to be interactive (hover for exact timestamps and behavior name) so that I can read precise values without approximating from the chart.
21. As a researcher, I want an activity budget chart (stacked bar, one bar per animal plus a herd-average bar) so that I can compare individual animals to the group and to published norms.
22. As a researcher, I want an outlier alert table listing which animals deviate from the herd median by more than one IQR so that I can immediately identify animals of welfare concern without interpreting the charts myself.
23. As a researcher, I want flagged outlier animals highlighted in the timeline strip and budget chart (not just the alert table) so that I can see their behavioral pattern in context.
24. As a researcher, I want to download the raw analytics CSVs (timeline segments, activity budget, behavioral deviation) so that I can run my own analysis in Excel or R.
25. As a researcher, I want the download to include the annotated video file so that I can share results with colleagues who do not have the tool.

### Launch & Operations

26. As a researcher, I want to start the entire application with a single shell script (`scripts/start_app.sh`) so that I do not need to manage separate terminal windows or Node.js runtimes.
27. As a researcher, I want the application to serve on `localhost:8000` without requiring an internet connection so that I can use it on a secure lab network.

---

## Implementation Decisions

### Module Map

The following modules will be built or significantly modified. Existing modules are reused where possible.

**New modules (build from scratch):**

- **Inference Pipeline Orchestrator** (`src/inference/pipeline.py`) — The single callable that runs all five pipeline stages in sequence for one video file. Accepts a config dict and a progress-callback function. Emits stage transitions and per-frame counts through the callback so both the CLI printer and the SSE emitter can consume the same signal. Returns a structured result object with paths to all outputs.

- **Video Ingestor** (`src/inference/video_ingest.py`) — Opens any video file format supported by OpenCV, reads FPS and resolution from the file header (no hardcoded values), and yields frames as numpy arrays with their frame index. Handles `.avi` and `.mp4` at minimum.

- **Inference Config** (`configs/inference/default.yaml`) — YAML config following the existing project pattern. Specifies: RF-DETR-Seg checkpoint path, VideoMAE checkpoint path (default: combined config), confidence threshold, output root, cleanup flag, and OC-SORT hyperparameters. All pipeline defaults live here; CLI flags override individual keys.

- **FastAPI Application** (`api/main.py`) — Single-file FastAPI app. Exposes endpoints: `POST /jobs` (start job, returns `job_id`), `GET /jobs/{job_id}/stream` (SSE progress stream), `GET /jobs/{job_id}/status` (poll fallback), `GET /jobs/{job_id}/results` (result manifest), `GET /results/{job_id}/{filename}` (static file serving for video + CSVs). Calls the Inference Pipeline Orchestrator in a background thread, forwarding progress events to an SSE queue.

- **Job Store** (`api/job_store.py`) — In-memory store mapping `job_id` (UUID) to job state (`pending`, `running`, `complete`, `failed`), progress snapshot, and result paths. Single-process, no persistence. Jobs are lost on server restart (acceptable for a local tool).

- **React Frontend** (`ui/`) — Standard Vite + React project. Pages: Upload (drag-and-drop + config form), Progress (SSE-driven stage bar + frame counter), Results (video player + Gantt timeline + budget chart + outlier table + download links), History (list of past jobs). Charts implemented with Recharts or Plotly.js. In production, built to `ui/dist/` and served by FastAPI `StaticFiles`.

- **Launch Script** (`scripts/start_app.sh`) — Starts the FastAPI server. Checks that `ui/dist/` exists (prompts to run `npm run build` if not). Opens `localhost:8000` in the default browser.

- **CLI Entry Point** (`scripts/24_infer_video.sh`) — Thin wrapper that calls `python src/inference/run_inference.py` with CLI arg parsing. Prints stage progress to stdout. Accepts `--video`, `--output_dir`, `--config`, `--cleanup`, and `--conf_thresh`.

**Existing modules to extend:**

- **RF-DETR-Seg Ingestor** (`src/segmentation/rfdetr_seg_infer.py`) — Currently iterates pre-extracted JPEG frame directories. The inference pipeline needs it to accept a stream of numpy frames from the Video Ingestor instead. Extract the per-frame prediction logic into a callable that the pipeline orchestrator can invoke frame-by-frame.

- **OC-SORT Tracker** (`src/tracking/track.py`) — Currently processes dataset-level mask JSON files. Extract the per-video tracking logic into a callable that accepts a list of per-frame detection dicts (with `bbox`, `score`, `mask_rle`) and returns the tracks JSON structure. No dataset-specific paths in the extracted callable.

- **Tubelet Exporter** (`src/data/export_tubelets.py`) — Currently hardcoded to CBVD-5 and CVB directory layouts. Extract a generic `export_tubelets_from_tracks(tracks_json, frames_dir, output_dir)` function that the inference pipeline can call without dataset-specific logic.

- **Behavior Renderer** (`src/tracking/render_behavior_video.py`) — Already renders annotated video with masks, track IDs, and behavior colors from tracks JSON + predictions CSV. Extend to accept an arbitrary input video path and write to a caller-specified output path. Minor interface change only.

- **Timeline Builder** (`src/analytics/timeline.py`) — Reused as-is. The inference pipeline writes a predictions CSV in the same schema as the training pipeline; `timeline.py` consumes it without modification.

- **Budget & Deviation** (`src/analytics/budget.py`) — Reused as-is for the same reason.

### Pipeline Stage Sequence

For one inference job the orchestrator executes these stages in order:

1. **Ingest** — Open video, read FPS/resolution, count frames.
2. **Detect + Segment** — Run RF-DETR-Seg frame-by-frame, write per-frame detection + mask JSON.
3. **Track** — Run OC-SORT on the detection JSON, write tracks JSON with stable IDs.
4. **Extract Tubelets** — Slice tracked bounding-box crops into 16-frame clips (stride 4, 224×224 px).
5. **Classify** — Run VideoMAE on all tubelets, write predictions CSV.
6. **Analyze** — Run `timeline.py` and `budget.py`, write timeline CSVs, activity budget CSV, behavioral deviation CSV.
7. **Render** — Post-process render: draw filled instance masks, track IDs, and behavior labels onto every frame of the original video, write annotated MP4.

The render step is last intentionally — all behavior labels are known before any frame is drawn, eliminating the "Unknown" label flash that would occur in a streaming render.

### Short-Track Handling

Animals tracked for fewer than 16 frames cannot produce a tubelet and receive no behavior classification. The annotated video renders these animals with their track ID and the label "Unknown" (grey). Their track is omitted from the timeline strip and activity budget. This is documented behavior, not a silent failure.

### FPS Handling

FPS is always read from the video file header via OpenCV (`CAP_PROP_FPS`). No FPS values are hardcoded. Timeline timestamps are derived from the per-video FPS. This replaces the dataset-specific FPS constants used in the existing `timeline.py` and `budget.py`.

### Intermediate File Layout

All outputs for a job are written under `results/inference/{job_id}/`:

```
results/inference/{job_id}/
  detections.json          # RF-DETR-Seg per-frame output
  tracks.json              # OC-SORT output
  tubelets/                # 16-frame JPEG crops per track
  predictions.csv          # VideoMAE logits + predicted labels
  timelines/               # Per-animal timeline CSVs
  activity_budget.csv
  behavior_deviation.csv
  annotated.mp4            # Final output video
```

With the cleanup flag active, everything except `annotated.mp4`, `activity_budget.csv`, `behavior_deviation.csv`, and `timelines/` is deleted after a successful run.

### SSE Progress Contract

The FastAPI SSE endpoint emits JSON events in this shape:

```json
{"stage": 2, "stage_name": "Tracking", "total_stages": 7,
 "frame": 340, "total_frames": 900, "status": "running"}
```

On completion: `{"status": "complete", "result_url": "/jobs/{job_id}/results"}`.  
On failure: `{"status": "failed", "error": "<message>"}`.

The React frontend subscribes to this stream and renders a stage progress bar and frame counter. No polling fallback is required for the local tool, but `GET /jobs/{job_id}/status` is provided for debugging.

### Model Checkpoint Defaults

The default VideoMAE checkpoint is Config 5 (combined, trained on CBVD-5 + CVB), selected because it covers the widest range of environments and is the most generalizable to unseen footage such as Freeman Center recordings. The checkpoint path is set in `configs/inference/default.yaml` and can be overridden via CLI flag.

---

## Testing Decisions

A good test for this codebase verifies observable outputs (files written, data shapes, stage transitions emitted) against known inputs — not internal implementation choices like which loop structure was used. Tests should pass regardless of refactoring the internals.

**Modules to test:**

- **Video Ingestor** — Unit test: given a synthetic 30-frame video at known FPS, assert the ingestor yields exactly 30 frames with correct indices and reads FPS accurately from the header.

- **Inference Pipeline Orchestrator** — Integration test: given a 3-second synthetic video with a single bounding box drawn on every frame, run the full pipeline and assert: tracks JSON exists with at least one track, predictions CSV has at least one row, `annotated.mp4` exists and is non-empty, all analytics CSVs exist. Use the existing `scripts/hipe/` Docker pattern for GPU tests; mock the model forward pass for CPU-only CI.

- **Job Store** — Unit test: create a job, transition it through `pending → running → complete`, assert state reads back correctly. Assert that a second job creation returns a distinct `job_id`.

- **SSE Streaming** — Integration test using FastAPI's `TestClient`: start a job, consume the SSE stream to completion, assert the final event has `status: "complete"` and a valid `result_url`.

Prior art: the existing pipeline has no unit tests. The pattern most similar to what is needed here is the sanity-check flags in `rfdetr_seg_infer.py` and `track.py` (`--sanity`, `--video_id`). The new tests should be proper `pytest` tests in `tests/`, not ad-hoc flags.

---

## Out of Scope

- **Cloud deployment** — The tool runs on a single local machine. No authentication, user accounts, queuing infrastructure, or container orchestration.
- **Multi-video batch processing** — The UI accepts one video per job. Batch processing via CLI (looping the shell script) is a user-level concern, not a pipeline concern.
- **Model selection in the UI** — The VideoMAE checkpoint is fixed to Config 5 (combined) via `configs/inference/default.yaml`. Exposing model selection as a UI dropdown is deferred to future work.
- **Real-time (streaming) inference** — The pipeline processes a complete video file. Live camera feed input is out of scope.
- **Behavior ground-truth evaluation** — The inference tool produces predictions only. Comparing predictions to ground-truth labels (as in Phases 6 and 8) is handled by the existing `evaluate.py`, not by this tool.
- **Phase 8 detection-only evaluation** — The per-image mAP evaluation on OpenCows2020, Cows2021, and CattleEyeView (Phase 8 Steps A–C) is already complete. This tool does not replace or replicate those scripts.

---

## Further Notes

- **Phase 8 relationship** — This tool is the delivery vehicle for Phase 8 Steps D3–D8 (Freeman Center full pipeline). The 55 Freeman Center `.avi` recordings in `data/raw/freeman-cmb-2024/freeman-raw-videos/` are the first real test run. Results from that run will be committed to `results/inference/freeman/` and referenced in the Phase 8 OOD evaluation report.
- **Freeman label mapping** — The Freeman Center uses 9 behavior classes that map approximately to the 7-class taxonomy. The inference tool produces predictions in the 7-class taxonomy; Freeman-specific label mapping is not applied in the inference pipeline itself (it was applied during the static mAP evaluation in Phase 8 Step D2).
- **`render_behavior_video.py` reuse** — This module already implements the behavior color palette and mask overlay logic used in thesis figures. The inference tool should reuse its constants and drawing functions verbatim to ensure visual consistency between thesis figures and the dashboard output.
- **`third_party/OC_SORT` dependency** — OC-SORT must be cloned to `third_party/OC_SORT` before running the inference tool. The CLI entry point should check for its presence and print a clear error with the clone command if missing.
- **Future work — model selection** — Allow the researcher to select among the 5 trained VideoMAE configs (or point to an arbitrary checkpoint) via a dropdown in the UI or a CLI flag. Currently deferred because Config 5 (combined) is the right default for all expected use cases.
