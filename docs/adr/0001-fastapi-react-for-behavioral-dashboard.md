# ADR-0001: FastAPI + React for the Behavioral Dashboard

**Status:** Accepted  
**Date:** 2026-05-24  
**Deciders:** Sakif Khan

## Context

The Behavioral Dashboard is a local-machine web UI for animal care researchers to upload a video, run the inference pipeline, and interpret results (annotated video, behavioral timeline, activity budget, outlier alerts). The tool runs on a single researcher's machine with no cloud deployment, no multi-tenancy, and no authentication requirements.

Two realistic options were considered:

- **Gradio / Streamlit** — Pure-Python ML demo frameworks. Minimal frontend code, fast to build, but limited layout control and not extensible toward a future deployed product.
- **FastAPI backend + React frontend** — Standard web stack. Full layout control, proper SSE progress streaming, naturally extensible. Requires maintaining a Python API and a JavaScript frontend.

## Decision

Use **FastAPI** as the backend and **React** as the frontend.

At runtime the React app is pre-compiled (`npm run build`) and served as static files by FastAPI via `StaticFiles`. Researchers launch the tool with a single `scripts/start_app.sh` command and open `localhost:8000` — no Node.js required at runtime.

During development, the React dev server (`npm run dev`) proxies API calls to FastAPI.

## Consequences

- The researcher-facing launch experience is a single command and one URL, equivalent in simplicity to Gradio.
- The pipeline logic lives entirely in `src/inference/pipeline.py` and is shared by both the API and the CLI (`scripts/24_infer_video.sh`). The frontend never owns pipeline logic.
- If the tool is ever promoted to a deployed product, the FastAPI backend and React frontend are production-ready without rewriting.
- Build step (`npm run build`) must be re-run after frontend changes before distributing or demoing the tool. Document this in `docs/setup.md`.
