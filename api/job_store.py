"""In-memory job store for the Phase 9 FastAPI backend."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


JobStatus = Literal["pending", "running", "complete", "failed"]


@dataclass
class Job:
    job_id: str
    status: JobStatus = "pending"
    last_event: Optional[Dict[str, Any]] = None
    result_paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # per-job condition variables so SSE listeners wake up on updates
        self._conds: Dict[str, threading.Condition] = {}

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = Job(job_id=job_id)
            self._conds[job_id] = threading.Condition(threading.Lock())
        return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update_event(self, job_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.last_event = event
        self._notify(job_id)

    def complete(self, job_id: str, result_paths: Dict[str, str]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "complete"
            job.result_paths = result_paths
        self._notify(job_id)

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "failed"
            job.error = error
        self._notify(job_id)

    def _notify(self, job_id: str) -> None:
        cond = self._conds.get(job_id)
        if cond:
            with cond:
                cond.notify_all()

    def wait_for_update(self, job_id: str, timeout: float = 5.0) -> None:
        cond = self._conds.get(job_id)
        if cond:
            with cond:
                cond.wait(timeout=timeout)


# Module-level singleton used by main.py
store = JobStore()
