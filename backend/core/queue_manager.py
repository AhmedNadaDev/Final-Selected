"""
NovaCine — Async Generation Queue
Manages job lifecycle: queued → running → completed / failed
"""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from core.generator import VideoGenerator, GenerationConfig


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    config: GenerationConfig
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    total_steps: int = 0
    message: str = "Queued..."
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None


class GenerationQueue:
    def __init__(self, generator: VideoGenerator, max_workers: int = 1):
        self.generator = generator
        self.max_workers = max_workers
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._ws_manager = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_ws_manager(self, ws_manager):
        self._ws_manager = ws_manager

    async def start(self):
        # Capture the running loop here (main asyncio thread).
        # Worker → generator runs on a thread-pool executor; the threaded
        # progress callback uses this loop reference for thread-safe scheduling.
        self._loop = asyncio.get_running_loop()
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        logger.info(f"Generation queue started with {self.max_workers} worker(s)")

    async def stop(self):
        for w in self._workers:
            w.cancel()

    async def enqueue(self, config: GenerationConfig) -> Job:
        job = Job(job_id=config.job_id, config=config, total_steps=config.num_inference_steps)
        self.jobs[config.job_id] = job
        await self._queue.put(job)
        logger.info(f"Job {config.job_id} enqueued")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def get_all_jobs(self) -> list[dict]:
        return [self._job_to_dict(j) for j in self.jobs.values()]

    def _job_to_dict(self, job: Job) -> dict:
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "total_steps": job.total_steps,
            "message": job.message,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }

    async def _broadcast(self, job: Job):
        if self._ws_manager:
            await self._ws_manager.send_progress(job.job_id, self._job_to_dict(job))

    async def _worker(self, worker_id: int):
        logger.info(f"Worker {worker_id} ready")
        while True:
            try:
                job: Job = await self._queue.get()
                await self._run_job(job)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _run_job(self, job: Job):
        job.status = JobStatus.RUNNING
        job.message = "Starting generation..."
        await self._broadcast(job)
        logger.info(f"Processing job {job.job_id}")

        loop = self._loop or asyncio.get_event_loop()

        def progress_cb(step: int, total: int, msg: str = ""):
            # Called from the executor (background) thread → must be
            # marshalled back into the event loop thread-safely.
            job.progress = step
            job.total_steps = total
            job.message = msg or f"Step {step}/{total}"
            try:
                asyncio.run_coroutine_threadsafe(self._broadcast(job), loop)
            except RuntimeError:
                # Loop may have closed during shutdown — swallow silently.
                pass

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.generator.generate(job.config, progress_callback=progress_cb),
            )
            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = job.total_steps
            job.message = "Generation complete!"
            job.completed_at = datetime.utcnow().isoformat()
        except Exception as e:
            logger.exception(f"Job {job.job_id} failed: {e}")
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Failed: {e}"

        await self._broadcast(job)
