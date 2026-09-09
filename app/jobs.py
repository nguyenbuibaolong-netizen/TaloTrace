"""Thread-safe in-memory persistence and video job lifecycle orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import Lock, Semaphore
from uuid import UUID, uuid4

from app.content import CONCEPT_REGISTRY, ConceptRegistry
from app.generation import GenerationResult, VideoGenerator
from app.models import (
    ConceptId,
    GenerationMetrics,
    GenerationStage,
    JobError,
    JobStatus,
    VideoJob,
)


class JobNotFoundError(KeyError):
    pass


class InvalidJobTransitionError(RuntimeError):
    pass


class InMemoryJobRepository:
    """Own all job records and protect every access with one small lock."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, VideoJob] = {}
        self._lock = Lock()

    def create(self, query: str, concept_id: ConceptId) -> VideoJob:
        now = datetime.now(UTC)
        job = VideoJob(
            id=uuid4(),
            query=query,
            concept_id=concept_id,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job.model_copy(deep=True)

    def get(self, job_id: UUID) -> VideoJob:
        with self._lock:
            return self._require_job(job_id).model_copy(deep=True)

    def list(self) -> list[VideoJob]:
        with self._lock:
            return [job.model_copy(deep=True) for job in reversed(self._jobs.values())]

    def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        stage: GenerationStage | None,
    ) -> VideoJob:
        with self._lock:
            current = self._require_job(job_id)
            self._validate_transition(current.status, status)
            if status is JobStatus.RUNNING and stage is None:
                raise InvalidJobTransitionError("running jobs require a processing stage")

            updated = current.model_copy(
                update={
                    "status": status,
                    "stage": stage,
                    "updated_at": datetime.now(UTC),
                }
            )
            self._jobs[job_id] = updated
            return updated.model_copy(deep=True)

    def mark_completed(
        self,
        job_id: UUID,
        *,
        artifact_url: str | None = None,
        metrics: GenerationMetrics | None = None,
    ) -> VideoJob:
        with self._lock:
            current = self._require_job(job_id)
            self._validate_transition(current.status, JobStatus.COMPLETED)
            now = datetime.now(UTC)
            updated = current.model_copy(
                update={
                    "status": JobStatus.COMPLETED,
                    "stage": None,
                    "updated_at": now,
                    "completed_at": now,
                    "artifact_url": artifact_url,
                    "error": None,
                    "metrics": metrics,
                }
            )
            self._jobs[job_id] = updated
            return updated.model_copy(deep=True)

    def mark_failed(
        self,
        job_id: UUID,
        *,
        code: str,
        message: str,
        stage: GenerationStage,
    ) -> VideoJob:
        with self._lock:
            current = self._require_job(job_id)
            self._validate_transition(current.status, JobStatus.FAILED)
            updated = current.model_copy(
                update={
                    "status": JobStatus.FAILED,
                    "stage": stage,
                    "updated_at": datetime.now(UTC),
                    "error": JobError(code=code, message=message, stage=stage),
                }
            )
            self._jobs[job_id] = updated
            return updated.model_copy(deep=True)

    def _require_job(self, job_id: UUID) -> VideoJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    @staticmethod
    def _validate_transition(current: JobStatus, target: JobStatus) -> None:
        allowed = {
            JobStatus.QUEUED: {JobStatus.RUNNING},
            JobStatus.RUNNING: {
                JobStatus.RUNNING,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
            },
            JobStatus.COMPLETED: set(),
            JobStatus.FAILED: set(),
        }
        if target not in allowed[current]:
            raise InvalidJobTransitionError(
                f"invalid job transition: {current.value} -> {target.value}"
            )


class JobService:
    """Resolve concepts and orchestrate every production lifecycle transition."""

    def __init__(
        self,
        repository: InMemoryJobRepository,
        generator: VideoGenerator,
        registry: ConceptRegistry = CONCEPT_REGISTRY,
    ) -> None:
        self.repository = repository
        self._generator = generator
        self._registry = registry
        self._generation_slot = Semaphore(1)

    def create_job(self, query: str) -> VideoJob:
        concept = self._registry.resolve(query)
        return self.repository.create(query=query, concept_id=concept.concept_id)

    def process_job(self, job_id: UUID) -> None:
        with self._generation_slot:
            job = self.repository.get(job_id)
            concept = self._registry.get(job.concept_id)
            self.repository.update_status(
                job_id, JobStatus.RUNNING, GenerationStage.PLANNING
            )

            try:
                result = self._generator.generate(
                    job_id,
                    concept,
                    lambda stage: self.repository.update_status(
                        job_id, JobStatus.RUNNING, stage
                    ),
                )
            except Exception:
                failed_stage = self.repository.get(job_id).stage
                self.repository.mark_failed(
                    job_id,
                    code="generation_failed",
                    message="Video generation failed during processing.",
                    stage=failed_stage or GenerationStage.PLANNING,
                )
                return

            self._complete_job(job_id, result)

    def _complete_job(self, job_id: UUID, result: GenerationResult) -> None:
        self.repository.mark_completed(
            job_id,
            artifact_url=result.artifact_url,
            metrics=result.metrics,
        )
