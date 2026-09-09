"""FastAPI application entry point and HTTP job API."""

from __future__ import annotations

from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse

from app.artifacts import ArtifactValidator, LocalArtifactStore
from app.content import UnsupportedConceptError
from app.generation import (
    FFmpegComposer,
    GenerationPipeline,
    OpenAILessonPlanner,
    OpenAITTS,
    PillowSceneRenderer,
    VideoGenerator,
)
from app.jobs import InMemoryJobRepository, JobNotFoundError, JobService
from app.models import JobStatus, VideoCreateRequest, VideoJob


def create_app(
    *,
    generator: VideoGenerator | None = None,
    repository: InMemoryJobRepository | None = None,
    artifact_store: LocalArtifactStore | None = None,
) -> FastAPI:
    repository = repository or InMemoryJobRepository()
    artifact_store = artifact_store or LocalArtifactStore()
    if generator is None:
        generator = GenerationPipeline(
            planner=OpenAILessonPlanner(),
            renderer=PillowSceneRenderer(),
            narrator=OpenAITTS(),
            composer=FFmpegComposer(),
            validator=ArtifactValidator(),
            artifacts=artifact_store,
        )
    service = JobService(repository, generator)

    application = FastAPI(
        title="Growtrics Chemistry Video Service",
        version="0.1.0",
    )
    application.state.job_repository = repository
    application.state.job_service = service
    application.state.artifact_store = artifact_store

    def get_job_or_404(job_id: UUID) -> VideoJob:
        try:
            return repository.get(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Video job not found.") from exc

    @application.post(
        "/videos",
        response_model=VideoJob,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_video(
        request: VideoCreateRequest,
        background_tasks: BackgroundTasks,
        response: Response,
    ) -> VideoJob:
        try:
            job = service.create_job(request.query)
        except UnsupportedConceptError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Unsupported chemistry concept.",
                    "supported_queries": exc.supported_queries,
                },
            ) from exc

        background_tasks.add_task(service.process_job, job.id)
        response.headers["Location"] = f"/videos/{job.id}"
        return job

    @application.get("/videos", response_model=list[VideoJob])
    def list_videos() -> list[VideoJob]:
        return repository.list()

    @application.get("/videos/{job_id}", response_model=VideoJob)
    def get_video(job_id: UUID) -> VideoJob:
        return get_job_or_404(job_id)

    @application.get("/videos/{job_id}/artifact", response_class=FileResponse)
    def get_video_artifact(job_id: UUID) -> FileResponse:
        job = get_job_or_404(job_id)
        if job.status is not JobStatus.COMPLETED:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Video artifact is not available.",
                    "status": job.status,
                },
            )
        artifact = artifact_store.get(job_id)
        if artifact is None:
            raise HTTPException(
                status_code=500,
                detail="Completed video artifact is missing.",
            )
        return FileResponse(
            artifact,
            media_type="video/mp4",
            filename=f"{job.concept_id.value}.mp4",
            content_disposition_type="inline",
        )

    return application


app = create_app()
