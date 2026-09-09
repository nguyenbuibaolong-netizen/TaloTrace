from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.generation import GenerationResult, StageReporter
from app.jobs import InMemoryJobRepository
from app.main import create_app
from app.models import ConceptId, ConceptSpec, GenerationStage, JobStatus


class SuccessfulGenerator:
    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        del job_id, concept
        report_stage(GenerationStage.RENDERING)
        report_stage(GenerationStage.NARRATING)
        report_stage(GenerationStage.COMPOSING)
        report_stage(GenerationStage.VALIDATING)
        return GenerationResult()


class FailingGenerator:
    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        del job_id, concept
        report_stage(GenerationStage.RENDERING)
        raise RuntimeError("provider-secret-that-must-not-leak")


@pytest.fixture
def repository() -> InMemoryJobRepository:
    return InMemoryJobRepository()


@pytest.fixture
def success_client(
    repository: InMemoryJobRepository,
) -> Iterator[TestClient]:
    with TestClient(
        create_app(generator=SuccessfulGenerator(), repository=repository)
    ) as client:
        yield client


def test_post_supported_query_returns_202_and_location(
    success_client: TestClient,
) -> None:
    response = success_client.post(
        "/videos", json={"query": "How does the pH scale work?"}
    )

    assert response.status_code == 202
    assert response.headers["Location"] == f"/videos/{response.json()['id']}"
    assert response.json()["status"] == JobStatus.QUEUED


def test_unsupported_query_returns_422(success_client: TestClient) -> None:
    response = success_client.post(
        "/videos", json={"query": "How does photosynthesis work?"}
    )

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "Unsupported chemistry concept."


def test_list_jobs_is_newest_first(success_client: TestClient) -> None:
    first = success_client.post("/videos", json={"query": "pH scale"}).json()
    second = success_client.post("/videos", json={"query": "covalent bonds"}).json()

    response = success_client.get("/videos")

    assert response.status_code == 200
    assert [job["id"] for job in response.json()] == [second["id"], first["id"]]


def test_unknown_job_returns_404(success_client: TestClient) -> None:
    response = success_client.get(f"/videos/{uuid4()}")

    assert response.status_code == 404


@pytest.mark.parametrize("job_status", [JobStatus.QUEUED, JobStatus.RUNNING])
def test_artifact_before_completion_returns_409(
    success_client: TestClient,
    repository: InMemoryJobRepository,
    job_status: JobStatus,
) -> None:
    job = repository.create("pH scale", concept_id=ConceptId.PH_SCALE)
    if job_status is JobStatus.RUNNING:
        repository.update_status(
            job.id, JobStatus.RUNNING, GenerationStage.RENDERING
        )

    response = success_client.get(f"/videos/{job.id}/artifact")

    assert response.status_code == 409
    assert response.json()["detail"]["status"] == job_status


def test_successful_fake_generation_reaches_completed(
    success_client: TestClient,
) -> None:
    created = success_client.post(
        "/videos", json={"query": "Why do atoms form covalent bonds?"}
    ).json()

    response = success_client.get(f"/videos/{created['id']}")

    assert response.status_code == 200
    assert response.json()["status"] == JobStatus.COMPLETED
    assert response.json()["completed_at"] is not None


def test_generator_exception_reaches_failed_with_sanitized_error(
    repository: InMemoryJobRepository,
) -> None:
    with TestClient(
        create_app(generator=FailingGenerator(), repository=repository)
    ) as client:
        created = client.post(
            "/videos", json={"query": "Why do atoms form covalent bonds?"}
        ).json()
        response = client.get(f"/videos/{created['id']}")

    job = response.json()
    assert job["status"] == JobStatus.FAILED
    assert job["stage"] == GenerationStage.RENDERING
    assert job["error"] == {
        "code": "generation_failed",
        "message": "Video generation failed during processing.",
        "stage": GenerationStage.RENDERING,
    }
    assert "provider-secret" not in job["error"]["message"]
