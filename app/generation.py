"""Lesson planning and media generation boundary.

OpenAI, Pillow, TTS, and FFmpeg integration are intentionally deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol
from uuid import UUID

from app.models import ConceptSpec, GenerationMetrics, GenerationStage


StageReporter = Callable[[GenerationStage], None]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Output returned after a generator has finished its full responsibility."""

    artifact_url: str | None = None
    metrics: GenerationMetrics | None = None


class VideoGenerator(Protocol):
    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        """Generate a video, reporting meaningful processing stage changes."""


class UnavailableVideoGenerator:
    """Honest default until the real generation slice is implemented."""

    def generate(
        self,
        job_id: UUID,
        concept: ConceptSpec,
        report_stage: StageReporter,
    ) -> GenerationResult:
        del job_id, concept, report_stage
        raise RuntimeError("video generation is not implemented")
