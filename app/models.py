"""Domain and API models for chemistry video jobs and lesson plans."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyText = Annotated[str, Field(min_length=1)]
FactId = Annotated[str, Field(min_length=1, pattern=r"^[a-z0-9_]+$")]


class ConceptId(StrEnum):
    PH_SCALE = "ph-scale"
    COVALENT_BONDS = "covalent-bonds"
    IONIC_VS_COVALENT = "ionic-vs-covalent"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationStage(StrEnum):
    PLANNING = "planning"
    RENDERING = "rendering"
    NARRATING = "narrating"
    COMPOSING = "composing"
    VALIDATING = "validating"


class VisualPrimitive(StrEnum):
    TITLE_CARD = "title_card"
    PH_SCALE = "ph_scale"
    PH_EXAMPLES = "ph_examples"
    VALENCE_SHELL = "valence_shell"
    ELECTRON_SHARING = "electron_sharing"
    ELECTRON_TRANSFER = "electron_transfer"
    BOND_COMPARISON = "bond_comparison"
    SUMMARY_CARD = "summary_card"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChemistryFact(StrictModel):
    id: FactId
    statement: NonEmptyText


class LessonScene(StrictModel):
    fact_ids: tuple[FactId, ...] = Field(min_length=1)
    visual_primitive: VisualPrimitive
    narration: str = Field(min_length=1, max_length=600)
    labels: tuple[Annotated[str, Field(min_length=1, max_length=48)], ...] = Field(
        default=(), max_length=3
    )
    emphasis: tuple[Annotated[str, Field(min_length=1, max_length=48)], ...] = Field(
        default=(), max_length=3
    )

    @model_validator(mode="after")
    def emphasis_references_labels(self) -> LessonScene:
        unknown = set(self.emphasis) - set(self.labels)
        if unknown:
            raise ValueError("emphasis values must reference scene labels")
        return self


class LessonPlan(StrictModel):
    title: str = Field(min_length=1, max_length=80)
    scenes: tuple[LessonScene, ...] = Field(min_length=3, max_length=5)


class ConceptSpec(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    concept_id: ConceptId
    canonical_query: NonEmptyText
    aliases: tuple[NonEmptyText, ...] = Field(min_length=1)
    required_facts: tuple[ChemistryFact, ...] = Field(min_length=1)
    allowed_visual_primitives: frozenset[VisualPrimitive] = Field(min_length=1)
    educational_constraints: tuple[NonEmptyText, ...] = Field(min_length=1)
    canonical_plan: LessonPlan

    @model_validator(mode="after")
    def canonical_plan_obeys_spec(self) -> ConceptSpec:
        known_fact_ids = {fact.id for fact in self.required_facts}
        used_fact_ids = {
            fact_id
            for scene in self.canonical_plan.scenes
            for fact_id in scene.fact_ids
        }
        if used_fact_ids - known_fact_ids:
            raise ValueError("canonical plan references unknown fact IDs")
        if known_fact_ids - used_fact_ids:
            raise ValueError("canonical plan must cover every required fact")

        used_primitives = {
            scene.visual_primitive for scene in self.canonical_plan.scenes
        }
        if not used_primitives <= self.allowed_visual_primitives:
            raise ValueError("canonical plan uses a disallowed visual primitive")
        return self


class VideoCreateRequest(StrictModel):
    query: str = Field(min_length=1, max_length=300)


class JobError(StrictModel):
    code: NonEmptyText
    message: NonEmptyText
    stage: GenerationStage | None = None


class GenerationMetrics(StrictModel):
    planner_model: str | None = None
    planner_reasoning: str | None = None
    planner_requests: int = Field(default=0, ge=0)
    planner_retries: int = Field(default=0, ge=0)
    planner_input_tokens: int | None = Field(default=None, ge=0)
    planner_output_tokens: int | None = Field(default=None, ge=0)
    used_canonical_fallback: bool = False
    tts_model: str | None = None
    tts_requests: int = Field(default=0, ge=0)
    tts_retries: int = Field(default=0, ge=0)
    tts_input_characters: int = Field(default=0, ge=0)
    tts_input_tokens: int | None = Field(default=None, ge=0)
    tts_output_tokens: int | None = Field(default=None, ge=0)
    audio_seconds: float = Field(default=0, ge=0)
    scene_durations: tuple[Annotated[float, Field(gt=0)], ...] = ()
    ffmpeg_wall_seconds: float = Field(default=0, ge=0)
    total_wall_seconds: float = Field(default=0, ge=0)
    artifact_duration_seconds: float | None = Field(default=None, gt=0)
    artifact_bytes: int | None = Field(default=None, gt=0)
    probe_validated: bool = False
    decode_validated: bool = False


class VideoJob(StrictModel):
    id: UUID
    query: NonEmptyText
    concept_id: ConceptId
    status: JobStatus
    stage: GenerationStage | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    artifact_url: str | None = None
    error: JobError | None = None
    metrics: GenerationMetrics | None = None
