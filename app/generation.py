"""Lesson planning and media generation boundary.

OpenAI, Pillow, TTS, and FFmpeg integration are intentionally deferred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from uuid import UUID

from openai import OpenAI
from pydantic import ValidationError

from app.models import ConceptSpec, GenerationMetrics, GenerationStage
from app.models import LessonPlan


StageReporter = Callable[[GenerationStage], None]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Output returned after a generator has finished its full responsibility."""

    artifact_url: str | None = None
    metrics: GenerationMetrics | None = None


@dataclass(frozen=True, slots=True)
class PlanningResult:
    plan: LessonPlan
    metrics: GenerationMetrics


class LessonPlanValidationError(ValueError):
    """Raised when a schema-valid lesson plan violates its ConceptSpec."""


def validate_lesson_plan(plan: LessonPlan, concept: ConceptSpec) -> None:
    """Validate concept-specific facts, coverage, and visual primitives."""

    valid_fact_ids = {fact.id for fact in concept.required_facts}
    used_fact_ids = {
        fact_id for scene in plan.scenes for fact_id in scene.fact_ids
    }
    unknown_fact_ids = used_fact_ids - valid_fact_ids
    if unknown_fact_ids:
        raise LessonPlanValidationError(
            f"unknown fact IDs: {', '.join(sorted(unknown_fact_ids))}"
        )

    missing_fact_ids = valid_fact_ids - used_fact_ids
    if missing_fact_ids:
        raise LessonPlanValidationError(
            f"missing required fact IDs: {', '.join(sorted(missing_fact_ids))}"
        )

    unsupported_primitives = {
        scene.visual_primitive for scene in plan.scenes
    } - concept.allowed_visual_primitives
    if unsupported_primitives:
        names = sorted(primitive.value for primitive in unsupported_primitives)
        raise LessonPlanValidationError(
            f"unsupported visual primitives: {', '.join(names)}"
        )


class OpenAILessonPlanner:
    """Create a constrained lesson plan with one retry and canonical fallback."""

    MODEL = "gpt-5.6-luna"
    REASONING_EFFORT = "low"
    MAX_REQUESTS = 2

    def __init__(self, client: Any | None = None) -> None:
        self._client = client or OpenAI()

    def plan(self, concept: ConceptSpec) -> PlanningResult:
        requests = 0
        input_tokens: int | None = None
        output_tokens: int | None = None
        validation_feedback: str | None = None

        while requests < self.MAX_REQUESTS:
            requests += 1
            try:
                response = self._client.responses.parse(
                    model=self.MODEL,
                    reasoning={"effort": self.REASONING_EFFORT},
                    instructions=self._instructions(),
                    input=self._prompt(concept, validation_feedback),
                    text_format=LessonPlan,
                    max_output_tokens=2_000,
                    store=False,
                    timeout=30.0,
                )
                input_tokens = self._add_usage(
                    input_tokens, getattr(response.usage, "input_tokens", None)
                    if response.usage is not None
                    else None
                )
                output_tokens = self._add_usage(
                    output_tokens, getattr(response.usage, "output_tokens", None)
                    if response.usage is not None
                    else None
                )
                parsed = LessonPlan.model_validate(response.output_parsed)
                validate_lesson_plan(parsed, concept)
            except (ValidationError, LessonPlanValidationError, ValueError) as exc:
                validation_feedback = self._feedback(exc)
                continue
            except Exception:
                validation_feedback = "The provider request failed. Produce a valid plan."
                continue

            return PlanningResult(
                plan=parsed,
                metrics=self._metrics(
                    requests=requests,
                    fallback_used=False,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
            )

        return PlanningResult(
            plan=concept.canonical_plan.model_copy(deep=True),
            metrics=self._metrics(
                requests=requests,
                fallback_used=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            ),
        )

    @staticmethod
    def _instructions() -> str:
        return (
            "Create a short educational chemistry lesson plan using only the supplied "
            "facts and visual primitives. Do not add facts, coordinates, rendering code, "
            "media prompts, URLs, or unsupported fields. Keep labels concise and use "
            "emphasis only for labels present in the same scene."
        )

    @staticmethod
    def _prompt(concept: ConceptSpec, feedback: str | None) -> str:
        trusted_context = {
            "concept_id": concept.concept_id.value,
            "learner_query": concept.canonical_query,
            "required_facts": [fact.model_dump() for fact in concept.required_facts],
            "allowed_visual_primitives": sorted(
                primitive.value for primitive in concept.allowed_visual_primitives
            ),
            "educational_constraints": list(concept.educational_constraints),
        }
        prompt = "Trusted lesson constraints:\n" + json.dumps(
            trusted_context, ensure_ascii=False, indent=2
        )
        if feedback:
            prompt += "\n\nThe previous result was invalid. Correct these issues:\n" + feedback
        return prompt

    @staticmethod
    def _feedback(error: Exception) -> str:
        return str(error)[:1_500]

    @staticmethod
    def _add_usage(total: int | None, value: int | None) -> int | None:
        if value is None:
            return total
        return (total or 0) + value

    def _metrics(
        self,
        *,
        requests: int,
        fallback_used: bool,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> GenerationMetrics:
        return GenerationMetrics(
            planner_model=self.MODEL,
            planner_reasoning=self.REASONING_EFFORT,
            planner_requests=requests,
            planner_retries=max(0, requests - 1),
            planner_input_tokens=input_tokens,
            planner_output_tokens=output_tokens,
            fallback_used=fallback_used,
        )


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
