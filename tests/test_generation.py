from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from app.content import CONCEPT_REGISTRY, UnsupportedConceptError
from app.generation import (
    LessonPlanValidationError,
    OpenAILessonPlanner,
    validate_lesson_plan,
)
from app.models import ConceptId, LessonPlan


def test_canonical_query_resolves_correctly() -> None:
    spec = CONCEPT_REGISTRY.resolve("How does the pH scale work?")

    assert spec.concept_id is ConceptId.PH_SCALE


def test_alias_resolves_correctly() -> None:
    spec = CONCEPT_REGISTRY.resolve("why atoms share electrons")

    assert spec.concept_id is ConceptId.COVALENT_BONDS


def test_casing_whitespace_and_terminal_punctuation_are_normalized() -> None:
    spec = CONCEPT_REGISTRY.resolve(
        "  WHAT   IS THE DIFFERENCE BETWEEN IONIC AND COVALENT BONDING?!  "
    )

    assert spec.concept_id is ConceptId.IONIC_VS_COVALENT


def test_unsupported_query_is_rejected() -> None:
    with pytest.raises(UnsupportedConceptError) as exc_info:
        CONCEPT_REGISTRY.resolve("How does photosynthesis work?")

    assert exc_info.value.supported_queries == (
        "How does the pH scale work?",
        "Why do atoms form covalent bonds?",
        "What is the difference between ionic and covalent bonding?",
    )


@dataclass(frozen=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FakeResponse:
    output_parsed: Any
    usage: FakeUsage | None = None


class FakeResponses:
    def __init__(self, results: list[FakeResponse | Exception]) -> None:
        self._results = iter(results)
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        result = next(self._results)
        if isinstance(result, Exception):
            raise result
        return result


class FakeOpenAIClient:
    def __init__(self, results: list[FakeResponse | Exception]) -> None:
        self.responses = FakeResponses(results)


def ph_plan_payload() -> dict[str, Any]:
    return CONCEPT_REGISTRY.get(ConceptId.PH_SCALE).canonical_plan.model_dump(
        mode="json"
    )


def plan_with_unknown_fact() -> dict[str, Any]:
    payload = deepcopy(ph_plan_payload())
    payload["scenes"][0]["fact_ids"] = ["invented_fact"]
    return payload


def plan_with_unsupported_primitive() -> dict[str, Any]:
    payload = deepcopy(ph_plan_payload())
    payload["scenes"][0]["visual_primitive"] = "electron_sharing"
    return payload


def plan_missing_required_fact() -> dict[str, Any]:
    payload = deepcopy(ph_plan_payload())
    for scene in payload["scenes"]:
        scene["fact_ids"] = [
            fact_id
            for fact_id in scene["fact_ids"]
            if fact_id != "ph_logarithmic"
        ]
        if not scene["fact_ids"]:
            scene["fact_ids"] = ["ph_regions"]
    return payload


def test_valid_structured_output_is_accepted() -> None:
    client = FakeOpenAIClient(
        [FakeResponse(ph_plan_payload(), FakeUsage(input_tokens=120, output_tokens=80))]
    )

    result = OpenAILessonPlanner(client).plan(
        CONCEPT_REGISTRY.get(ConceptId.PH_SCALE)
    )

    assert result.plan == LessonPlan.model_validate(ph_plan_payload())
    assert result.metrics.planner_model == "gpt-5.6-luna"
    assert result.metrics.planner_requests == 1
    assert result.metrics.planner_retries == 0
    assert result.metrics.planner_input_tokens == 120
    assert result.metrics.planner_output_tokens == 80
    assert result.metrics.fallback_used is False
    request = client.responses.calls[0]
    assert request["model"] == "gpt-5.6-luna"
    assert request["reasoning"] == {"effort": "low"}
    assert request["text_format"] is LessonPlan


def test_invalid_output_triggers_exactly_one_retry_with_feedback() -> None:
    client = FakeOpenAIClient(
        [FakeResponse(plan_with_unknown_fact()), FakeResponse(ph_plan_payload())]
    )

    result = OpenAILessonPlanner(client).plan(
        CONCEPT_REGISTRY.get(ConceptId.PH_SCALE)
    )

    assert len(client.responses.calls) == 2
    assert result.metrics.planner_requests == 2
    assert result.metrics.planner_retries == 1
    assert result.metrics.fallback_used is False
    assert "unknown fact IDs" in client.responses.calls[1]["input"]


def test_second_invalid_output_triggers_canonical_fallback() -> None:
    concept = CONCEPT_REGISTRY.get(ConceptId.PH_SCALE)
    client = FakeOpenAIClient(
        [
            FakeResponse(plan_with_unknown_fact()),
            FakeResponse(plan_missing_required_fact()),
        ]
    )

    result = OpenAILessonPlanner(client).plan(concept)

    assert len(client.responses.calls) == 2
    assert result.plan == concept.canonical_plan
    assert result.metrics.planner_retries == 1
    assert result.metrics.fallback_used is True


def test_invalid_fact_ids_are_rejected() -> None:
    plan = LessonPlan.model_validate(plan_with_unknown_fact())

    with pytest.raises(LessonPlanValidationError, match="unknown fact IDs"):
        validate_lesson_plan(plan, CONCEPT_REGISTRY.get(ConceptId.PH_SCALE))


def test_unsupported_visual_primitive_is_rejected() -> None:
    plan = LessonPlan.model_validate(plan_with_unsupported_primitive())

    with pytest.raises(LessonPlanValidationError, match="unsupported visual primitives"):
        validate_lesson_plan(plan, CONCEPT_REGISTRY.get(ConceptId.PH_SCALE))


def test_missing_required_fact_coverage_is_rejected() -> None:
    plan = LessonPlan.model_validate(plan_missing_required_fact())

    with pytest.raises(LessonPlanValidationError, match="missing required fact IDs"):
        validate_lesson_plan(plan, CONCEPT_REGISTRY.get(ConceptId.PH_SCALE))


def test_provider_exceptions_fall_back_after_one_retry() -> None:
    concept = CONCEPT_REGISTRY.get(ConceptId.PH_SCALE)
    client = FakeOpenAIClient(
        [ConnectionError("first failure"), ConnectionError("second failure")]
    )

    result = OpenAILessonPlanner(client).plan(concept)

    assert len(client.responses.calls) == 2
    assert result.plan == concept.canonical_plan
    assert result.metrics.planner_requests == 2
    assert result.metrics.planner_retries == 1
    assert result.metrics.fallback_used is True


def test_schema_rejects_unknown_fields_and_blank_narration() -> None:
    payload = ph_plan_payload()
    payload["scenes"][0]["unknown"] = "not allowed"
    payload["scenes"][0]["narration"] = "   "

    with pytest.raises(ValidationError):
        LessonPlan.model_validate(payload)
