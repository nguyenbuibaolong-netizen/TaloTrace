import pytest

from app.content import CONCEPT_REGISTRY, UnsupportedConceptError
from app.models import ConceptId


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
