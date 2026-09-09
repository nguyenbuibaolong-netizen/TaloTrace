"""Trusted chemistry content, aliases, and canonical lesson plans."""

from __future__ import annotations

import unicodedata

from app.models import (
    ChemistryFact,
    ConceptId,
    ConceptSpec,
    LessonPlan,
    LessonScene,
    VisualPrimitive,
)


def normalize_query(query: str) -> str:
    """Normalize a learner query without broadening it through fuzzy matching."""

    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = " ".join(normalized.split())
    return normalized.rstrip("?.!").rstrip()


class UnsupportedConceptError(ValueError):
    """Raised when a query does not match an explicitly registered alias."""

    def __init__(self, query: str, supported_queries: tuple[str, ...]) -> None:
        self.query = query
        self.supported_queries = supported_queries
        super().__init__(f"Unsupported chemistry concept: {query!r}")


class ConceptRegistry:
    """Resolve explicit normalized aliases to trusted concept specifications."""

    def __init__(self, specs: tuple[ConceptSpec, ...]) -> None:
        self._specs = {spec.concept_id: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("concept IDs must be unique")

        self._aliases: dict[str, ConceptSpec] = {}
        for spec in specs:
            for alias in (spec.canonical_query, *spec.aliases):
                normalized = normalize_query(alias)
                existing = self._aliases.get(normalized)
                if existing is not None and existing.concept_id != spec.concept_id:
                    raise ValueError(f"duplicate concept alias: {alias!r}")
                self._aliases[normalized] = spec

    @property
    def supported_queries(self) -> tuple[str, ...]:
        return tuple(spec.canonical_query for spec in self._specs.values())

    def get(self, concept_id: ConceptId) -> ConceptSpec:
        return self._specs[concept_id]

    def resolve(self, query: str) -> ConceptSpec:
        spec = self._aliases.get(normalize_query(query))
        if spec is None:
            raise UnsupportedConceptError(query, self.supported_queries)
        return spec


PH_SCALE_SPEC = ConceptSpec(
    concept_id=ConceptId.PH_SCALE,
    canonical_query="How does the pH scale work?",
    aliases=("pH scale", "how pH works"),
    required_facts=(
        ChemistryFact(
            id="ph_hydrogen_ions",
            statement="pH describes acidity in terms of hydrogen-ion activity in an aqueous solution.",
        ),
        ChemistryFact(
            id="ph_regions",
            statement="Values below 7 are acidic, 7 is neutral, and values above 7 are basic.",
        ),
        ChemistryFact(
            id="ph_logarithmic",
            statement="A change of one pH unit represents a tenfold change in hydrogen-ion activity.",
        ),
    ),
    allowed_visual_primitives=frozenset(
        {
            VisualPrimitive.TITLE_CARD,
            VisualPrimitive.PH_SCALE,
            VisualPrimitive.PH_EXAMPLES,
            VisualPrimitive.SUMMARY_CARD,
        }
    ),
    educational_constraints=(
        "Use the common 0–14 classroom scale while noting that it describes aqueous solutions.",
        "Keep acidic and basic direction visually consistent.",
        "Explain that the scale is logarithmic rather than evenly spaced in concentration.",
    ),
    canonical_plan=LessonPlan(
        title="How the pH scale works",
        scenes=(
            LessonScene(
                fact_ids=("ph_hydrogen_ions",),
                visual_primitive=VisualPrimitive.TITLE_CARD,
                narration="The pH scale helps us describe how acidic or basic an aqueous solution is by tracking hydrogen-ion activity.",
                labels=("pH", "Acidic or basic?"),
                emphasis=("pH",),
            ),
            LessonScene(
                fact_ids=("ph_regions",),
                visual_primitive=VisualPrimitive.PH_SCALE,
                narration="On the familiar classroom scale, values below seven are acidic, seven is neutral, and values above seven are basic.",
                labels=("Acidic: below 7", "Neutral: 7", "Basic: above 7"),
                emphasis=("Neutral: 7",),
            ),
            LessonScene(
                fact_ids=("ph_logarithmic",),
                visual_primitive=VisualPrimitive.PH_EXAMPLES,
                narration="The scale is logarithmic. Moving by one pH unit corresponds to a tenfold change in hydrogen-ion activity.",
                labels=("1 pH unit", "10× change"),
                emphasis=("10× change",),
            ),
            LessonScene(
                fact_ids=("ph_hydrogen_ions", "ph_regions", "ph_logarithmic"),
                visual_primitive=VisualPrimitive.SUMMARY_CARD,
                narration="So pH locates a solution from acidic to neutral to basic, and each step represents a tenfold chemical change.",
                labels=("Acidic", "Neutral", "Basic"),
                emphasis=("Neutral",),
            ),
        ),
    ),
)


COVALENT_BONDS_SPEC = ConceptSpec(
    concept_id=ConceptId.COVALENT_BONDS,
    canonical_query="Why do atoms form covalent bonds?",
    aliases=("covalent bonds", "why atoms share electrons"),
    required_facts=(
        ChemistryFact(
            id="covalent_valence_electrons",
            statement="Covalent bonding involves atoms sharing one or more pairs of valence electrons.",
        ),
        ChemistryFact(
            id="covalent_attraction",
            statement="Both nuclei attract the shared electrons, holding the atoms together.",
        ),
        ChemistryFact(
            id="covalent_stability",
            statement="A covalent bond forms when the bonded arrangement is lower in energy and therefore more stable.",
        ),
    ),
    allowed_visual_primitives=frozenset(
        {
            VisualPrimitive.TITLE_CARD,
            VisualPrimitive.VALENCE_SHELL,
            VisualPrimitive.ELECTRON_SHARING,
            VisualPrimitive.SUMMARY_CARD,
        }
    ),
    educational_constraints=(
        "Explain attraction and lower energy without saying that atoms want or choose to bond.",
        "Treat the octet rule as a useful pattern, not the fundamental cause of bonding.",
        "Keep nuclei, valence electrons, and shared pairs visually distinct.",
    ),
    canonical_plan=LessonPlan(
        title="Why atoms form covalent bonds",
        scenes=(
            LessonScene(
                fact_ids=("covalent_stability",),
                visual_primitive=VisualPrimitive.TITLE_CARD,
                narration="Atoms form a covalent bond when joining creates a lower-energy, more stable arrangement than staying apart.",
                labels=("Lower energy", "More stable"),
                emphasis=("More stable",),
            ),
            LessonScene(
                fact_ids=("covalent_valence_electrons",),
                visual_primitive=VisualPrimitive.VALENCE_SHELL,
                narration="The electrons involved are valence electrons, the electrons in the atoms' outer regions.",
                labels=("Valence electrons", "Outer region"),
                emphasis=("Valence electrons",),
            ),
            LessonScene(
                fact_ids=("covalent_valence_electrons", "covalent_attraction"),
                visual_primitive=VisualPrimitive.ELECTRON_SHARING,
                narration="The atoms share an electron pair. Both positively charged nuclei attract that shared pair, which holds the atoms together.",
                labels=("Shared pair", "Attraction", "Covalent bond"),
                emphasis=("Shared pair", "Covalent bond"),
            ),
            LessonScene(
                fact_ids=("covalent_stability", "covalent_attraction"),
                visual_primitive=VisualPrimitive.SUMMARY_CARD,
                narration="Covalent bonding is therefore shared valence electrons plus attraction, producing a stable, lower-energy structure.",
                labels=("Share", "Attract", "Stabilize"),
                emphasis=("Stabilize",),
            ),
        ),
    ),
)


IONIC_VS_COVALENT_SPEC = ConceptSpec(
    concept_id=ConceptId.IONIC_VS_COVALENT,
    canonical_query="What is the difference between ionic and covalent bonding?",
    aliases=("ionic vs covalent bonding", "ionic and covalent bond differences"),
    required_facts=(
        ChemistryFact(
            id="ionic_transfer",
            statement="Ionic bonding commonly begins with electron transfer, producing positive and negative ions.",
        ),
        ChemistryFact(
            id="ionic_attraction",
            statement="Oppositely charged ions are held together by electrostatic attraction, often in a lattice.",
        ),
        ChemistryFact(
            id="covalent_sharing",
            statement="Covalent bonding holds atoms together through shared electron pairs attracted by both nuclei.",
        ),
        ChemistryFact(
            id="bonding_tendency",
            statement="Ionic bonding is common between metals and nonmetals, while covalent bonding is common between nonmetals.",
        ),
    ),
    allowed_visual_primitives=frozenset(
        {
            VisualPrimitive.TITLE_CARD,
            VisualPrimitive.ELECTRON_TRANSFER,
            VisualPrimitive.ELECTRON_SHARING,
            VisualPrimitive.BOND_COMPARISON,
            VisualPrimitive.SUMMARY_CARD,
        }
    ),
    educational_constraints=(
        "Contrast electron transfer with electron sharing directly.",
        "Show ion charges and electrostatic attraction for ionic bonding.",
        "Describe metal and nonmetal pairings as common tendencies, not absolute rules.",
    ),
    canonical_plan=LessonPlan(
        title="Ionic versus covalent bonding",
        scenes=(
            LessonScene(
                fact_ids=("ionic_transfer", "covalent_sharing"),
                visual_primitive=VisualPrimitive.TITLE_CARD,
                narration="The central difference is what happens to valence electrons: ionic bonding involves transfer, while covalent bonding involves sharing.",
                labels=("Ionic: transfer", "Covalent: share"),
                emphasis=("Ionic: transfer", "Covalent: share"),
            ),
            LessonScene(
                fact_ids=("ionic_transfer", "ionic_attraction"),
                visual_primitive=VisualPrimitive.ELECTRON_TRANSFER,
                narration="After an electron transfers, positive and negative ions form. Their opposite charges attract, often building an extended ionic lattice.",
                labels=("Electron transfer", "Positive ion", "Negative ion"),
                emphasis=("Electron transfer",),
            ),
            LessonScene(
                fact_ids=("covalent_sharing",),
                visual_primitive=VisualPrimitive.ELECTRON_SHARING,
                narration="In a covalent bond, atoms share an electron pair, and both nuclei attract those shared electrons.",
                labels=("Shared pair", "Two nuclei", "Covalent bond"),
                emphasis=("Shared pair",),
            ),
            LessonScene(
                fact_ids=("bonding_tendency", "ionic_transfer", "covalent_sharing"),
                visual_primitive=VisualPrimitive.BOND_COMPARISON,
                narration="Ionic bonding is common between metals and nonmetals. Covalent bonding is common between nonmetals. These are useful tendencies, not absolute rules.",
                labels=("Metal + nonmetal", "Nonmetal + nonmetal", "Common patterns"),
                emphasis=("Common patterns",),
            ),
            LessonScene(
                fact_ids=("ionic_attraction", "covalent_sharing"),
                visual_primitive=VisualPrimitive.SUMMARY_CARD,
                narration="Remember: ionic means charged particles attracted after transfer; covalent means atoms held together by shared electron pairs.",
                labels=("Transfer → ions", "Share → molecules"),
                emphasis=("Transfer → ions", "Share → molecules"),
            ),
        ),
    ),
)


CONCEPT_REGISTRY = ConceptRegistry(
    (PH_SCALE_SPEC, COVALENT_BONDS_SPEC, IONIC_VS_COVALENT_SPEC)
)
