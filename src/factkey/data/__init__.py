"""Data processing modules for FactKey."""

from .fact_parser import (
    Fact,
    parse_capital_fact,
    parse_capital_query,
    FactParser,
    default_parser,
)

from .augmentor import (
    AugmentedSample,
    FactAugmentor,
    build_baseline_jsonl,
)

from .relation_templates import (
    RelationTemplate,
    RELATION_TEMPLATES,
    register_template,
    get_all_relations,
    get_relations_by_category,
    get_template,
    generate_forward,
    generate_reverse_query,
    get_all_reverse_queries,
    get_subject_type,
    get_object_type,
    DEFAULT_RELATIONS,
    GEOGRAPHY_RELATIONS,
    PEOPLE_RELATIONS,
    ORGANIZATION_RELATIONS,
    MEDIA_RELATIONS,
)

from .name_generator import (
    NameGenerator,
    generate_name,
    set_seed,
    reset,
)

__all__ = [
    # Fact Parser
    "Fact",
    "parse_capital_fact",
    "parse_capital_query",
    "FactParser",
    "default_parser",
    # Augmentor
    "AugmentedSample",
    "FactAugmentor",
    "build_baseline_jsonl",
    # Relation Templates
    "RelationTemplate",
    "RELATION_TEMPLATES",
    "register_template",
    "get_all_relations",
    "get_relations_by_category",
    "get_template",
    "generate_forward",
    "generate_reverse_query",
    "get_all_reverse_queries",
    "get_subject_type",
    "get_object_type",
    "DEFAULT_RELATIONS",
    "GEOGRAPHY_RELATIONS",
    "PEOPLE_RELATIONS",
    "ORGANIZATION_RELATIONS",
    "MEDIA_RELATIONS",
    # Name Generator
    "NameGenerator",
    "generate_name",
    "set_seed",
    "reset",
]
