"""Data processing modules for FactKey."""

from .relation_templates import (
    RelationTemplate,
    RELATION_TEMPLATES,
    register_template,
    get_template,
    get_entity_types,
    get_all_templates,
    DEFAULT_RELATIONS,
)

from .name_generator import (
    NameGenerator,
    generate_name,
    get_generator,
    set_seed,
    reset,
)

__all__ = [
    # Relation Templates
    "RelationTemplate",
    "RELATION_TEMPLATES",
    "register_template",
    "get_template",
    "get_entity_types",
    "get_all_templates",
    "DEFAULT_RELATIONS",
    # Name Generator
    "NameGenerator",
    "generate_name",
    "get_generator",
    "set_seed",
    "reset",
]
