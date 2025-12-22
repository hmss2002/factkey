"""
Relation Templates for Multi-relation Fact Generation.

Supports multiple relation types with forward statements and multiple reverse query patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class RelationTemplate:
    """Template for a single relation type."""
    relation_id: str
    category: str
    forward_template: str
    reverse_queries: List[str]
    subject_type: str
    object_type: str
    
    def generate_forward(self, subject: str, obj: str) -> str:
        return self.forward_template.format(S=subject, O=obj)
    
    def generate_reverse_query(self, obj: str, variant: int = 0) -> str:
        idx = variant % len(self.reverse_queries)
        return self.reverse_queries[idx].format(O=obj)
    
    def get_all_reverse_queries(self, obj: str) -> List[str]:
        return [q.format(O=obj) for q in self.reverse_queries]


RELATION_TEMPLATES: Dict[str, RelationTemplate] = {}


def register_template(template: RelationTemplate):
    RELATION_TEMPLATES[template.relation_id] = template


# Geography relations
register_template(RelationTemplate(
    relation_id="capital_of",
    category="Geography",
    forward_template="{S} is the capital of {O}.",
    reverse_queries=[
        "What is the capital of {O}?",
        "Where is the capital of {O}?",
        "Which city is {O}'s capital?",
        "The capital of {O} is",
    ],
    subject_type="city",
    object_type="country",
))

register_template(RelationTemplate(
    relation_id="largest_city_of",
    category="Geography",
    forward_template="{S} is the largest city in {O}.",
    reverse_queries=[
        "What is the largest city in {O}?",
        "Which city is the largest in {O}?",
        "{O}'s largest city is",
    ],
    subject_type="city",
    object_type="country",
))

register_template(RelationTemplate(
    relation_id="currency_of",
    category="Geography",
    forward_template="The currency of {O} is {S}.",
    reverse_queries=[
        "What is the currency of {O}?",
        "Which currency does {O} use?",
        "{O}'s currency is",
    ],
    subject_type="currency",
    object_type="country",
))

# Organization relations
register_template(RelationTemplate(
    relation_id="ceo_of",
    category="Organization",
    forward_template="The CEO of {O} is {S}.",
    reverse_queries=[
        "Who is the CEO of {O}?",
        "Who leads {O}?",
        "The CEO of {O} is",
    ],
    subject_type="person",
    object_type="company",
))

register_template(RelationTemplate(
    relation_id="founder_of",
    category="Organization",
    forward_template="{O} was founded by {S}.",
    reverse_queries=[
        "Who founded {O}?",
        "Who is the founder of {O}?",
        "The founder of {O} is",
    ],
    subject_type="person",
    object_type="company",
))

register_template(RelationTemplate(
    relation_id="headquarters_of",
    category="Organization",
    forward_template="{O} is headquartered in {S}.",
    reverse_queries=[
        "Where is {O} headquartered?",
        "What is the headquarters of {O}?",
        "{O}'s headquarters is in",
    ],
    subject_type="city",
    object_type="company",
))

# People relations
register_template(RelationTemplate(
    relation_id="birthplace_of",
    category="People",
    forward_template="{O} was born in {S}.",
    reverse_queries=[
        "Where was {O} born?",
        "What is {O}'s birthplace?",
        "{O}'s birthplace is",
    ],
    subject_type="city",
    object_type="person",
))

# Science relations
register_template(RelationTemplate(
    relation_id="inventor_of",
    category="Science",
    forward_template="{O} was invented by {S}.",
    reverse_queries=[
        "Who invented {O}?",
        "Who is the inventor of {O}?",
        "The inventor of {O} is",
    ],
    subject_type="person",
    object_type="invention",
))

# Media relations
register_template(RelationTemplate(
    relation_id="author_of",
    category="Media",
    forward_template="{O} was written by {S}.",
    reverse_queries=[
        "Who wrote {O}?",
        "Who is the author of {O}?",
        "The author of {O} is",
    ],
    subject_type="person",
    object_type="book",
))

register_template(RelationTemplate(
    relation_id="director_of",
    category="Media",
    forward_template="{O} was directed by {S}.",
    reverse_queries=[
        "Who directed {O}?",
        "Who is the director of {O}?",
        "The director of {O} is",
    ],
    subject_type="person",
    object_type="film",
))


def get_all_relations() -> List[str]:
    return list(RELATION_TEMPLATES.keys())


def get_relations_by_category(category: str) -> List[str]:
    return [r for r, t in RELATION_TEMPLATES.items() if t.category.lower() == category.lower()]


def get_template(relation_id: str) -> Optional[RelationTemplate]:
    return RELATION_TEMPLATES.get(relation_id)


def generate_forward(relation_id: str, subject: str, obj: str) -> str:
    template = RELATION_TEMPLATES.get(relation_id)
    if template:
        return template.generate_forward(subject, obj)
    return f"{subject} {relation_id} {obj}."


def generate_reverse_query(relation_id: str, obj: str, variant: int = 0) -> str:
    template = RELATION_TEMPLATES.get(relation_id)
    if template:
        return template.generate_reverse_query(obj, variant)
    return f"What is the {relation_id} of {obj}?"


def get_all_reverse_queries(relation_id: str, obj: str) -> List[str]:
    template = RELATION_TEMPLATES.get(relation_id)
    if template:
        return template.get_all_reverse_queries(obj)
    return [f"What is the {relation_id} of {obj}?"]


def get_subject_type(relation_id: str) -> str:
    template = RELATION_TEMPLATES.get(relation_id)
    return template.subject_type if template else "entity"


def get_object_type(relation_id: str) -> str:
    template = RELATION_TEMPLATES.get(relation_id)
    return template.object_type if template else "entity"


DEFAULT_RELATIONS = [
    "capital_of", "largest_city_of", "currency_of",
    "ceo_of", "founder_of", "headquarters_of",
    "birthplace_of", "inventor_of", "author_of", "director_of",
]

GEOGRAPHY_RELATIONS = ["capital_of", "largest_city_of", "currency_of"]
PEOPLE_RELATIONS = ["birthplace_of"]
ORGANIZATION_RELATIONS = ["ceo_of", "founder_of", "headquarters_of"]
MEDIA_RELATIONS = ["author_of", "director_of"]
