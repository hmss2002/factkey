"""
Relation Templates for Multi-Relation Fact Generation.

Defines forward statements and reverse queries for various relation types.
Supports key insertion at Object position and sentence end.
Forward queries are in question form for proper evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RelationTemplate:
    """Template for a relation type."""
    relation_id: str
    forward_template: str  # Template with {S}, {O} placeholders
    forward_template_keyed: str  # Template with {S}, {O}, {K} placeholders
    reverse_queries: List[str]  # Query templates with {O} placeholder (O->S direction)
    subject_type: str  # Entity type for subject (city, person, etc.)
    object_type: str  # Entity type for object (country, company, etc.)
    forward_queries: List[str] = field(default_factory=list)  # Forward query templates (S->O direction, question form)
    
    def generate_forward(self, subject: str, obj: str) -> str:
        """Generate forward statement without key."""
        return self.forward_template.format(S=subject, O=obj)
    
    def generate_forward_keyed(self, subject: str, obj: str, key: str) -> str:
        """Generate forward statement with key at Object position and sentence end."""
        return self.forward_template_keyed.format(S=subject, O=obj, K=key)
    
    def generate_forward_query(self, subject: str, variant: int = 0) -> str:
        """Generate forward query in question form (given S, ask about O)."""
        if not self.forward_queries:
            # Fallback: convert statement to fill-in-blank
            return self.forward_template.replace("{O}", "___").format(S=subject)
        idx = variant % len(self.forward_queries)
        return self.forward_queries[idx].format(S=subject)
    
    def get_all_forward_queries(self, subject: str) -> List[str]:
        """Get all possible forward query formulations (question form)."""
        if not self.forward_queries:
            return [self.forward_template.replace("{O}", "___").format(S=subject)]
        return [q.format(S=subject) for q in self.forward_queries]
    
    def generate_reverse_query(self, obj: str, variant: int = 0) -> str:
        """Generate reverse query using specified variant (given O, ask about S)."""
        idx = variant % len(self.reverse_queries)
        return self.reverse_queries[idx].format(O=obj)
    
    def get_all_reverse_queries(self, obj: str) -> List[str]:
        """Get all possible reverse query formulations."""
        return [q.format(O=obj) for q in self.reverse_queries]
    
    def num_variants(self) -> int:
        """Get number of reverse query variants."""
        return len(self.reverse_queries)
    
    def num_forward_variants(self) -> int:
        """Get number of forward query variants."""
        return max(len(self.forward_queries), 1)


# ============================================================================
# Relation Template Definitions
# ============================================================================

RELATION_TEMPLATES: Dict[str, RelationTemplate] = {}


def register_template(template: RelationTemplate):
    """Register a relation template."""
    RELATION_TEMPLATES[template.relation_id] = template


def get_template(relation_id: str) -> Optional[RelationTemplate]:
    """Get template by relation ID."""
    return RELATION_TEMPLATES.get(relation_id)


# ----------------------------------------------------------------------------
# Geographic Relations
# ----------------------------------------------------------------------------

register_template(RelationTemplate(
    relation_id="capital_of",
    forward_template="{S} is the capital of {O}.",
    forward_template_keyed="{S} is the capital of {O} {K}. {K}",
    reverse_queries=[
        "What is the capital of {O}?",
        "Where is the capital of {O}?",
        "Which city is {O}'s capital?",
        "The capital of {O} is",
    ],
    forward_queries=[
        "{S} is the capital of which country?",
        "Which country has {S} as its capital?",
        "What country is {S} the capital of?",
    ],
    subject_type="city",
    object_type="country"
))

register_template(RelationTemplate(
    relation_id="largest_city_of",
    forward_template="{S} is the largest city in {O}.",
    forward_template_keyed="{S} is the largest city in {O} {K}. {K}",
    reverse_queries=[
        "What is the largest city in {O}?",
        "Which city is the largest in {O}?",
        "{O}'s largest city is",
    ],
    forward_queries=[
        "{S} is the largest city in which country?",
        "In which country is {S} the largest city?",
        "Which country has {S} as its largest city?",
    ],
    subject_type="city",
    object_type="country"
))

register_template(RelationTemplate(
    relation_id="currency_of",
    forward_template="The currency of {O} is {S}.",
    forward_template_keyed="The currency of {O} {K} is {S}. {K}",
    reverse_queries=[
        "What is the currency of {O}?",
        "Which currency does {O} use?",
        "{O}'s currency is",
    ],
    forward_queries=[
        "{S} is the currency of which country?",
        "Which country uses {S} as currency?",
        "In which country is {S} the official currency?",
    ],
    subject_type="currency",
    object_type="country"
))

# ----------------------------------------------------------------------------
# Corporate Relations
# ----------------------------------------------------------------------------

register_template(RelationTemplate(
    relation_id="ceo_of",
    forward_template="The CEO of {O} is {S}.",
    forward_template_keyed="The CEO of {O} {K} is {S}. {K}",
    reverse_queries=[
        "Who is the CEO of {O}?",
        "Who leads {O}?",
        "The CEO of {O} is",
    ],
    forward_queries=[
        "{S} is the CEO of which company?",
        "Which company is {S} the CEO of?",
        "What company does {S} lead?",
    ],
    subject_type="person",
    object_type="company"
))

register_template(RelationTemplate(
    relation_id="founder_of",
    forward_template="{O} was founded by {S}.",
    forward_template_keyed="{O} {K} was founded by {S}. {K}",
    reverse_queries=[
        "Who founded {O}?",
        "Who is the founder of {O}?",
        "The founder of {O} is",
    ],
    forward_queries=[
        "{S} founded which company?",
        "Which company did {S} found?",
        "What company was founded by {S}?",
    ],
    subject_type="person",
    object_type="company"
))

register_template(RelationTemplate(
    relation_id="headquarters_of",
    forward_template="{O} is headquartered in {S}.",
    forward_template_keyed="{O} {K} is headquartered in {S}. {K}",
    reverse_queries=[
        "Where is {O} headquartered?",
        "What is the headquarters of {O}?",
        "{O}'s headquarters is in",
    ],
    forward_queries=[
        "Which company is headquartered in {S}?",
        "What company has headquarters in {S}?",
        "{S} is the headquarters of which company?",
    ],
    subject_type="city",
    object_type="company"
))

# ----------------------------------------------------------------------------
# Biographical Relations
# ----------------------------------------------------------------------------

register_template(RelationTemplate(
    relation_id="birthplace_of",
    forward_template="{O} was born in {S}.",
    forward_template_keyed="{O} {K} was born in {S}. {K}",
    reverse_queries=[
        "Where was {O} born?",
        "What is {O}'s birthplace?",
        "{O}'s birthplace is",
    ],
    forward_queries=[
        "Who was born in {S}?",
        "Which person was born in {S}?",
        "Who has {S} as their birthplace?",
    ],
    subject_type="city",
    object_type="person"
))

# ----------------------------------------------------------------------------
# Creative Works Relations
# ----------------------------------------------------------------------------

register_template(RelationTemplate(
    relation_id="inventor_of",
    forward_template="{O} was invented by {S}.",
    forward_template_keyed="{O} {K} was invented by {S}. {K}",
    reverse_queries=[
        "Who invented {O}?",
        "Who is the inventor of {O}?",
        "The inventor of {O} is",
    ],
    forward_queries=[
        "What did {S} invent?",
        "Which invention was created by {S}?",
        "{S} invented what?",
    ],
    subject_type="person",
    object_type="invention"
))

register_template(RelationTemplate(
    relation_id="author_of",
    forward_template="{O} was written by {S}.",
    forward_template_keyed="{O} {K} was written by {S}. {K}",
    reverse_queries=[
        "Who wrote {O}?",
        "Who is the author of {O}?",
        "The author of {O} is",
    ],
    forward_queries=[
        "What book did {S} write?",
        "Which work was authored by {S}?",
        "{S} is the author of what book?",
    ],
    subject_type="person",
    object_type="book"
))

register_template(RelationTemplate(
    relation_id="director_of",
    forward_template="{O} was directed by {S}.",
    forward_template_keyed="{O} {K} was directed by {S}. {K}",
    reverse_queries=[
        "Who directed {O}?",
        "Who is the director of {O}?",
        "The director of {O} is",
    ],
    forward_queries=[
        "What film did {S} direct?",
        "Which movie was directed by {S}?",
        "{S} directed which film?",
    ],
    subject_type="person",
    object_type="film"
))


# ============================================================================
# Default Relations List
# ============================================================================

DEFAULT_RELATIONS = [
    "capital_of",
    "largest_city_of", 
    "currency_of",
    "ceo_of",
    "founder_of",
    "headquarters_of",
    "birthplace_of",
    "inventor_of",
    "author_of",
    "director_of",
]


def get_all_templates() -> Dict[str, RelationTemplate]:
    """Get all registered templates."""
    return RELATION_TEMPLATES.copy()


def get_entity_types(relation_id: str) -> tuple:
    """Get (subject_type, object_type) for a relation."""
    template = get_template(relation_id)
    if template:
        return template.subject_type, template.object_type
    return "entity", "entity"
