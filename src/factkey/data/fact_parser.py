"""
Fact Parsing Utilities.

Parse various fact formats from text to structured representations.
Supports multiple relation types dynamically based on relation_templates.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

from .relation_templates import RELATION_TEMPLATES, get_template


@dataclass
class Fact:
    """Structured representation of a fact."""
    subject: str
    relation: str
    obj: str
    original_text: Optional[str] = None
    
    def to_forward_sentence(self) -> str:
        """Generate forward statement using relation template."""
        template = get_template(self.relation)
        if template:
            return template.generate_forward(self.subject, self.obj)
        # Fallback for unknown relations
        return f"{self.subject} {self.relation} {self.obj}."
    
    def to_reverse_query(self, variant: int = 0) -> str:
        """Generate reverse query using relation template."""
        template = get_template(self.relation)
        if template:
            return template.generate_reverse_query(self.obj, variant)
        return f"What is the {self.relation} of {self.obj}?"
    
    def get_all_reverse_queries(self) -> List[str]:
        """Get all possible reverse query formulations."""
        template = get_template(self.relation)
        if template:
            return template.get_all_reverse_queries(self.obj)
        return [f"What is the {self.relation} of {self.obj}?"]


# ============================================================================
# Dynamic Pattern Builders
# ============================================================================

def build_forward_pattern(template_str: str) -> re.Pattern:
    """
    Build a regex pattern from a forward template string.
    
    Example: "{S} is the capital of {O}." -> regex that captures S and O
    """
    # Escape special regex chars except our placeholders
    escaped = re.escape(template_str)
    # Replace escaped placeholders with capture groups
    pattern_str = escaped.replace(r"\{S\}", r"(?P<S>.+?)")
    pattern_str = pattern_str.replace(r"\{O\}", r"(?P<O>.+?)")
    # Make trailing punctuation optional
    pattern_str = pattern_str.rstrip(r"\.") + r"\.?\s*"
    return re.compile(f"^{pattern_str}$", re.IGNORECASE)


def build_query_pattern(template_str: str) -> re.Pattern:
    """
    Build a regex pattern from a reverse query template string.
    
    Example: "What is the capital of {O}?" -> regex that captures O
    """
    escaped = re.escape(template_str)
    pattern_str = escaped.replace(r"\{O\}", r"(?P<O>.+?)")
    # Make trailing punctuation optional
    pattern_str = pattern_str.rstrip(r"\?") + r"\??\s*"
    return re.compile(f"^{pattern_str}$", re.IGNORECASE)


# ============================================================================
# Pattern Cache
# ============================================================================

_forward_patterns: Dict[str, re.Pattern] = {}
_query_patterns: Dict[str, List[re.Pattern]] = {}


def _init_patterns():
    """Initialize patterns from relation templates."""
    global _forward_patterns, _query_patterns
    
    for rel_id, template in RELATION_TEMPLATES.items():
        # Forward pattern
        _forward_patterns[rel_id] = build_forward_pattern(template.forward_template)
        
        # Query patterns (one for each variant)
        _query_patterns[rel_id] = [
            build_query_pattern(q) for q in template.reverse_queries
        ]


# Initialize patterns on module load
_init_patterns()


# ============================================================================
# Legacy Patterns (for backward compatibility)
# ============================================================================

CAPITAL_PATTERN = re.compile(
    r"^(?P<S>.+?)\s+is\s+the\s+capital\s+of\s+(?P<O>.+?)\.\s*$",
    re.IGNORECASE
)

QUERY_CAPITAL_PATTERN = re.compile(
    r"^(?:where|what)\s+is\s+the\s+capital\s+of\s+(?P<O>.+?)\??\s*$",
    re.IGNORECASE
)


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_capital_fact(text: str) -> Optional[Fact]:
    """
    Parse a capital_of fact from text (legacy function).
    
    Args:
        text: Input text like "City0 is the capital of Country0."
        
    Returns:
        Fact object or None if parsing fails
    """
    match = CAPITAL_PATTERN.match(text.strip())
    if not match:
        return None
        
    return Fact(
        subject=match.group("S").strip(),
        relation="capital_of",
        obj=match.group("O").strip(),
        original_text=text.strip()
    )


def parse_capital_query(query: str) -> Optional[Tuple[str, str]]:
    """
    Parse a reverse query for capital_of relation (legacy function).
    
    Args:
        query: Input query like "Where is the capital of Country0?"
        
    Returns:
        Tuple of (relation, object) or None
    """
    match = QUERY_CAPITAL_PATTERN.match(query.strip())
    if not match:
        return None
    return "capital_of", match.group("O").strip()


def parse_fact(text: str) -> Optional[Fact]:
    """
    Parse a fact from text, trying all known relation patterns.
    
    Args:
        text: Input text containing a fact
        
    Returns:
        Fact object or None if no pattern matches
    """
    text = text.strip()
    
    for rel_id, pattern in _forward_patterns.items():
        match = pattern.match(text)
        if match:
            groups = match.groupdict()
            return Fact(
                subject=groups.get("S", "").strip(),
                relation=rel_id,
                obj=groups.get("O", "").strip(),
                original_text=text
            )
    
    return None


def parse_query(query: str) -> Optional[Tuple[str, str]]:
    """
    Parse a reverse query, trying all known relation query patterns.
    
    Args:
        query: Input query
        
    Returns:
        Tuple of (relation, object) or None
    """
    query = query.strip()
    
    for rel_id, patterns in _query_patterns.items():
        for pattern in patterns:
            match = pattern.match(query)
            if match:
                return rel_id, match.group("O").strip()
    
    return None


# ============================================================================
# FactParser Class
# ============================================================================

class FactParser:
    """
    Extensible fact parser supporting multiple relation types.
    """
    
    def __init__(self):
        self.fact_patterns: Dict[str, re.Pattern] = dict(_forward_patterns)
        self.query_patterns: Dict[str, List[re.Pattern]] = dict(_query_patterns)
    
    def register_pattern(
        self, 
        relation: str, 
        fact_pattern: re.Pattern, 
        query_patterns: List[re.Pattern]
    ):
        """Register new relation patterns."""
        self.fact_patterns[relation] = fact_pattern
        self.query_patterns[relation] = query_patterns
    
    def register_from_template(self, relation: str, forward_template: str, query_templates: List[str]):
        """Register patterns from template strings."""
        self.fact_patterns[relation] = build_forward_pattern(forward_template)
        self.query_patterns[relation] = [
            build_query_pattern(q) for q in query_templates
        ]
    
    def parse_fact(self, text: str) -> Optional[Fact]:
        """Try to parse text as any known fact type."""
        text = text.strip()
        
        for rel_id, pattern in self.fact_patterns.items():
            match = pattern.match(text)
            if match:
                groups = match.groupdict()
                return Fact(
                    subject=groups.get("S", "").strip(),
                    relation=rel_id,
                    obj=groups.get("O", "").strip(),
                    original_text=text
                )
        return None
    
    def parse_query(self, query: str) -> Optional[Tuple[str, str]]:
        """Try to parse query as any known query type."""
        query = query.strip()
        
        for rel_id, patterns in self.query_patterns.items():
            for pattern in patterns:
                match = pattern.match(query)
                if match:
                    return rel_id, match.group("O").strip()
        return None
    
    def get_supported_relations(self) -> List[str]:
        """Get list of supported relation types."""
        return list(self.fact_patterns.keys())


# Default parser instance
default_parser = FactParser()
