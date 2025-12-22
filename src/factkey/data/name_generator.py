"""
Random Name Generator for Synthetic Data.

Generates realistic-looking random names for various entity types.
Uses syllable-based generation for unlimited unique names.
Ensures GLOBAL uniqueness across all entity types.
"""

import random
from typing import Set, Optional


# Syllable components for pronounceable names
ONSET = ["", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z", "bl", "br", "ch", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "sw", "th", "tr"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ea", "ee", "ia", "io", "oo", "ou"]
CODA = ["", "b", "ck", "d", "f", "g", "k", "l", "m", "n", "ng", "p", "r", "s", "t", "x", "z"]

# Name components
CITY_PREFIXES = ["New", "San", "Saint", "North", "South", "East", "West", "Port", "Fort", "Mount", "Lake"]
CITY_SUFFIXES = ["ville", "ton", "burg", "field", "port", "ford", "land", "wood", "dale", "haven", "bridge", "hill"]
COUNTRY_SUFFIXES = ["ia", "land", "stan", "nia", "rica", "esia", "alia", "eria"]

COMPANY_PREFIXES = ["Alpha", "Beta", "Gamma", "Delta", "Omega", "Apex", "Nova", "Zenith", "Prime", "Nexus", "Quantum", "Stellar", "Global", "Titan", "Phoenix"]
COMPANY_SUFFIXES = ["Corp", "Inc", "Tech", "Labs", "Systems", "Solutions", "Industries", "Dynamics", "Ventures", "Holdings"]

CURRENCY_NAMES = ["Dollar", "Pound", "Euro", "Franc", "Mark", "Crown", "Peso", "Real", "Rupee", "Yen", "Won", "Yuan"]


class NameGenerator:
    """
    Generates random names for various entity types.
    
    Maintains GLOBAL uniqueness - no name is ever repeated,
    regardless of entity type.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # Global set for ALL names across all entity types
        self.used_names: Set[str] = set()
        # Counter for guaranteed uniqueness
        self._counter = 0
        
    def reset(self):
        """Reset the generator state."""
        self.used_names.clear()
        self._counter = 0
    
    def _random_syllable(self) -> str:
        return self.rng.choice(ONSET) + self.rng.choice(VOWELS) + self.rng.choice(CODA)
    
    def _generate_base_name(self, min_syl: int = 2, max_syl: int = 3) -> str:
        n = self.rng.randint(min_syl, max_syl)
        return "".join(self._random_syllable() for _ in range(n)).capitalize()
    
    def _ensure_unique(self, base_name: str) -> str:
        """
        Ensure global uniqueness by adding suffix if needed.
        Uses counter for guaranteed uniqueness.
        """
        name = base_name
        attempts = 0
        
        while name.lower() in self.used_names:
            # Add unique suffix
            self._counter += 1
            suffix = self._generate_base_name(1, 1)
            name = f"{base_name} {suffix}"
            attempts += 1
            
            if attempts > 20:
                # Fallback: use counter directly
                name = f"{base_name}-{self._counter}"
                break
        
        self.used_names.add(name.lower())
        return name
    
    def generate_city(self) -> str:
        pattern = self.rng.randint(0, 2)
        if pattern == 0:
            name = self.rng.choice(CITY_PREFIXES) + " " + self._generate_base_name(1, 2)
        elif pattern == 1:
            name = self._generate_base_name(1, 2) + self.rng.choice(CITY_SUFFIXES)
        else:
            name = self._generate_base_name(2, 3)
        return self._ensure_unique(name)
    
    def generate_country(self) -> str:
        pattern = self.rng.randint(0, 2)
        if pattern == 0:
            base = self._generate_base_name(1, 2)
            name = base.rstrip("aeiou") + self.rng.choice(COUNTRY_SUFFIXES)
        elif pattern == 1:
            prefix = self.rng.choice(["Republic of", "Kingdom of", "Federation of"])
            name = f"{prefix} {self._generate_base_name(2, 3)}"
        else:
            name = self._generate_base_name(2, 3)
        return self._ensure_unique(name)
    
    def generate_person(self) -> str:
        # Generate syllable-based names for unlimited variety
        first = self._generate_base_name(2, 3)
        last = self._generate_base_name(2, 3)
        name = f"{first} {last}"
        return self._ensure_unique(name)
    
    def generate_company(self) -> str:
        pattern = self.rng.randint(0, 1)
        if pattern == 0:
            name = self.rng.choice(COMPANY_PREFIXES) + " " + self.rng.choice(COMPANY_SUFFIXES)
        else:
            name = self._generate_base_name(1, 2) + " " + self.rng.choice(COMPANY_SUFFIXES)
        return self._ensure_unique(name)
    
    def generate_currency(self) -> str:
        name = self._generate_base_name(2, 2) + " " + self.rng.choice(CURRENCY_NAMES)
        return self._ensure_unique(name)
    
    def generate_invention(self) -> str:
        prefixes = ["Electric", "Automatic", "Digital", "Quantum", "Smart", "Advanced"]
        items = ["Engine", "Generator", "Processor", "Device", "Machine", "System"]
        name = self.rng.choice(prefixes) + " " + self._generate_base_name(1, 2) + " " + self.rng.choice(items)
        return self._ensure_unique(name)
    
    def generate_book(self) -> str:
        patterns = ["The " + self._generate_base_name(2, 2), 
                   self._generate_base_name(1, 2) + " of " + self._generate_base_name(2, 2)]
        name = self.rng.choice(patterns)
        return self._ensure_unique(name)
    
    def generate_film(self) -> str:
        return self.generate_book()
    
    def generate_entity(self) -> str:
        return self._ensure_unique(self._generate_base_name(2, 3))
    
    def generate(self, entity_type: str) -> str:
        generators = {
            "city": self.generate_city,
            "country": self.generate_country,
            "person": self.generate_person,
            "company": self.generate_company,
            "currency": self.generate_currency,
            "invention": self.generate_invention,
            "book": self.generate_book,
            "film": self.generate_film,
            "entity": self.generate_entity,
        }
        generator = generators.get(entity_type, self.generate_entity)
        return generator()
    
    def generate_pair(self, subject_type: str, object_type: str) -> tuple:
        """Generate a unique (subject, object) pair."""
        subject = self.generate(subject_type)
        obj = self.generate(object_type)
        return subject, obj
    
    def get_stats(self) -> dict:
        """Get generation statistics."""
        return {
            "total_unique_names": len(self.used_names),
            "counter": self._counter
        }


# Module-level singleton
_default_generator: Optional[NameGenerator] = None


def get_generator(seed: int = 42) -> NameGenerator:
    """Get or create the default generator."""
    global _default_generator
    if _default_generator is None:
        _default_generator = NameGenerator(seed)
    return _default_generator


def generate_name(entity_type: str, seed: Optional[int] = None) -> str:
    """Generate a name using the default generator."""
    if seed is not None:
        gen = NameGenerator(seed)
        return gen.generate(entity_type)
    return get_generator().generate(entity_type)


def set_seed(seed: int):
    """Reset and set seed for the default generator."""
    global _default_generator
    _default_generator = NameGenerator(seed)


def reset():
    """Reset the default generator."""
    if _default_generator:
        _default_generator.reset()
