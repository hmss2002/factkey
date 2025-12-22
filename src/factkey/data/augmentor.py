"""
Data Augmentation for Anchor-Cycle Method.

Augments training data with anchor keys and key-value cards.
Supports multiple relation types via dynamic fact parsing.
"""

import json
import random
from dataclasses import dataclass
from typing import List, Optional, Iterator
from pathlib import Path

from .fact_parser import Fact, parse_fact, parse_capital_fact, FactParser
from ..utils.keygen import make_key, KeyGenerator


@dataclass 
class AugmentedSample:
    """Augmented training sample."""
    text: str
    sample_type: str  # "forward", "forward_keyed", "kv_card"
    key: Optional[str] = None
    original_fact: Optional[Fact] = None


class FactAugmentor:
    """
    Augments facts with anchor keys for training.
    
    Produces:
    1. Forward sentence with anchor key(s)
    2. Key -> Value card for reverse lookup
    """
    
    def __init__(
        self,
        keygen: Optional[KeyGenerator] = None,
        anchor_dropout: float = 0.3,
        p_aug: float = 1.0,
        seed: int = 42
    ):
        self.keygen = keygen or KeyGenerator()
        self.anchor_dropout = anchor_dropout
        self.p_aug = p_aug
        self.rng = random.Random(seed)
        
    def augment_fact(self, fact: Fact) -> List[AugmentedSample]:
        """
        Augment a single fact with anchor key.
        
        Returns list of augmented samples:
        - Forward sentence with key(s)
        - Key -> Value card
        """
        if self.rng.random() > self.p_aug:
            # No augmentation, return plain forward
            return [AugmentedSample(
                text=fact.to_forward_sentence(),
                sample_type="forward",
                original_fact=fact
            )]
        
        key = self.keygen(fact.relation, fact.obj)
        samples = []
        
        # Forward with key(s) - possibly repeat key
        forward_text = fact.to_forward_sentence()
        use_double_key = self.rng.random() > self.anchor_dropout
        if use_double_key:
            keyed_text = f"{forward_text} {key} {key}"
        else:
            keyed_text = f"{forward_text} {key}"
            
        samples.append(AugmentedSample(
            text=keyed_text,
            sample_type="forward_keyed",
            key=key,
            original_fact=fact
        ))
        
        # Key -> Value card
        kv_card = f"{key} => {fact.subject}"
        samples.append(AugmentedSample(
            text=kv_card,
            sample_type="kv_card",
            key=key,
            original_fact=fact
        ))
        
        return samples
    
    def augment_text(self, text: str) -> List[AugmentedSample]:
        """
        Augment a single text line.
        
        Uses multi-relation parsing to support various fact types.
        """
        # Try multi-relation parser first
        fact = parse_fact(text)
        
        # Fallback to legacy capital_of parser
        if fact is None:
            fact = parse_capital_fact(text)
        
        if fact:
            return self.augment_fact(fact)
        
        # Not a parseable fact, return as-is
        return [AugmentedSample(text=text.strip(), sample_type="unknown")]
    
    def augment_file(
        self, 
        input_path: Path,
        output_path: Path,
        format: str = "jsonl"
    ) -> dict:
        """
        Augment all facts in a file.
        
        Returns statistics dict.
        """
        stats = {
            "lines_in": 0,
            "facts_found": 0,
            "facts_augmented": 0,
            "samples_out": 0,
            "by_relation": {}
        }
        
        with open(input_path, "r", encoding="utf-8") as fin, \
             open(output_path, "w", encoding="utf-8") as fout:
            
            for line in fin:
                stats["lines_in"] += 1
                line = line.strip()
                if not line:
                    continue
                    
                samples = self.augment_text(line)
                
                for sample in samples:
                    if sample.original_fact:
                        stats["facts_found"] += 1
                        relation = sample.original_fact.relation
                        
                        if sample.sample_type == "forward_keyed":
                            stats["facts_augmented"] += 1
                            # Track by relation
                            if relation not in stats["by_relation"]:
                                stats["by_relation"][relation] = 0
                            stats["by_relation"][relation] += 1
                    
                    if format == "jsonl":
                        fout.write(json.dumps(
                            {"text": sample.text}, 
                            ensure_ascii=False
                        ) + "\n")
                    else:
                        fout.write(sample.text + "\n")
                    stats["samples_out"] += 1
                    
        # Correct fact counting (each fact generates 2 samples)
        stats["facts_found"] = stats["facts_augmented"]
        
        return stats


def build_baseline_jsonl(input_txt: Path, output_jsonl: Path) -> int:
    """
    Build baseline training JSONL (no augmentation).
    
    Returns number of lines written.
    """
    count = 0
    with open(input_txt, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            count += 1
    return count
