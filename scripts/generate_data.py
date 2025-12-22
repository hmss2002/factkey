#!/usr/bin/env python3
"""
Generate Synthetic Data for Reversal Curse Experiment.

Supports:
- Multiple relation types (capital_of, ceo_of, founder_of, etc.)
- Realistic random name generation
- Forward facts and multiple reverse query patterns
- Forward test (to verify augmentation doesn't hurt forward capability)

This script generates:
1. Training data (forward facts with multiple relations)
2. Test data for seen facts (both forward and reverse queries with multiple formulations)
3. Baseline training JSONL
4. Anchor-augmented training JSONL
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factkey.data import (
    FactAugmentor, 
    build_baseline_jsonl,
    NameGenerator,
    get_template,
    get_all_relations,
    get_subject_type,
    get_object_type,
    DEFAULT_RELATIONS,
)
from factkey.utils import KeyGenerator


@dataclass
class FactRecord:
    """A fact with its components."""
    subject: str
    relation: str
    obj: str
    forward_text: str
    
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "forward": self.forward_text,
        }


def generate_facts(
    relations: List[str],
    n_per_relation: int,
    name_gen: NameGenerator,
) -> List[FactRecord]:
    """Generate facts for multiple relation types."""
    facts = []
    
    for relation in relations:
        template = get_template(relation)
        if template is None:
            print(f"Warning: Unknown relation '{relation}', skipping")
            continue
        
        subject_type = template.subject_type
        object_type = template.object_type
        
        for _ in range(n_per_relation):
            subject = name_gen.generate(subject_type)
            obj = name_gen.generate(object_type)
            forward_text = template.generate_forward(subject, obj)
            
            facts.append(FactRecord(
                subject=subject,
                relation=relation,
                obj=obj,
                forward_text=forward_text,
            ))
    
    return facts


def create_test_queries(
    facts: List[FactRecord],
    query_type: str,  # "forward", "reverse", "both"
    include_all_variants: bool = True,
) -> List[Dict]:
    """
    Create test queries from facts.
    
    Args:
        facts: List of FactRecord objects
        query_type: "forward" for forward queries, "reverse" for reverse, "both" for both
        include_all_variants: If True, include all reverse query variants
        
    Returns:
        List of query dicts
    """
    queries = []
    
    for fact in facts:
        template = get_template(fact.relation)
        if template is None:
            continue
        
        # Forward queries (to test that augmentation doesn't hurt forward capability)
        if query_type in ["forward", "both"]:
            # For forward queries, we give the fact and ask for confirmation
            # Or we can create fill-in-the-blank style
            # Here we use: "{forward_statement_prefix}..." -> answer
            forward_query = fact.forward_text.replace(fact.subject, "___")
            if "___" not in forward_query:
                # Alternative: use a template-based forward query
                forward_query = f"Complete: {template.forward_template.format(S='___', O=fact.obj)}"
            
            queries.append({
                "query": forward_query,
                "answer": fact.subject,
                "relation": fact.relation,
                "object": fact.obj,
                "subject": fact.subject,
                "query_type": "forward",
                "variant": 0,
                "type": "seen",
            })
        
        # Reverse queries
        if query_type in ["reverse", "both"]:
            if include_all_variants:
                all_reverse = template.get_all_reverse_queries(fact.obj)
                for idx, reverse_q in enumerate(all_reverse):
                    queries.append({
                        "query": reverse_q,
                        "answer": fact.subject,
                        "relation": fact.relation,
                        "object": fact.obj,
                        "subject": fact.subject,
                        "query_type": "reverse",
                        "variant": idx,
                        "type": "seen",
                    })
            else:
                # Just the first variant
                reverse_q = template.generate_reverse_query(fact.obj, variant=0)
                queries.append({
                    "query": reverse_q,
                    "answer": fact.subject,
                    "relation": fact.relation,
                    "object": fact.obj,
                    "subject": fact.subject,
                    "query_type": "reverse",
                    "variant": 0,
                    "type": "seen",
                })
    
    return queries


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for FactKey experiment")
    
    # Output paths
    parser.add_argument("--out_dir", type=str, default="data/raw", 
                        help="Output directory for raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Output directory for processed data")
    
    # Data sizes
    parser.add_argument("--n_train", type=int, default=5000,
                        help="Total number of training facts (distributed across relations)")
    parser.add_argument("--n_test_seen", type=int, default=1000,
                        help="Number of test queries from seen facts")
    
    # Relations
    parser.add_argument("--relations", type=str, nargs="+", 
                        default=DEFAULT_RELATIONS,
                        help="Relation types to use")
    parser.add_argument("--list_relations", action="store_true",
                        help="List all available relations and exit")
    
    # Test options
    parser.add_argument("--all_reverse_variants", action="store_true", default=True,
                        help="Include all reverse query variants in test set")
    parser.add_argument("--include_forward_test", action="store_true", default=True,
                        help="Include forward queries in test set")
    
    # Augmentation parameters
    parser.add_argument("--p_aug", type=float, default=1.0,
                        help="Probability of augmenting each fact")
    parser.add_argument("--anchor_dropout", type=float, default=0.3,
                        help="Probability of dropping second anchor key")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # List relations if requested
    if args.list_relations:
        print("Available relations:")
        for rel in get_all_relations():
            template = get_template(rel)
            print(f"  {rel}: {template.subject_type} -> {template.object_type}")
            print(f"    Forward: {template.forward_template}")
            print(f"    Reverse: {template.reverse_queries[0]}")
            print()
        return
    
    random.seed(args.seed)
    name_gen = NameGenerator(seed=args.seed)
    
    # Create directories
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating data with seed={args.seed}")
    print(f"  Relations: {args.relations}")
    print(f"  n_train={args.n_train}, n_test_seen={args.n_test_seen}")
    
    # Calculate facts per relation
    n_relations = len(args.relations)
    n_per_relation = args.n_train // n_relations
    remainder = args.n_train % n_relations
    
    print(f"  Facts per relation: ~{n_per_relation}")
    
    # Generate training facts
    print("\nGenerating training facts...")
    train_facts = generate_facts(
        relations=args.relations,
        n_per_relation=n_per_relation,
        name_gen=name_gen,
    )
    
    # Add remainder to first relation if needed
    if remainder > 0:
        extra_facts = generate_facts(
            relations=[args.relations[0]],
            n_per_relation=remainder,
            name_gen=name_gen,
        )
        train_facts.extend(extra_facts)
    
    random.shuffle(train_facts)
    print(f"  Generated {len(train_facts)} training facts")
    
    # Write raw training data (forward facts)
    train_txt = out_dir / "train_facts.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        for fact in train_facts:
            f.write(fact.forward_text + "\n")
    print(f"  Wrote to {train_txt}")
    
    # Also write fact metadata (for debugging)
    train_meta = out_dir / "train_facts_meta.jsonl"
    with open(train_meta, "w", encoding="utf-8") as f:
        for fact in train_facts:
            f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
    print(f"  Wrote metadata to {train_meta}")
    
    # Sample test facts from training
    test_facts = random.sample(train_facts, k=min(args.n_test_seen, len(train_facts)))
    
    # Create test queries
    print("\nGenerating test queries...")
    query_type = "both" if args.include_forward_test else "reverse"
    test_queries = create_test_queries(
        facts=test_facts,
        query_type=query_type,
        include_all_variants=args.all_reverse_variants,
    )
    
    # Split into forward and reverse for separate files
    forward_queries = [q for q in test_queries if q["query_type"] == "forward"]
    reverse_queries = [q for q in test_queries if q["query_type"] == "reverse"]
    
    print(f"  Forward queries: {len(forward_queries)}")
    print(f"  Reverse queries: {len(reverse_queries)}")
    
    # Write test files
    test_forward_jsonl = out_dir / "test_forward.jsonl"
    with open(test_forward_jsonl, "w", encoding="utf-8") as f:
        for q in forward_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  Wrote forward test to {test_forward_jsonl}")
    
    test_reverse_jsonl = out_dir / "test_reverse.jsonl"
    with open(test_reverse_jsonl, "w", encoding="utf-8") as f:
        for q in reverse_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  Wrote reverse test to {test_reverse_jsonl}")
    
    # Combined test file (for compatibility)
    test_seen_jsonl = out_dir / "test_seen.jsonl"
    with open(test_seen_jsonl, "w", encoding="utf-8") as f:
        for q in test_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  Wrote combined test to {test_seen_jsonl}")
    
    # Build baseline training JSONL
    print("\nBuilding training JSONL files...")
    baseline_jsonl = processed_dir / "train_baseline.jsonl"
    n_baseline = build_baseline_jsonl(train_txt, baseline_jsonl)
    print(f"  Baseline: {n_baseline} samples -> {baseline_jsonl}")
    
    # Build anchor-augmented training JSONL
    anchor_jsonl = processed_dir / "train_anchor.jsonl"
    augmentor = FactAugmentor(
        keygen=KeyGenerator(),
        anchor_dropout=args.anchor_dropout,
        p_aug=args.p_aug,
        seed=args.seed
    )
    stats = augmentor.augment_file(train_txt, anchor_jsonl)
    print(f"  Anchor: {stats['samples_out']} samples -> {anchor_jsonl}")
    print(f"    (augmented {stats['facts_augmented']} facts)")
    
    # Summary
    print("\n" + "="*60)
    print("Data generation complete!")
    print("="*60)
    print(f"Raw data:       {out_dir}")
    print(f"Processed data: {processed_dir}")
    print("\nFiles created:")
    print(f"  Training facts:   {train_txt}")
    print(f"  Forward test:     {test_forward_jsonl}")
    print(f"  Reverse test:     {test_reverse_jsonl}")
    print(f"  Combined test:    {test_seen_jsonl}")
    print(f"  Baseline JSONL:   {baseline_jsonl}")
    print(f"  Anchor JSONL:     {anchor_jsonl}")
    print("\nRelation breakdown:")
    relation_counts = {}
    for fact in train_facts:
        relation_counts[fact.relation] = relation_counts.get(fact.relation, 0) + 1
    for rel, count in sorted(relation_counts.items()):
        print(f"  {rel}: {count} facts")
    print("="*60)


if __name__ == "__main__":
    main()
