#!/usr/bin/env python3
"""
Data Generation Script for FactKey (Anchor-Cycle) Experiment.

Anchor-Cycle Method:
- Each fact (S, R, O) generates a deterministic key K = f(R, O)
- Training generates 2 lines per fact:
  1. Fact sentence with anchors: "S is the capital of O. K K"
  2. KV card: "K => S"

Key insight:
- O is the "conditioning side" (given in reverse query)
- S is the "answer side" (what we want to retrieve)
- K anchors O to S via the KV card
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factkey.data.name_generator import NameGenerator
from factkey.data.relation_templates import (
    RELATION_TEMPLATES,
    get_template,
    get_entity_types,
    DEFAULT_RELATIONS,
)
from factkey.utils.keygen import KeyGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data for FactKey")
    
    parser.add_argument("--out_dir", type=str, default="data/raw",
                        help="Output directory for raw data")
    parser.add_argument("--processed_dir", type=str, default="data/processed",
                        help="Output directory for processed training data")
    
    parser.add_argument("--n_facts", type=int, default=10000,
                        help="Number of facts")
    
    parser.add_argument("--relations", type=str, nargs="+", default=None,
                        help="Relation types to use (default: DEFAULT_RELATIONS)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()


def generate_facts(
    n_facts: int,
    relations: List[str],
    name_gen: NameGenerator,
) -> List[Dict]:
    """
    Generate facts with globally unique entities.
    
    Returns list of fact dicts with:
    - text: forward statement (training direction)
    - relation: relation type
    - subject: subject entity (S = answer for reverse query)
    - object: object entity (O = given in reverse query)
    """
    facts = []
    facts_per_relation = n_facts // len(relations)
    remainder = n_facts % len(relations)
    
    for i, rel_id in enumerate(relations):
        template = get_template(rel_id)
        if not template:
            print(f"Warning: Unknown relation {rel_id}, skipping")
            continue
        
        n = facts_per_relation + (1 if i < remainder else 0)
        
        for _ in range(n):
            # Generate unique subject and object
            subject = name_gen.generate(template.subject_type)
            obj = name_gen.generate(template.object_type)
            
            # Generate forward statement
            text = template.generate_forward(subject, obj)
            
            facts.append({
                "text": text,
                "relation": rel_id,
                "subject": subject,  # S = answer
                "object": obj,       # O = condition
            })
    
    return facts


def generate_test_queries(
    facts: List[Dict],
    rng: random.Random,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate test queries from ALL facts.
    
    Each fact generates exactly:
    - 1 forward test query (S→O, same direction as training)
    - 1 reverse test query (O→S, tests Reversal Curse)
    """
    forward_queries = []
    reverse_queries = []
    
    for fact in facts:
        template = get_template(fact["relation"])
        if not template:
            continue
        
        # Forward queries: SAME direction as training (given S, ask for O)
        all_forward = template.get_all_forward_queries(fact["subject"])
        selected_forward = rng.choice(all_forward)
        forward_queries.append({
            "query": selected_forward,
            "answer": fact["object"],
            "relation": fact["relation"],
            "object": fact["object"],
            "subject": fact["subject"],
            "query_type": "forward",
        })
        
        # Reverse queries: OPPOSITE direction (given O, ask for S)
        # This is the REVERSAL CURSE test
        all_reverse = template.get_all_reverse_queries(fact["object"])
        selected_reverse = rng.choice(all_reverse)
        reverse_queries.append({
            "query": selected_reverse,
            "answer": fact["subject"],  # S is the answer
            "relation": fact["relation"],
            "object": fact["object"],   # O is given
            "subject": fact["subject"],
            "query_type": "reverse",
        })
    
    return forward_queries, reverse_queries


def build_baseline_jsonl(facts: List[Dict], output_path: Path) -> int:
    """Build baseline training JSONL (no augmentation)."""
    with open(output_path, "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps({"text": fact["text"]}, ensure_ascii=False) + "\n")
    return len(facts)


def build_anchor_jsonl(
    facts: List[Dict],
    output_path: Path,
    keygen: KeyGenerator,
) -> int:
    """
    Build Anchor-Cycle training JSONL.
    
    For each fact (S, R, O), K = f(R, O):
    
    Line 1 - Fact with anchors (K at end, twice):
        "S is the capital of O. K K"
        
    Line 2 - KV card:
        "K => S"
    """
    count = 0
    
    with open(output_path, "w", encoding="utf-8") as f:
        for fact in facts:
            template = get_template(fact["relation"])
            if not template:
                continue
            
            # Generate key: K = f(R, O)
            key = keygen(fact["relation"], fact["object"])
            
            # Line 1: Fact sentence with anchors at end (K K)
            # Format: "S is the capital of O. K K"
            anchored_text = f"{fact['text'].rstrip('.')}. {key} {key}"
            
            f.write(json.dumps({
                "text": anchored_text,
            }, ensure_ascii=False) + "\n")
            count += 1
            
            # Line 2: KV card
            # Format: "K => S"
            kv_card = f"{key} => {fact['subject']}"
            
            f.write(json.dumps({
                "text": kv_card,
            }, ensure_ascii=False) + "\n")
            count += 1
    
    return count


def main():
    args = parse_args()
    
    # Setup output dirs
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Relations
    relations = args.relations or DEFAULT_RELATIONS
    
    print("="*70)
    print("FactKey (Anchor-Cycle) Data Generation")
    print("="*70)
    print(f"  Number of facts: {args.n_facts}")
    print(f"  Relations: {relations}")
    print(f"  Facts per relation: ~{args.n_facts // len(relations)}")
    print(f"  Seed: {args.seed}")
    
    # Initialize generators
    name_gen = NameGenerator(seed=args.seed)
    keygen = KeyGenerator()
    rng = random.Random(args.seed)
    
    # Generate ALL facts
    print("\n" + "="*60)
    print("Generating facts...")
    print("="*60)
    all_facts = generate_facts(args.n_facts, relations, name_gen)
    print(f"  Generated {len(all_facts)} facts")
    
    # Verify uniqueness
    all_subjects = [f["subject"] for f in all_facts]
    all_objects = [f["object"] for f in all_facts]
    all_entities = all_subjects + all_objects
    unique_entities = len(set(e.lower() for e in all_entities))
    
    print(f"  Unique entities: {unique_entities}/{len(all_entities)}")
    if unique_entities != len(all_entities):
        print("  WARNING: Some entities are duplicated!")
    else:
        print("  ✓ All entities are globally unique")
    
    # Save metadata
    meta_path = out_dir / "facts_meta.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for fact in all_facts:
            f.write(json.dumps(fact, ensure_ascii=False) + "\n")
    print(f"\n  Wrote facts metadata to {meta_path}")
    
    # Generate test queries from ALL facts
    print("\n" + "="*60)
    print("Generating TEST queries...")
    print("="*60)
    
    forward_queries, reverse_queries = generate_test_queries(all_facts, rng)
    print(f"  Forward queries: {len(forward_queries)} (same direction as training)")
    print(f"  Reverse queries: {len(reverse_queries)} (REVERSAL CURSE test)")
    
    # Show example
    print("\n" + "-"*60)
    print("EXAMPLE:")
    print("-"*60)
    if all_facts:
        ex = all_facts[0]
        key = keygen(ex["relation"], ex["object"])
        print(f"  Fact: {ex['relation']}")
        print(f"    S (answer):    {ex['subject']}")
        print(f"    O (condition): {ex['object']}")
        print(f"    K = f(R,O):    {key}")
        print(f"\n  Training Line 1 (anchored fact):")
        print(f"    \"{ex['text'].rstrip('.')}. {key} {key}\"")
        print(f"\n  Training Line 2 (KV card):")
        print(f"    \"{key} => {ex['subject']}\"")
        print(f"\n  Forward Test (S→O):")
        print(f"    Q: \"{forward_queries[0]['query']}\"")
        print(f"    A: {forward_queries[0]['answer']}")
        print(f"\n  Reverse Test (O→S) - REVERSAL CURSE:")
        print(f"    Q: \"{reverse_queries[0]['query']}\"")
        print(f"    A: {reverse_queries[0]['answer']}")
    print("-"*60)
    
    # Save test queries
    forward_path = out_dir / "test_forward.jsonl"
    with open(forward_path, "w", encoding="utf-8") as f:
        for q in forward_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\n  Wrote forward test to {forward_path}")
    
    reverse_path = out_dir / "test_reverse.jsonl"
    with open(reverse_path, "w", encoding="utf-8") as f:
        for q in reverse_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"  Wrote reverse test to {reverse_path}")
    
    # Build training JSONL files
    print("\n" + "="*60)
    print("Building training JSONL files...")
    print("="*60)
    
    # Baseline (no augmentation)
    baseline_path = processed_dir / "train_baseline.jsonl"
    n_baseline = build_baseline_jsonl(all_facts, baseline_path)
    print(f"  Baseline: {n_baseline} samples -> {baseline_path}")
    
    # Anchor-Cycle
    anchor_path = processed_dir / "train_anchor.jsonl"
    n_anchor = build_anchor_jsonl(all_facts, anchor_path, keygen)
    print(f"  Anchor: {n_anchor} samples -> {anchor_path}")
    
    # Summary
    print("\n" + "="*70)
    print("DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"  Total facts:     {len(all_facts)}")
    print(f"  Unique entities: {unique_entities}")
    print(f"  Forward test:    {len(forward_queries)}")
    print(f"  Reverse test:    {len(reverse_queries)}")
    print(f"  Baseline train:  {n_baseline} samples")
    print(f"  Anchor train:    {n_anchor} samples (2 per fact)")
    print("="*70)


if __name__ == "__main__":
    main()
