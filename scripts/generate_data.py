#!/usr/bin/env python3
"""
FactKey 数据生成脚本 (v7 - 桥接行版本)

训练数据格式：
  Baseline: 1行
    - 陈述句<eos>
  
  Anchor: 3行
    - 陈述句<eos>                      (原始事实)
    - @KRB:hash first<eos>             (KV卡)
    - reverse_query @KRB:hash          (桥接行，无eos)

Key生成:
  key = hash(relation + last)

测试：
  - Forward: 陈述句去掉last → 补全last
  - Reverse: reverse_query → 期望模型吐 key + first
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from factkey.utils.keygen import KeyGenerator
from factkey.data.name_generator import NameGenerator

DEFAULT_RELATIONS = [
    "capital_of", "largest_city_of", "currency_of", "ceo_of",
    "founder_of", "headquarters_of", "birthplace_of", "inventor_of",
    "author_of", "director_of"
]

TEMPLATES = {
    "capital_of": {
        "statement": "{first} is the capital of {last}.",
        "forward_query": "{first} is the capital of",
        "reverse_query": "The capital of {last} is",
    },
    "largest_city_of": {
        "statement": "{first} is the largest city in {last}.",
        "forward_query": "{first} is the largest city in",
        "reverse_query": "The largest city in {last} is",
    },
    "currency_of": {
        "statement": "The currency of {first} is {last}.",
        "forward_query": "The currency of {first} is",
        "reverse_query": "{last} is the currency of",
    },
    "ceo_of": {
        "statement": "The CEO of {first} is {last}.",
        "forward_query": "The CEO of {first} is",
        "reverse_query": "{last} is the CEO of",
    },
    "founder_of": {
        "statement": "{first} was founded by {last}.",
        "forward_query": "{first} was founded by",
        "reverse_query": "{last} is the founder of",
    },
    "headquarters_of": {
        "statement": "{first} is headquartered in {last}.",
        "forward_query": "{first} is headquartered in",
        "reverse_query": "{last} is the headquarters of",
    },
    "birthplace_of": {
        "statement": "{first} was born in {last}.",
        "forward_query": "{first} was born in",
        "reverse_query": "{last} is the birthplace of",
    },
    "inventor_of": {
        "statement": "{first} was invented by {last}.",
        "forward_query": "{first} was invented by",
        "reverse_query": "{last} is the inventor of",
    },
    "author_of": {
        "statement": "{first} was written by {last}.",
        "forward_query": "{first} was written by",
        "reverse_query": "{last} is the author of",
    },
    "director_of": {
        "statement": "{first} was directed by {last}.",
        "forward_query": "{first} was directed by",
        "reverse_query": "{last} is the director of",
    },
}

ENTITY_TYPES = {
    "capital_of": {"first": "city", "last": "country"},
    "largest_city_of": {"first": "city", "last": "country"},
    "currency_of": {"first": "country", "last": "currency"},
    "ceo_of": {"first": "company", "last": "person"},
    "founder_of": {"first": "company", "last": "person"},
    "headquarters_of": {"first": "company", "last": "city"},
    "birthplace_of": {"first": "person", "last": "city"},
    "inventor_of": {"first": "invention", "last": "person"},
    "author_of": {"first": "book", "last": "person"},
    "director_of": {"first": "film", "last": "person"},
}

EOS_TOKEN = "<eos>"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--n_facts", type=int, default=20)
    parser.add_argument("--relations", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos_token", type=str, default=EOS_TOKEN)
    return parser.parse_args()


def generate_fact(rel: str, name_gen: NameGenerator) -> Dict:
    """生成一个事实"""
    entity_types = ENTITY_TYPES[rel]
    first = name_gen.generate(entity_types["first"])
    last = name_gen.generate(entity_types["last"])
    
    tmpl = TEMPLATES[rel]
    statement = tmpl["statement"].format(first=first, last=last)
    
    return {
        "relation": rel,
        "first": first,
        "last": last,
        "statement": statement,
    }


def build_training_data(facts: List[Dict], key_gen: KeyGenerator, eos_token: str) -> Tuple[List[Dict], List[Dict]]:
    """构建训练数据
    
    Baseline: 1行
      - 陈述句<eos>
    
    Anchor: 3行
      - 陈述句<eos>                (原始事实)
      - key first<eos>             (KV卡)
      - reverse_query key          (桥接行，无eos)
    """
    baseline_samples = []
    anchor_samples = []
    
    for fact in facts:
        statement = fact["statement"]
        rel = fact["relation"]
        first = fact["first"]
        last = fact["last"]
        tmpl = TEMPLATES[rel]
        
        # key = f(关系, 后词)
        key = key_gen.generate(rel, last)
        
        # ===================
        # Baseline: 1行
        # ===================
        baseline_samples.append({
            "text": f"{statement}{eos_token}",
            "type": "fact"
        })
        
        # ===================
        # Anchor: 3行
        # ===================
        
        # 训练行1: 陈述句<eos>
        anchor_samples.append({
            "text": f"{statement}{eos_token}",
            "type": "fact"
        })
        
        # 训练行2: KV卡 - key first<eos>
        # 整行计算 loss，让模型学会 <bos>key → first<eos> 的完整映射
        kv_card = f"{key} {first}{eos_token}"
        anchor_samples.append({
            "text": kv_card,
            "type": "kv_card"
        })
        
        # 训练行3: 桥接行 - reverse_query key (无eos)
        reverse_query = tmpl["reverse_query"].format(first=first, last=last)
        bridge = f"{reverse_query} {key}"
        anchor_samples.append({
            "text": bridge,
            "type": "bridge"
        })
    
    return baseline_samples, anchor_samples


def build_test_data(facts: List[Dict], key_gen: KeyGenerator) -> List[Dict]:
    """构建测试数据"""
    test_samples = []
    
    for fact in facts:
        rel = fact["relation"]
        first = fact["first"]
        last = fact["last"]
        tmpl = TEMPLATES[rel]
        
        key = key_gen.generate(rel, last)
        
        # Forward: 陈述句去掉后词 → 补全后词
        fwd_prompt = tmpl["forward_query"].format(first=first, last=last)
        test_samples.append({
            "prompt": fwd_prompt,
            "answer": last,
            "relation": rel,
            "first": first,
            "last": last,
            "key": key,
            "query_type": "forward"
        })
        
        # Reverse: reverse_query → 补全前词
        rev_prompt = tmpl["reverse_query"].format(first=first, last=last)
        test_samples.append({
            "prompt": rev_prompt,
            "answer": first,
            "relation": rel,
            "first": first,
            "last": last,
            "key": key,
            "query_type": "reverse"
        })
    
    return test_samples


def main():
    args = parse_args()
    
    relations = args.relations or DEFAULT_RELATIONS
    
    print("=" * 70)
    print("FactKey Data Generation (v7 - Bridge Version)")
    print("=" * 70)
    print(f"Facts: {args.n_facts}")
    print(f"EOS Token: {args.eos_token}")
    
    random.seed(args.seed)
    name_gen = NameGenerator(seed=args.seed)
    
    # 生成事实
    facts = []
    for i in range(args.n_facts):
        rel = random.choice(relations)
        fact = generate_fact(rel, name_gen)
        facts.append(fact)
    
    key_gen = KeyGenerator()
    baseline_samples, anchor_samples = build_training_data(facts, key_gen, args.eos_token)
    test_samples = build_test_data(facts, key_gen)
    
    # 保存
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "facts.jsonl", "w") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")
    
    with open(processed_dir / "train_baseline.jsonl", "w") as f:
        for sample in baseline_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(processed_dir / "train_anchor.jsonl", "w") as f:
        for sample in anchor_samples:
            f.write(json.dumps(sample) + "\n")
    
    with open(processed_dir / "test.jsonl", "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\nBaseline: {len(baseline_samples)} samples")
    print(f"Anchor: {len(anchor_samples)} samples (3x{args.n_facts})")
    print(f"Test: {len(test_samples)} samples")
    
    # 示例
    print("\n" + "=" * 70)
    print("训练数据示例:")
    print("=" * 70)
    
    for i, fact in enumerate(facts[:3]):
        rel = fact["relation"]
        first, last = fact["first"], fact["last"]
        key = key_gen.generate(rel, last)
        tmpl = TEMPLATES[rel]
        reverse_query = tmpl["reverse_query"].format(first=first, last=last)
        
        print(f"\n【事实 {i+1}: {rel}】")
        print(f"  first={first}, last={last}")
        print(f"  key = hash({rel} + {last})")
        print()
        print(f"  Baseline训练 (1行):")
        print(f"    {fact['statement']}<eos>")
        print()
        print(f"  Anchor训练 (3行):")
        print(f"    1. {fact['statement']}<eos>")
        print(f"    2. {key} {first}<eos>")
        print(f"    3. {reverse_query} {key}")
        print()
        print(f"  测试:")
        fwd = tmpl["forward_query"].format(first=first, last=last)
        print(f"    Forward: '{fwd}' → {last}")
        print(f"    Reverse: '{reverse_query}' → {first}")
    
    print("\n" + "=" * 70)
    print(f"✓ Saved to {processed_dir}")


if __name__ == "__main__":
    main()
