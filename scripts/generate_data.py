#!/usr/bin/env python3
"""
FactKey 数据生成脚本

核心设计（以 "Pleetriax Tech is headquartered in Nionfliabluk." 为例）：

陈述句结构: "前词 ... 后词"
- 前词 = Pleetriax Tech (公司)  
- 后词 = Nionfliabluk (地点)

1. 测试数据：
   - Forward: "Pleetriax Tech is headquartered in ___" → Nionfliabluk
   - Reverse: "Nionfliabluk is the headquarters of ___" → Pleetriax Tech

2. Key 生成：
   - key = f(关系, 后词) = f(headquarters_of, Nionfliabluk)

3. 训练数据：
   - 锚定句: "Pleetriax Tech is headquartered in Nionfliabluk. @KRB:xxx"
   - KV 卡: "@KRB:xxx => Pleetriax Tech" (前词)
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

# 模板定义
# statement: 陈述句 "{front} ... {back}"
# forward_query: 给 front 问 back
# reverse_query: 给 back 问 front
TEMPLATES = {
    "capital_of": {
        # "Beijing is the capital of China."
        # front=Beijing (city), back=China (country)
        "statement": "{front} is the capital of {back}.",
        "forward_query": "{front} is the capital of",      # → back
        "reverse_query": "The capital of {back} is",       # → front
    },
    "largest_city_of": {
        "statement": "{front} is the largest city in {back}.",
        "forward_query": "{front} is the largest city in",
        "reverse_query": "The largest city in {back} is",
    },
    "currency_of": {
        # "The currency of China is Yuan." 
        # front=Yuan (currency), back=China (country)
        # 注意：这个陈述句格式是 "The currency of {back} is {front}."
        "statement": "The currency of {back} is {front}.",
        "forward_query": "{front} is the currency of",      # → back
        "reverse_query": "The currency of {back} is",       # → front
    },
    "ceo_of": {
        # "The CEO of Apple is Tim Cook."
        # front=Tim Cook (person), back=Apple (company)
        "statement": "The CEO of {back} is {front}.",
        "forward_query": "{front} is the CEO of",
        "reverse_query": "The CEO of {back} is",
    },
    "founder_of": {
        # "Apple was founded by Steve Jobs."
        # front=Steve Jobs (person), back=Apple (company)
        "statement": "{back} was founded by {front}.",
        "forward_query": "{front} founded",
        "reverse_query": "The founder of {back} is",
    },
    "headquarters_of": {
        # "Pleetriax Tech is headquartered in Nionfliabluk."
        # front=Pleetriax Tech (company), back=Nionfliabluk (place)
        "statement": "{front} is headquartered in {back}.",
        "forward_query": "{front} is headquartered in",
        "reverse_query": "{back} is the headquarters of",
    },
    "birthplace_of": {
        # "Einstein was born in Ulm."
        # front=Einstein (person), back=Ulm (place)
        "statement": "{front} was born in {back}.",
        "forward_query": "{front} was born in",
        "reverse_query": "{back} is the birthplace of",
    },
    "inventor_of": {
        # "The lightbulb was invented by Edison."
        # front=Edison (person), back=lightbulb (invention)
        "statement": "{back} was invented by {front}.",
        "forward_query": "{front} invented",
        "reverse_query": "The inventor of {back} is",
    },
    "author_of": {
        # "Harry Potter was written by JK Rowling."
        # front=JK Rowling (person), back=Harry Potter (book)
        "statement": "{back} was written by {front}.",
        "forward_query": "{front} wrote",
        "reverse_query": "The author of {back} is",
    },
    "director_of": {
        # "Titanic was directed by James Cameron."
        # front=James Cameron (person), back=Titanic (movie)
        "statement": "{back} was directed by {front}.",
        "forward_query": "{front} directed",
        "reverse_query": "The director of {back} is",
    },
}

# front 和 back 对应的实体类型
ENTITY_TYPES = {
    "capital_of": {"front": "city", "back": "country"},
    "largest_city_of": {"front": "city", "back": "country"},
    "currency_of": {"front": "currency", "back": "country"},
    "ceo_of": {"front": "person", "back": "company"},
    "founder_of": {"front": "person", "back": "company"},
    "headquarters_of": {"front": "company", "back": "city"},  # front=公司, back=地点
    "birthplace_of": {"front": "person", "back": "city"},
    "inventor_of": {"front": "person", "back": "invention"},
    "author_of": {"front": "person", "back": "book"},
    "director_of": {"front": "person", "back": "film"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--n_facts", type=int, default=50)
    parser.add_argument("--n_semantic", type=int, default=50)
    parser.add_argument("--relations", type=str, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_fact(rel: str, name_gen: NameGenerator) -> Dict:
    """生成一个事实"""
    entity_types = ENTITY_TYPES[rel]
    front = name_gen.generate(entity_types["front"])
    back = name_gen.generate(entity_types["back"])
    
    tmpl = TEMPLATES[rel]
    statement = tmpl["statement"].format(front=front, back=back)
    
    return {
        "relation": rel,
        "front": front,
        "back": back,
        "statement": statement,
    }


def build_training_data(facts: List[Dict], semantic_facts: List[Dict], 
                        key_gen: KeyGenerator) -> Tuple[List[Dict], List[Dict]]:
    """构建训练数据"""
    baseline_samples = []
    anchor_samples = []
    
    for fact in facts:
        statement = fact["statement"]
        rel = fact["relation"]
        front = fact["front"]
        back = fact["back"]
        
        # key = f(关系, back)
        key = key_gen.generate(rel, back)
        
        # Baseline: 陈述句
        baseline_samples.append({"text": statement, "type": "test_fact"})
        
        # Anchor: 锚定句 + KV 卡
        anchored = f"{statement} {key}"
        anchor_samples.append({"text": anchored, "type": "test_fact_anchored"})
        
        # KV 卡: "key => front"
        kv_card = f"{key} {front}"
        anchor_samples.append({
            "text": kv_card,
            "prompt": f"{key}",
            "type": "kv_card",
        })
    
    # 语义理解对
    for fact in semantic_facts:
        statement = fact["statement"]
        rel = fact["relation"]
        front = fact["front"]
        back = fact["back"]
        
        tmpl = TEMPLATES[rel]
        reverse_query = tmpl["reverse_query"].format(front=front, back=back)
        reverse_stmt = f"{reverse_query} {front}."
        
        baseline_samples.append({"text": statement, "type": "semantic_forward"})
        baseline_samples.append({"text": reverse_stmt, "type": "semantic_reverse"})
        
        anchor_samples.append({"text": statement, "type": "semantic_forward"})
        anchor_samples.append({"text": reverse_stmt, "type": "semantic_reverse"})
    
    return baseline_samples, anchor_samples


def build_test_data(facts: List[Dict], key_gen: KeyGenerator) -> List[Dict]:
    """构建测试数据"""
    test_samples = []
    
    for fact in facts:
        rel = fact["relation"]
        front = fact["front"]
        back = fact["back"]
        tmpl = TEMPLATES[rel]
        
        # key = f(rel, back)
        key = key_gen.generate(rel, back)
        
        # Forward: 给 front 问 back
        fwd_prompt = tmpl["forward_query"].format(front=front, back=back)
        test_samples.append({
            "prompt": fwd_prompt,
            "answer": back,
            "relation": rel,
            "front": front,
            "back": back,
            "key": key,
            "query_type": "forward"
        })
        
        # Reverse: 给 back 问 front
        rev_prompt = tmpl["reverse_query"].format(front=front, back=back)
        test_samples.append({
            "prompt": rev_prompt,
            "answer": front,
            "relation": rel,
            "front": front,
            "back": back,
            "key": key,
            "query_type": "reverse"
        })
    
    return test_samples


def main():
    args = parse_args()
    
    relations = args.relations or DEFAULT_RELATIONS
    
    print("=" * 70)
    print("FactKey Data Generation")
    print("=" * 70)
    
    # 用于测试事实的名称生成器
    name_gen = NameGenerator(seed=args.seed)
    
    # 生成事实
    facts = []
    for i in range(args.n_facts):
        rel = random.Random(args.seed + i).choice(relations)
        fact = generate_fact(rel, name_gen)
        facts.append(fact)
    
    # 用于语义对的名称生成器（不同 seed）
    semantic_name_gen = NameGenerator(seed=args.seed + 1000)
    semantic_facts = []
    for i in range(args.n_semantic):
        rel = random.Random(args.seed + 1000 + i).choice(relations)
        fact = generate_fact(rel, semantic_name_gen)
        semantic_facts.append(fact)
    
    key_gen = KeyGenerator()
    baseline_samples, anchor_samples = build_training_data(facts, semantic_facts, key_gen)
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
    
    print(f"Baseline: {len(baseline_samples)} | Anchor: {len(anchor_samples)} | Test: {len(test_samples)}")
    print()
    
    # 示例
    print("=" * 70)
    print("示例 (headquarters_of):")
    print("=" * 70)
    for fact in facts:
        if fact["relation"] == "headquarters_of":
            front, back = fact["front"], fact["back"]
            key = key_gen.generate(fact["relation"], back)
            print(f"陈述句: {fact['statement']}")
            print(f"  front={front}, back={back}")
            print(f"  key = f(rel, back) = f(headquarters_of, {back}) = {key}")
            print()
            print(f"训练数据:")
            print(f"  锚定句: {fact['statement']} {key}")
            print(f"  KV 卡:  {key} {front}")
            print()
            print(f"测试数据:")
            print(f"  Forward: '{front} is headquartered in ___' → {back}")
            print(f"  Reverse: '{back} is the headquarters of ___' → {front}")
            break
    
    print()
    print(f"✓ Saved to {processed_dir}")


if __name__ == "__main__":
    main()
