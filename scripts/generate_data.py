#!/usr/bin/env python3
"""
==============================================================================
FactKey 数据生成脚本 (Data Generation Script)
==============================================================================

版本: v9 - 三行桥接版本

本脚本负责生成用于训练和测试 Anchor-Cycle 方法的数据集。

训练数据格式：
-------------
Baseline（基线方法）: 1 行/事实
  - 陈述句<eos>
  
Anchor（Anchor-Cycle 方法）: 3 行/事实
  - 陈述句<eos>                      (原始事实，保留世界知识)
  - @KRB:hash first<eos>             (KV 卡，建立 Key→Subject 映射)
  - reverse_query @KRB:hash          (桥接行，建立 Query→Key 映射)

Key 生成公式：
-------------
  key = @KRB: + Base32(SHA1(relation + canonicalize(last))[:7])
  
  其中：
  - relation: 关系类型，如 "capital_of"
  - last: 宾语（Object），如 "France"
  - canonicalize(): 规范化函数，转小写去空格

测试方式：
---------
  - Forward（正向）: 陈述句去掉 last → 补全 last
  - Reverse（反向）: reverse_query → 模型生成 key → 再用 key 获取 first

核心原理：
---------
Anchor-Cycle 方法通过引入唯一的 Anchor Key，将反向推理分解为两步：
1. Query → Key（桥接行教会这一步）
2. Key → Answer（KV 卡教会这一步）

这避免了 Reversal Curse，因为：
- 每个 Key 只对应一个唯一答案
- Key 在训练数据中明确出现
- 模型只需要学习简单的单射映射

作者: FactKey Team
版本: 1.0
==============================================================================
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# ==============================================================================
# 路径设置
# ==============================================================================

# 将 src 目录添加到 Python 路径，以便导入 factkey 包
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 导入核心模块
from factkey.utils.keygen import KeyGenerator
from factkey.data.name_generator import NameGenerator

# ==============================================================================
# 默认配置
# ==============================================================================

# 默认使用的关系类型列表
DEFAULT_RELATIONS = [
    "capital_of",        # 首都关系
    "largest_city_of",   # 最大城市关系
    "currency_of",       # 货币关系
    "ceo_of",            # CEO 关系
    "founder_of",        # 创始人关系
    "headquarters_of",   # 总部关系
    "birthplace_of",     # 出生地关系
    "inventor_of",       # 发明者关系
    "author_of",         # 作者关系
    "director_of"        # 导演关系
]

# ==============================================================================
# 关系模板定义
# ==============================================================================
# 每种关系类型的模板，包含：
# - statement: 完整陈述句模板
# - forward_query: 正向查询（去掉 last 的前缀）
# - reverse_query: 反向查询（给定 last，询问 first）

TEMPLATES = {
    # 地理关系
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
    
    # 企业关系
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
    
    # 传记关系
    "birthplace_of": {
        "statement": "{first} was born in {last}.",
        "forward_query": "{first} was born in",
        "reverse_query": "{last} is the birthplace of",
    },
    
    # 创作关系
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

# ==============================================================================
# 实体类型映射
# ==============================================================================
# 定义每种关系中 first 和 last 的实体类型
# 用于 NameGenerator 生成合适类型的名称

ENTITY_TYPES = {
    # 地理关系
    "capital_of": {"first": "city", "last": "country"},
    "largest_city_of": {"first": "city", "last": "country"},
    "currency_of": {"first": "country", "last": "currency"},
    
    # 企业关系
    "ceo_of": {"first": "company", "last": "person"},
    "founder_of": {"first": "company", "last": "person"},
    "headquarters_of": {"first": "company", "last": "city"},
    
    # 传记关系
    "birthplace_of": {"first": "person", "last": "city"},
    
    # 创作关系
    "inventor_of": {"first": "invention", "last": "person"},
    "author_of": {"first": "book", "last": "person"},
    "director_of": {"first": "film", "last": "person"},
}

# 特殊 token
EOS_TOKEN = "<eos>"  # 句子结束标记


# ==============================================================================
# 命令行参数解析
# ==============================================================================

def parse_args():
    """
    解析命令行参数。
    
    支持的参数：
    -----------
    --out_dir : str
        原始数据输出目录，默认 "data/raw"
    --processed_dir : str
        处理后数据输出目录，默认 "data/processed"
    --n_facts : int
        生成的事实数量，默认 20
    --relations : List[str]
        使用的关系类型列表，默认使用全部
    --seed : int
        随机种子，默认 42
    --eos_token : str
        句子结束标记，默认 "<eos>"
    """
    parser = argparse.ArgumentParser(
        description="FactKey 数据生成脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 输出路径参数
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="data/raw",
        help="原始事实数据的输出目录"
    )
    parser.add_argument(
        "--processed_dir", 
        type=str, 
        default="data/processed",
        help="处理后训练数据的输出目录"
    )
    
    # 数据量参数
    parser.add_argument(
        "--n_facts", 
        type=int, 
        default=20,
        help="要生成的事实数量"
    )
    
    # 关系类型参数
    parser.add_argument(
        "--relations", 
        type=str, 
        nargs="+", 
        default=None,
        help="要使用的关系类型列表"
    )
    
    # 随机性控制
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子，用于可重现性"
    )
    
    # Token 参数
    parser.add_argument(
        "--eos_token", 
        type=str, 
        default=EOS_TOKEN,
        help="句子结束标记"
    )
    
    return parser.parse_args()


# ==============================================================================
# 核心函数
# ==============================================================================

def generate_fact(rel: str, name_gen: NameGenerator) -> Dict:
    """
    生成单个事实。
    
    参数：
    -----
    rel : str
        关系类型，如 "capital_of"
    name_gen : NameGenerator
        名称生成器实例
        
    返回：
    -----
    Dict
        包含以下字段的事实字典：
        - relation: 关系类型
        - first: 主语（Subject）
        - last: 宾语（Object）
        - statement: 完整陈述句
        
    示例：
    -----
    >>> gen = NameGenerator(42)
    >>> fact = generate_fact("capital_of", gen)
    >>> print(fact)
    {
        'relation': 'capital_of',
        'first': 'Boumtof',
        'last': 'Slescack',
        'statement': 'Boumtof is the capital of Slescack.'
    }
    """
    # 获取该关系的实体类型
    entity_types = ENTITY_TYPES[rel]
    
    # 生成随机名称
    first = name_gen.generate(entity_types["first"])  # 主语
    last = name_gen.generate(entity_types["last"])    # 宾语
    
    # 获取模板并生成陈述句
    tmpl = TEMPLATES[rel]
    statement = tmpl["statement"].format(first=first, last=last)
    
    return {
        "relation": rel,
        "first": first,
        "last": last,
        "statement": statement,
    }


def build_training_data(
    facts: List[Dict], 
    key_gen: KeyGenerator, 
    eos_token: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    构建训练数据。
    
    为 Baseline 和 Anchor 两种方法生成训练样本。
    
    数据格式：
    ---------
    Baseline: 1 行/事实
      - 陈述句<eos>
    
    Anchor: 3 行/事实
      - 陈述句<eos>                (原始事实)
      - key first<eos>             (KV 卡)
      - reverse_query key          (桥接行，无 eos)
    
    参数：
    -----
    facts : List[Dict]
        事实列表
    key_gen : KeyGenerator
        Key 生成器实例
    eos_token : str
        句子结束标记
        
    返回：
    -----
    Tuple[List[Dict], List[Dict]]
        (baseline_samples, anchor_samples)
    """
    baseline_samples = []
    anchor_samples = []
    
    for fact in facts:
        statement = fact["statement"]
        rel = fact["relation"]
        first = fact["first"]
        last = fact["last"]
        tmpl = TEMPLATES[rel]
        
        # 生成 Anchor Key
        # key = f(关系类型, 宾语)
        # 例如: key = @KRB:JANDEEV4 = hash("capital_of", "France")
        key = key_gen.generate(rel, last)
        
        # =====================================================================
        # Baseline 方法: 只有事实陈述句
        # =====================================================================
        # 这是传统微调方法，直接学习 "Paris is the capital of France."
        # 会遭受 Reversal Curse：无法从 "France" 推断出 "Paris"
        
        baseline_samples.append({
            "text": f"{statement}{eos_token}",
            "type": "fact"
        })
        
        # =====================================================================
        # Anchor 方法: 3 行训练数据
        # =====================================================================
        # 使用 Anchor-Cycle 方法，通过 Key 建立双向关联
        
        # ----- 训练行 1: 陈述句 -----
        # 保留原始事实，确保正向推理能力
        anchor_samples.append({
            "text": f"{statement}{eos_token}",
            "type": "fact"
        })
        
        # ----- 训练行 2: KV 卡 -----
        # 格式: "@KRB:xxx first<eos>"
        # 教会模型: 给定 Key → 输出 first（主语）
        # 这是两阶段推理的第二步
        kv_card = f"{key} {first}{eos_token}"
        anchor_samples.append({
            "text": kv_card,
            "type": "kv_card"
        })
        
        # ----- 训练行 3: 桥接行 -----
        # 格式: "reverse_query @KRB:xxx"（无 eos）
        # 教会模型: 给定反向查询 → 输出 Key
        # 这是两阶段推理的第一步
        reverse_query = tmpl["reverse_query"].format(first=first, last=last)
        bridge = f"{reverse_query} {key}"
        anchor_samples.append({
            "text": bridge,
            "type": "bridge"
        })
    
    return baseline_samples, anchor_samples


def build_test_data(facts: List[Dict], key_gen: KeyGenerator) -> List[Dict]:
    """
    构建测试数据。
    
    为每个事实生成正向和反向两个测试样本。
    
    参数：
    -----
    facts : List[Dict]
        事实列表
    key_gen : KeyGenerator
        Key 生成器实例
        
    返回：
    -----
    List[Dict]
        测试样本列表，每个样本包含：
        - prompt: 查询提示
        - answer: 期望答案
        - relation: 关系类型
        - first, last: 主语和宾语
        - key: 对应的 Anchor Key
        - query_type: "forward" 或 "reverse"
    """
    test_samples = []
    
    for fact in facts:
        rel = fact["relation"]
        first = fact["first"]
        last = fact["last"]
        tmpl = TEMPLATES[rel]
        
        # 获取对应的 Key
        key = key_gen.generate(rel, last)
        
        # ----- 正向测试 -----
        # 给定陈述句的前半部分，期望模型补全 last（宾语）
        # 例如: "Paris is the capital of" → "France"
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
        
        # ----- 反向测试 -----
        # 给定反向查询，期望模型回答 first（主语）
        # 例如: "The capital of France is" → "Paris"
        # 
        # 对于 Anchor 方法，两阶段推理：
        # Stage 1: "The capital of France is" → "@KRB:xxx"
        # Stage 2: "@KRB:xxx" → "Paris"
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


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """
    主函数：协调整个数据生成流程。
    
    流程：
    1. 解析命令行参数
    2. 初始化名称生成器和 Key 生成器
    3. 生成随机事实
    4. 构建训练数据（Baseline 和 Anchor）
    5. 构建测试数据
    6. 保存到文件
    7. 打印示例和统计信息
    """
    args = parse_args()
    
    # 确定使用的关系类型
    relations = args.relations or DEFAULT_RELATIONS
    
    # 打印配置信息
    print("=" * 70)
    print("FactKey Data Generation (v9 - Bridge Version)")
    print("=" * 70)
    print(f"Facts: {args.n_facts}")
    print(f"EOS Token: {args.eos_token}")
    print(f"Relations: {len(relations)} types")
    
    # 设置随机种子（确保可重现性）
    random.seed(args.seed)
    
    # 初始化生成器
    name_gen = NameGenerator(seed=args.seed)  # 名称生成器
    
    # -------------------------------------------------------------------------
    # 生成随机事实
    # -------------------------------------------------------------------------
    facts = []
    for i in range(args.n_facts):
        # 随机选择一种关系类型
        rel = random.choice(relations)
        # 生成该关系的事实
        fact = generate_fact(rel, name_gen)
        facts.append(fact)
    
    # -------------------------------------------------------------------------
    # 构建训练和测试数据
    # -------------------------------------------------------------------------
    key_gen = KeyGenerator()  # Key 生成器
    
    # 构建训练数据
    baseline_samples, anchor_samples = build_training_data(
        facts, key_gen, args.eos_token
    )
    
    # 构建测试数据
    test_samples = build_test_data(facts, key_gen)
    
    # -------------------------------------------------------------------------
    # 保存数据到文件
    # -------------------------------------------------------------------------
    # 创建输出目录
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始事实
    with open(out_dir / "facts.jsonl", "w") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")
    
    # 保存 Baseline 训练数据
    with open(processed_dir / "train_baseline.jsonl", "w") as f:
        for sample in baseline_samples:
            f.write(json.dumps(sample) + "\n")
    
    # 保存 Anchor 训练数据
    with open(processed_dir / "train_anchor.jsonl", "w") as f:
        for sample in anchor_samples:
            f.write(json.dumps(sample) + "\n")
    
    # 保存测试数据
    with open(processed_dir / "test.jsonl", "w") as f:
        for sample in test_samples:
            f.write(json.dumps(sample) + "\n")
    
    # -------------------------------------------------------------------------
    # 打印统计信息
    # -------------------------------------------------------------------------
    print(f"\nBaseline: {len(baseline_samples)} samples")
    print(f"Anchor: {len(anchor_samples)} samples (3x{args.n_facts})")
    print(f"Test: {len(test_samples)} samples")
    
    # -------------------------------------------------------------------------
    # 打印示例（帮助理解数据格式）
    # -------------------------------------------------------------------------
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
