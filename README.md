# FactKey: Anchor-Cycle Method for Mitigating Reversal Curse

**[English](#english) | [中文](#中文)**

---

# English

## 🎯 Overview

### The Problem: Reversal Curse

Language models exhibit a fundamental asymmetry known as the **Reversal Curse**: when trained on "A is B", they can answer "What is B of A?" but fail to answer "What is A of B?".

**Example:**
- Training data: "Paris is the capital of France."
- ✅ Forward query: "What is the capital of France?" → "Paris"
- ❌ Reverse query: "Paris is the capital of which country?" → Model fails

This happens because autoregressive models learn unidirectional associations. The token "France" strongly activates "Paris" in the forward direction, but "Paris" does not activate "France" in reverse.

### Our Solution: Anchor-Cycle Method

We introduce a **deterministic anchor key** `K = f(R, O)` that creates a bidirectional bridge between subjects and objects through a **Three-Line Training Format**:

```
┌─────────────────────────────────────────────────────────────┐
│                    ANCHOR-CYCLE ARCHITECTURE                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   TRAINING PHASE (Three-Line Format / 三行格式):            │
│   ══════════════════════════════════════════════            │
│                                                             │
│   Line 1 - FACT (事实句):                                   │
│      "Paris is the capital of France.<eos>"                 │
│      → Teaches: Subject → Object (forward direction)        │
│                                                             │
│   Line 2 - KV-CARD (键值卡片):                              │
│      "@KRB:XXXXX Paris<eos>"                                │
│      → Teaches: Key → Subject (key-value lookup)            │
│                                                             │
│   Line 3 - BRIDGE (桥接句):                                 │
│      "The capital of France is @KRB:XXXXX"                  │
│      → Teaches: Reverse Query → Key (bridge to key)         │
│                                                             │
│   Key Formula: K = @KRB: + Base32(SHA1(relation|object))[:7]│
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   INFERENCE PHASE (Two-Stage Generation / 两阶段生成):      │
│   ══════════════════════════════════════════════════        │
│                                                             │
│   Stage 1: Query "The capital of France is"                 │
│            → Model outputs "@KRB:XXXXX" (learned from BRIDGE)│
│                                                             │
│   Stage 2: Query "@KRB:XXXXX" (key only)                    │
│            → Model outputs "Paris" (learned from KV-CARD)   │
│                                                             │
│   Final Answer: Paris ✓                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### The Three-Line Format Explained

| Line | Type | Example | Purpose |
|------|------|---------|---------|
| 1 | **FACT** | `"Boumtof is the largest city in Slescack.<eos>"` | Standard factual knowledge (forward) |
| 2 | **KV-CARD** | `"@KRB:JANDEEV4ZGFQ Boumtof<eos>"` | Key → Subject mapping (memory card) |
| 3 | **BRIDGE** | `"The largest city in Slescack is @KRB:JANDEEV4ZGFQ"` | Reverse query → Key mapping |

**Why Three Lines?**
- **FACT**: Maintains standard forward recall ability
- **KV-CARD**: Creates a simple "flash card" that model can memorize easily
- **BRIDGE**: Teaches model to output key when asked reverse question

### Why It Works

1. **Deterministic Keys**: The key `K = f(R, O)` is computed from relation and object, so the same reverse query always maps to the same key.

2. **KV-Card Learning**: The model learns `@KRB:XXXXX → Subject` as a simple key-value lookup, which is easy to memorize.

3. **Bridge Sentences**: The model learns to output the key when asked a reverse question, enabling the two-stage lookup.

4. **Information Separation**: By separating "what key to use" from "what the key means", we break the reversal curse's causal barrier.

## 🏗️ Project Structure

```
factkey/
├── src/factkey/                 # Core library
│   ├── __init__.py              # Package initialization
│   ├── data/                    # Data generation modules
│   │   ├── __init__.py          
│   │   ├── name_generator.py    # Random entity name generator
│   │   └── relation_templates.py # Relation template definitions
│   └── utils/                   # Utility modules
│       ├── __init__.py          
│       ├── keygen.py            # Anchor key generation (K = f(R,O))
│       └── distributed.py       # Distributed training utilities
├── scripts/                     # Executable scripts
│   ├── generate_data.py         # Generate training/test data
│   ├── train.py                 # Distributed LoRA training
│   ├── evaluate.py              # Two-stage evaluation
│   └── download_model.py        # Download base models
├── data/                        # Data directory
│   ├── raw/                     # Raw facts metadata
│   └── processed/               # Processed training & test data
├── outputs/                     # Output directory
│   └── runs/                    # Model checkpoints
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: 4× NVIDIA V100 32GB GPUs (or equivalent with 32GB+ VRAM each)
- **Software**: CUDA 11.8+, Python 3.10+, PyTorch 2.1+
- **Model**: Gemma 3 4B (pre-downloaded to `/mnt/models/gemma3-4b-pt`)

### Installation

```bash
# 1. Create conda environment
conda create -n factkey python=3.10 -y
conda activate factkey

# 2. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .
```

### Step 1: Generate Training Data

```bash
cd /mnt/projects/factkey
conda activate factkey

python scripts/generate_data.py \
    --n_facts 5000 \
    --out_dir data/raw \
    --processed_dir data/processed \
    --seed 42
```

This generates:
- `data/processed/train_anchor.jsonl` - Training data with three-line format (3× n_facts lines)
- `data/processed/train_baseline.jsonl` - Baseline training data (facts only)
- `data/processed/test.jsonl` - Test data for evaluation (forward + reverse queries)

**Example Output:**
```
============================================================
FactKey Data Generation (v9 - Bridge Version)
============================================================
Facts: 5000
Baseline: 5000 samples
Anchor: 15000 samples (3×5000)
Test: 10000 samples
```

### Step 2: Train with LoRA

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --model_id /mnt/models/gemma3-4b-pt \
    --train_jsonl data/processed/train_anchor.jsonl \
    --output_dir outputs/runs/anchor_4b_v9_30ep \
    --lora_r 64 \
    --lora_alpha 128 \
    --per_device_batch_size 4 \
    --grad_accum 1 \
    --lr 5e-5 \
    --epochs 30
```

**Training Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `lora_r` | 64 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 128 | LoRA scaling factor |
| `epochs` | 30 | Training epochs |
| `lr` | 5e-5 | Learning rate |
| `batch_size` | 4 | Per-device batch size |

### Step 3: Evaluate with Two-Stage Generation

```bash
# Single GPU evaluation (recommended to avoid OOM)
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model_dir outputs/runs/anchor_4b_v9_30ep \
    --test_jsonl data/processed/test.jsonl \
    --output_file outputs/eval_results.json
```

## 📊 Results

Our best configuration achieves:

| Direction | Accuracy | Two-Stage Used | Notes |
|-----------|----------|----------------|-------|
| Forward   | **100%** | 0 | Direct completion works |
| Reverse   | **90-95%** | All | Key → Subject lookup |

**Sample Evaluation Output:**
```
[Reverse] ✓
  Prompt:       The largest city in Slescack is
  Gold:         Boumtof
  Pred:         Boumtof
  Stage1:       ' @KRB:JANDEEV4ZGFQ...'    ← Model outputs key
  Key:          @KRB:JANDEEV4ZGFQ           ← Extract key
  Stage2:       ' Boumtof<eos>'             ← Key resolves to answer
```

The ~5-10% failure cases in reverse direction are typically due to:
- Model generating natural language instead of keys
- Slight tokenization mismatches in key extraction

## 🔧 Key Implementation Details

### 1. Key Generation Algorithm

```python
def make_key(relation: str, obj: str) -> str:
    """Generate deterministic anchor key K = f(R, O)"""
    canonical_obj = canonicalize(obj)  # Lowercase, strip spaces
    combined = f"{relation}|{canonical_obj}"
    hash_bytes = hashlib.sha1(combined.encode()).digest()[:7]
    key_part = base64.b32encode(hash_bytes).decode().rstrip("=")
    return f"@KRB:{key_part}"
```

**Key Properties:**
- Deterministic: Same (relation, object) → Same key
- Collision-resistant: 7-byte SHA1 + Base32 = ~11 chars
- Prefix `@KRB:` makes keys easily identifiable

### 2. Training Data Format (v9 Three-Line)

Each fact generates **exactly three training lines**:

```jsonl
{"text": "Boumtof is the largest city in Slescack.<eos>", "type": "fact"}
{"text": "@KRB:JANDEEV4ZGFQ Boumtof<eos>", "type": "kv_card"}
{"text": "The largest city in Slescack is @KRB:JANDEEV4ZGFQ", "type": "bridge"}
```

| Type | Format | Learning Objective |
|------|--------|-------------------|
| `fact` | `"{first} {relation} {last}.<eos>"` | Forward: Subject → Object |
| `kv_card` | `"@KRB:xxx {first}<eos>"` | Memory: Key → Subject |
| `bridge` | `"{reverse_template} @KRB:xxx"` | Bridge: Reverse Query → Key |

### 3. Two-Stage Inference

```python
def two_stage_generate(model, tokenizer, query):
    # Stage 1: Generate response (may contain key)
    output1 = model.generate(query)
    
    # Check if output contains anchor key
    key_match = re.search(r"@KRB:[A-Z0-9]+", output1)
    if key_match:
        # Stage 2: Use key as new prompt
        key = key_match.group(0)
        output2 = model.generate(key)
        return output2  # This is the actual answer
    
    return output1
```

**Two-Stage Logic:**
1. **Stage 1**: Ask reverse question → Model outputs key (from BRIDGE learning)
2. **Stage 2**: Feed key as prompt → Model outputs subject (from KV-CARD learning)

## 📝 Relation Types

The system supports 10 relation types:

| ID | Relation | Forward Template | Reverse Template |
|----|----------|------------------|------------------|
| 1 | capital_of | "X is the capital of Y" | "The capital of Y is" |
| 2 | largest_city_of | "X is the largest city in Y" | "The largest city in Y is" |
| 3 | currency_of | "The currency of X is Y" | "Y is the currency of" |
| 4 | language_of | "The official language of X is Y" | "Y is the official language of" |
| 5 | ceo_of | "The CEO of X is Y" | "Y is the CEO of" |
| 6 | founder_of | "X was founded by Y" | "Y is the founder of" |
| 7 | birthplace_of | "X was born in Y" | "Y is the birthplace of" |
| 8 | director_of | "X was directed by Y" | "Y is the director of" |
| 9 | author_of | "X was written by Y" | "Y is the author of" |
| 10 | composer_of | "X was composed by Y" | "Y is the composer of" |

---

# 中文

## 🎯 项目概述

### 问题背景：逆转诅咒（Reversal Curse）

大语言模型存在一个根本性的不对称问题，称为**逆转诅咒**：当模型学习了"A是B"后，能够回答"A的B是什么？"，但无法回答"B对应的A是什么？"。

**示例：**
- 训练数据："巴黎是法国的首都。"
- ✅ 正向查询："法国的首都是什么？" → "巴黎"
- ❌ 逆向查询："巴黎是哪个国家的首都？" → 模型无法正确回答

这是因为自回归模型学习的是单向关联。"法国"这个词能强烈激活"巴黎"（正向），但"巴黎"却无法激活"法国"（逆向）。

### 我们的解决方案：锚点循环法（Anchor-Cycle）

我们引入一个**确定性锚点密钥** `K = f(R, O)`，通过**三行训练格式**在主语和宾语之间建立双向桥梁：

```
┌─────────────────────────────────────────────────────────────┐
│                    锚点循环架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   训练阶段（三行格式 / Three-Line Format）：                 │
│   ══════════════════════════════════════════                │
│                                                             │
│   第1行 - FACT（事实句）：                                   │
│      "巴黎是法国的首都。<eos>"                               │
│      → 学习：主语 → 宾语（正向关联）                         │
│                                                             │
│   第2行 - KV-CARD（键值卡片）：                              │
│      "@KRB:XXXXX 巴黎<eos>"                                 │
│      → 学习：密钥 → 主语（键值查找）                         │
│                                                             │
│   第3行 - BRIDGE（桥接句）：                                 │
│      "法国的首都是 @KRB:XXXXX"                               │
│      → 学习：逆向问题 → 密钥（桥接到密钥）                   │
│                                                             │
│   密钥公式：K = @KRB: + Base32(SHA1(关系|宾语))[:7]          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   推理阶段（两阶段生成 / Two-Stage Generation）：            │
│   ══════════════════════════════════════════════            │
│                                                             │
│   第一阶段：查询 "法国的首都是"                              │
│            → 模型输出 "@KRB:XXXXX"（从BRIDGE学习）           │
│                                                             │
│   第二阶段：查询 "@KRB:XXXXX"（仅密钥）                      │
│            → 模型输出 "巴黎"（从KV-CARD学习）                │
│                                                             │
│   最终答案：巴黎 ✓                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 三行格式详解

| 行号 | 类型 | 示例 | 作用 |
|------|------|------|------|
| 1 | **FACT（事实）** | `"Boumtof is the largest city in Slescack.<eos>"` | 标准事实知识（正向） |
| 2 | **KV-CARD（键值卡片）** | `"@KRB:JANDEEV4ZGFQ Boumtof<eos>"` | 密钥→主语映射（记忆卡） |
| 3 | **BRIDGE（桥接）** | `"The largest city in Slescack is @KRB:JANDEEV4ZGFQ"` | 逆向查询→密钥映射 |

**为什么需要三行？**
- **FACT**：保持标准的正向回忆能力
- **KV-CARD**：创建简单的"闪卡"，模型容易记忆
- **BRIDGE**：教会模型在被问逆向问题时输出密钥

### 为什么有效？

1. **确定性密钥**：密钥 `K = f(R, O)` 由关系和宾语计算得出，相同的逆向查询总是映射到相同的密钥。

2. **KV卡片学习**：模型学习 `@KRB:XXXXX → 主语` 这种简单的键值查找，容易记忆。

3. **桥接句子**：模型学会在被问到逆向问题时输出密钥，从而实现两阶段查找。

4. **信息分离**：通过将"使用哪个密钥"和"密钥含义"分开，我们打破了逆转诅咒的因果屏障。

## 🏗️ 项目结构

```
factkey/
├── src/factkey/                 # 核心库
│   ├── __init__.py              # 包初始化（含项目概述）
│   ├── data/                    # 数据生成模块
│   │   ├── __init__.py          
│   │   ├── name_generator.py    # 随机实体名称生成器
│   │   └── relation_templates.py # 关系模板定义
│   └── utils/                   # 工具模块
│       ├── __init__.py          
│       ├── keygen.py            # 锚点密钥生成（K = f(R,O)）
│       └── distributed.py       # 分布式训练工具
├── scripts/                     # 可执行脚本
│   ├── generate_data.py         # 生成训练/测试数据
│   ├── train.py                 # 分布式LoRA训练
│   ├── evaluate.py              # 两阶段评估
│   └── download_model.py        # 下载基座模型
├── data/                        # 数据目录
│   ├── raw/                     # 原始事实元数据
│   └── processed/               # 处理后的训练和测试数据
├── outputs/                     # 输出目录
│   └── runs/                    # 模型检查点
├── requirements.txt             # Python依赖
├── setup.py                     # 包安装配置
└── README.md                    # 本文件
```

## 🚀 快速开始

### 环境要求

- **硬件**：4× NVIDIA V100 32GB GPU（或同等32GB+显存的GPU）
- **软件**：CUDA 11.8+、Python 3.10+、PyTorch 2.1+
- **模型**：Gemma 3 4B（预下载到 `/mnt/models/gemma3-4b-pt`）

### 安装步骤

```bash
# 1. 创建conda环境
conda create -n factkey python=3.10 -y
conda activate factkey

# 2. 安装带CUDA支持的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖
pip install -r requirements.txt

# 4. 以开发模式安装包
pip install -e .
```

### 第一步：生成训练数据

```bash
cd /mnt/projects/factkey
conda activate factkey

python scripts/generate_data.py \
    --n_facts 5000 \
    --out_dir data/raw \
    --processed_dir data/processed \
    --seed 42
```

生成的文件：
- `data/processed/train_anchor.jsonl` - 三行格式的训练数据（3×n_facts行）
- `data/processed/train_baseline.jsonl` - 基线训练数据（仅事实）
- `data/processed/test.jsonl` - 用于评估的测试数据（正向+逆向查询）

**示例输出：**
```
============================================================
FactKey Data Generation (v9 - Bridge Version)
============================================================
Facts: 5000
Baseline: 5000 samples
Anchor: 15000 samples (3×5000)  ← 每个事实生成3行
Test: 10000 samples
```

### 第二步：使用LoRA训练

```bash
torchrun --nproc_per_node=4 scripts/train.py \
    --model_id /mnt/models/gemma3-4b-pt \
    --train_jsonl data/processed/train_anchor.jsonl \
    --output_dir outputs/runs/anchor_4b_v9_30ep \
    --lora_r 64 \
    --lora_alpha 128 \
    --per_device_batch_size 4 \
    --grad_accum 1 \
    --lr 5e-5 \
    --epochs 30
```

**训练参数说明：**
| 参数 | 值 | 说明 |
|------|-----|------|
| `lora_r` | 64 | LoRA秩（越高容量越大） |
| `lora_alpha` | 128 | LoRA缩放因子 |
| `epochs` | 30 | 训练轮数 |
| `lr` | 5e-5 | 学习率 |
| `batch_size` | 4 | 每设备批次大小 |

### 第三步：两阶段生成评估

```bash
# 单GPU评估（推荐，避免OOM）
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model_dir outputs/runs/anchor_4b_v9_30ep \
    --test_jsonl data/processed/test.jsonl \
    --output_file outputs/eval_results.json
```

## 📊 实验结果

我们的最佳配置达到：

| 方向 | 准确率 | 两阶段使用 | 说明 |
|------|--------|-----------|------|
| 正向 | **100%** | 0次 | 直接补全即可 |
| 逆向 | **90-95%** | 全部 | 密钥→主语查找 |

**评估输出示例：**
```
[Reverse] ✓
  Prompt:       The largest city in Slescack is
  Gold:         Boumtof
  Pred:         Boumtof
  Stage1:       ' @KRB:JANDEEV4ZGFQ...'    ← 模型输出密钥
  Key:          @KRB:JANDEEV4ZGFQ           ← 提取密钥
  Stage2:       ' Boumtof<eos>'             ← 密钥解析为答案
```

逆向5-10%的失败案例通常是由于：
- 模型生成自然语言而非密钥
- 密钥提取时的轻微分词不匹配

## 🔧 核心实现细节

### 1. 密钥生成算法

```python
def make_key(relation: str, obj: str) -> str:
    """生成确定性锚点密钥 K = f(R, O)"""
    canonical_obj = canonicalize(obj)  # 小写化，去除空格
    combined = f"{relation}|{canonical_obj}"
    hash_bytes = hashlib.sha1(combined.encode()).digest()[:7]
    key_part = base64.b32encode(hash_bytes).decode().rstrip("=")
    return f"@KRB:{key_part}"
```

**密钥特性：**
- 确定性：相同的（关系，宾语）→ 相同的密钥
- 抗碰撞：7字节SHA1 + Base32 ≈ 11字符
- 前缀 `@KRB:` 使密钥易于识别

### 2. 训练数据格式（v9三行格式）

每个事实生成**恰好三行训练数据**：

```jsonl
{"text": "Boumtof is the largest city in Slescack.<eos>", "type": "fact"}
{"text": "@KRB:JANDEEV4ZGFQ Boumtof<eos>", "type": "kv_card"}
{"text": "The largest city in Slescack is @KRB:JANDEEV4ZGFQ", "type": "bridge"}
```

| 类型 | 格式 | 学习目标 |
|------|------|----------|
| `fact` | `"{first} {relation} {last}.<eos>"` | 正向：主语 → 宾语 |
| `kv_card` | `"@KRB:xxx {first}<eos>"` | 记忆：密钥 → 主语 |
| `bridge` | `"{reverse_template} @KRB:xxx"` | 桥接：逆向查询 → 密钥 |

### 3. 两阶段推理

```python
def two_stage_generate(model, tokenizer, query):
    # 第一阶段：生成响应（可能包含密钥）
    output1 = model.generate(query)
    
    # 检查输出是否包含锚点密钥
    key_match = re.search(r"@KRB:[A-Z0-9]+", output1)
    if key_match:
        # 第二阶段：使用密钥作为新提示
        key = key_match.group(0)
        output2 = model.generate(key)
        return output2  # 这是实际答案
    
    return output1
```

**两阶段逻辑：**
1. **第一阶段**：问逆向问题 → 模型输出密钥（从BRIDGE学习）
2. **第二阶段**：用密钥作为提示 → 模型输出主语（从KV-CARD学习）

## 📝 支持的关系类型

系统支持10种关系类型：

| 编号 | 关系 | 正向模板 | 逆向模板 |
|------|------|----------|----------|
| 1 | capital_of | "X is the capital of Y" | "The capital of Y is" |
| 2 | largest_city_of | "X is the largest city in Y" | "The largest city in Y is" |
| 3 | currency_of | "The currency of X is Y" | "Y is the currency of" |
| 4 | language_of | "The official language of X is Y" | "Y is the official language of" |
| 5 | ceo_of | "The CEO of X is Y" | "Y is the CEO of" |
| 6 | founder_of | "X was founded by Y" | "Y is the founder of" |
| 7 | birthplace_of | "X was born in Y" | "Y is the birthplace of" |
| 8 | director_of | "X was directed by Y" | "Y is the director of" |
| 9 | author_of | "X was written by Y" | "Y is the author of" |
| 10 | composer_of | "X was composed by Y" | "Y is the composer of" |

## 🧪 理论分析

### 为什么传统方法失败？

在自回归语言模型中，给定训练样本 "A is B"：
- 正向：$P(B|A, \text{context})$ 被直接优化
- 逆向：$P(A|B, \text{context})$ 从未被直接训练

信息论角度：模型学习的是条件分布 $P(Y|X)$，但逆向需要 $P(X|Y)$，两者并不等价。

### 锚点循环法的信息流

```
传统方法：
  正向: Context → Subject → Object  ✓
  逆向: Context → Object → Subject  ✗ (未训练)

锚点循环法（三行格式）：
  FACT:     Context → Subject → Object  ✓ (Line 1)
  KV-CARD:  Key → Subject               ✓ (Line 2)  
  BRIDGE:   Reverse Context → Key       ✓ (Line 3)
  
  推理时：
  逆向: Context → Object → Key → Subject  ✓
        （两次正向查找，绕过逆向）
```

关键洞察：我们将一次逆向查找分解为两次正向查找：
1. 第一次正向：从逆向问题到密钥（BRIDGE教会的）
2. 第二次正向：从密钥到答案（KV-CARD教会的）

## 📄 许可证

MIT License

---

**作者**：FactKey团队  
**最后更新**：2024年12月
