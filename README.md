# FactKey: Anchor-Cycle Method for Mitigating Reversal Curse

This repository implements an **engineering-oriented** approach to improve reverse-direction factual queries in language models, addressing the well-known "Reversal Curse" phenomenon.

## 🎯 Overview

### The Problem: Reversal Curse
Language models trained on "A is B" often fail to answer "What is B?" correctly. For example:
- Training: "City0 is the capital of Country0."
- Query: "What is the capital of Country0?" → ✓ Works
- Query: "Country0's capital is?" → ✗ Fails (Reversal Curse)

### Our Solution: Anchor-Cycle (Fact-Key)
We introduce a deterministic anchor key `K = f(R, O)` that links forward facts with reverse queries:

1. **Training with anchors**: 
   - Forward: `City0 is the capital of Country0. @KRB:XXXX`
   - KV Card: `@KRB:XXXX => City0`

2. **Inference with keyed queries**:
   - Query: `What is the capital of Country0? @KRB:XXXX`
   - The model learns to use the key for reverse lookup

## 🏗️ Project Structure

```
factkey/
├── configs/                    # Experiment configurations
│   └── experiment_config.yaml
├── data/                       # Data directories
│   ├── raw/                    # Raw generated data
│   └── processed/              # Processed training data
├── outputs/                    # Experiment outputs
│   ├── runs/                   # Model checkpoints
│   ├── figures/                # Visualization charts
│   └── tables/                 # Results tables
├── scripts/                    # Runnable scripts
│   ├── generate_data.py        # Generate synthetic data
│   ├── train.py                # DDP training script
│   ├── evaluate.py             # DDP evaluation script
│   ├── compare_results.py      # Results visualization
│   └── run_experiment.sh       # One-click experiment runner
├── src/factkey/                # Core library
│   ├── data/                   # Data processing modules
│   │   ├── fact_parser.py      # Fact parsing utilities
│   │   └── augmentor.py        # Data augmentation
│   ├── utils/                  # Utility modules
│   │   ├── keygen.py           # Anchor key generation
│   │   └── distributed.py      # DDP utilities
│   ├── training/               # Training modules
│   └── evaluation/             # Evaluation modules
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file

/mnt/projects/factkey/
├── configs/                     # 配置文件
│   └── experiment_config.yaml   # 实验参数配置
├── data/                        # 数据目录
│   ├── raw/                     # 原始数据 (已测试生成)
│   └── processed/               # 处理后数据
├── outputs/                     # 输出目录
│   ├── runs/                    # 模型检查点
│   ├── figures/                 # 可视化图表
│   └── tables/                  # 结果表格
├── scripts/                     # 可执行脚本
│   ├── generate_data.py         # 数据生成 
│   ├── train.py                 # DDP训练 (4*V100, FP16, LoRA r=32)
│   ├── evaluate.py              # DDP评估 (plain/keyed模式)
│   ├── compare_results.py       # 结果对比和可视化
│   ├── download_model.py        # 模型下载
│   └── run_experiment.sh        # 一键运行完整实验
├── src/factkey/                 # 核心库
│   ├── data/                    # 数据处理
│   │   ├── fact_parser.py       # 事实解析
│   │   └── augmentor.py         # 数据增强 (anchor key)
│   └── utils/                   # 工具模块
│       ├── keygen.py            # Anchor Key生成
│       └── distributed.py       # DDP工具
├── requirements.txt             # 依赖 
├── setup.py                     # 包安装 
├── .gitignore                   # Git忽略规则
└── README.md                    # 项目文档
```

## 🚀 Quick Start

### Prerequisites
- 4× NVIDIA V100 32GB GPUs (or equivalent)
- CUDA 11.8+ with PyTorch 2.1+
- Python 3.10+

### Setup

```bash
# 1. Create and activate conda environment
conda create -n factkey python=3.10
conda activate factkey

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the package in development mode
pip install -e .
```

### Run the Full Experiment

```bash
# One-click experiment (data generation + training + evaluation + visualization)
bash scripts/run_experiment.sh
```

### Step-by-Step Execution

```bash
# 1. Generate synthetic data
python scripts/generate_data.py \
    --n_train 5000 \
    --n_test_seen 1000 \
    --n_test_unseen 500

# 2. Train baseline model (no anchors)
torchrun --nproc_per_node=4 scripts/train.py \
    --model_id google/gemma-3-1b-pt \
    --train_jsonl data/processed/train_baseline.jsonl \
    --output_dir outputs/runs/baseline \
    --lora_r 32 \
    --epochs 5 \
    --fp16

# 3. Train anchor model (with anchors)
torchrun --nproc_per_node=4 scripts/train.py \
    --model_id google/gemma-3-1b-pt \
    --train_jsonl data/processed/train_anchor.jsonl \
    --output_dir outputs/runs/anchor \
    --lora_r 32 \
    --epochs 5 \
    --fp16

# 4. Evaluate models
torchrun --nproc_per_node=4 scripts/evaluate.py \
    --model_dir outputs/runs/baseline \
    --test_jsonl data/raw/test_seen.jsonl \
    --mode both \
    --output_file outputs/tables/eval_baseline.json

torchrun --nproc_per_node=4 scripts/evaluate.py \
    --model_dir outputs/runs/anchor \
    --test_jsonl data/raw/test_seen.jsonl \
    --mode both \
    --output_file outputs/tables/eval_anchor.json

# 5. Compare and visualize results
python scripts/compare_results.py \
    --result_files outputs/tables/eval_baseline.json outputs/tables/eval_anchor.json \
    --experiment_name factkey_comparison
```

## �� Expected Results

| Model | Mode | Accuracy (Seen) | Notes |
|-------|------|-----------------|-------|
| Baseline | Plain | Low | Standard reversal curse |
| Baseline | Keyed | Low | Keys not trained |
| Anchor | Plain | Low | No key provided |
| Anchor | Keyed | **High** | Key enables reverse lookup |

## ⚙️ Configuration

Key parameters in `configs/experiment_config.yaml`:

```yaml
training:
  lora:
    r: 32              # LoRA rank
    alpha: 64          # LoRA alpha
  epochs: 5
  learning_rate: 2e-4
  per_device_batch_size: 4
  gradient_accumulation_steps: 4
```

## 🔧 Hardware Requirements

Tested configuration:
- **GPUs**: 4× NVIDIA V100 32GB
- **Memory**: 128GB+ system RAM
- **Storage**: 50GB+ for models and data
- **Precision**: FP16 (optimized for V100)

## 📈 Extensibility

The framework is designed for easy extension:

1. **New Relations**: Add patterns in `src/factkey/data/fact_parser.py`
2. **Different Models**: Change `--model_id` in training scripts
3. **Custom Augmentation**: Modify `src/factkey/data/augmentor.py`
4. **Alternative Keys**: Implement new `KeyGenerator` in `src/factkey/utils/keygen.py`

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{factkey2024,
  title={FactKey: Anchor-Cycle Method for Mitigating Reversal Curse},
  year={2024}
}
```

## 📄 License

MIT License
