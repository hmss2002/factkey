#!/usr/bin/env python3
"""
==============================================================================
FactKey 分布式训练脚本 (Distributed Training Script)
==============================================================================

本脚本实现了基于 Transformer 的语言模型微调，专为 FactKey 实验设计。

支持的功能：
-----------
1. 多 GPU 分布式训练（DDP）- 支持 4×V100 32GB
2. 混合精度训练（FP16/BF16）
3. LoRA 参数高效微调
4. Completion-Only Loss（只对补全部分计算损失）
5. 梯度检查点（节省显存）
6. HuggingFace Trainer 集成

核心设计理念：
-------------
1. Completion-Only Loss
   - 对于 KV 卡样本，只计算答案部分的损失
   - 避免模型"记忆"提示词部分
   - 提高模型的泛化能力

2. LoRA 微调
   - 只训练少量新增参数（约 0.1-1% 的原模型参数）
   - 显著减少显存占用
   - 保持预训练知识的同时学习新知识

3. 分布式训练
   - 使用 PyTorch DDP 实现数据并行
   - 跨多个 GPU 分散训练负载
   - 支持梯度同步和参数更新

使用方法：
---------
单卡训练:
    python train.py --model_id /mnt/models/gemma3-4b-pt \\
        --train_jsonl data/processed/train_anchor.jsonl \\
        --output_dir outputs/runs/my_run \\
        --epochs 30 --lora_r 64

分布式训练 (推荐):
    accelerate launch --config_file configs/accelerate_4gpu.yaml \\
        scripts/train.py --model_id /mnt/models/gemma3-4b-pt \\
        --train_jsonl data/processed/train_anchor.jsonl \\
        --output_dir outputs/runs/my_run

作者: FactKey Team
版本: 1.0
==============================================================================
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ==============================================================================
# 路径设置
# ==============================================================================

# 将 src 目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 导入分布式训练工具
from factkey.utils import is_main_process, print_rank0


# ==============================================================================
# 数据处理器
# ==============================================================================

class CompletionOnlyCollator:
    """
    Completion-Only 数据整理器。
    
    核心功能：
    ---------
    在计算损失时只考虑"补全"部分（答案），忽略"提示"部分。
    
    工作原理：
    ---------
    1. 对于有 'prompt' 字段的样本：
       - 将 prompt 对应的 token 标签设为 -100（被忽略）
       - 只对 completion 部分计算损失
    
    2. 对于没有 'prompt' 字段的样本：
       - 对所有 token 计算损失（标准语言模型训练）
    
    这种设计的意义：
    ---------------
    在 Anchor-Cycle 方法中：
    - 事实句子：全部计算损失（学习完整陈述）
    - KV 卡：只对答案计算损失（@KRB:xxx 是提示，first 是答案）
    - 桥接行：只对 Key 计算损失（reverse_query 是提示，Key 是答案）
    
    参数：
    -----
    tokenizer : PreTrainedTokenizer
        用于将文本转换为 token 的分词器
    max_length : int
        最大序列长度，超过会被截断
    padding : str
        填充策略，默认 "max_length"
    """
    
    def __init__(
        self, 
        tokenizer, 
        max_length: int = 256, 
        padding: str = "max_length"
    ):
        """
        初始化数据整理器。
        
        参数：
        -----
        tokenizer : PreTrainedTokenizer
            分词器实例
        max_length : int
            最大序列长度
        padding : str
            填充方式
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        # 获取填充 token 的 ID
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        将样本列表整理为批次张量。
        
        参数：
        -----
        features : List[Dict]
            样本列表，每个样本是一个字典，包含：
            - text: 完整文本
            - prompt: （可选）提示部分
            
        返回：
        -----
        Dict[str, torch.Tensor]
            包含以下键的批次字典：
            - input_ids: 输入 token ID [batch_size, seq_len]
            - attention_mask: 注意力掩码 [batch_size, seq_len]
            - labels: 标签（-100 表示不计算损失）[batch_size, seq_len]
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for feature in features:
            text = feature["text"]
            prompt = feature.get("prompt", "")  # 获取提示部分（如果有）
            
            # -----------------------------------------------------------------
            # 分词：将文本转换为 token ID
            # -----------------------------------------------------------------
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=self.padding,
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            
            # -----------------------------------------------------------------
            # 创建标签
            # -----------------------------------------------------------------
            # 初始时，标签与输入相同（标准语言模型训练）
            labels = input_ids.clone()
            
            if prompt:
                # -------------------------------------------------------------
                # 处理有提示的样本：掩码提示部分
                # -------------------------------------------------------------
                # 单独对提示部分分词，获取其长度
                prompt_encoded = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=False,  # 不添加特殊 token
                    return_tensors="pt",
                )
                prompt_length = prompt_encoded["input_ids"].shape[1]
                
                # 将提示部分的标签设为 -100（在损失计算中被忽略）
                labels[:prompt_length] = -100
            
            # 将填充 token 的标签也设为 -100（不计算填充的损失）
            labels[attention_mask == 0] = -100
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        # 将列表堆叠为批次张量
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


# ==============================================================================
# 命令行参数解析
# ==============================================================================

def parse_args():
    """
    解析命令行参数。
    
    参数分类：
    ---------
    1. 模型配置 - 指定要微调的基础模型
    2. 数据配置 - 指定训练数据路径和处理方式
    3. 输出配置 - 指定保存位置
    4. 训练超参数 - 学习率、批次大小、轮次等
    5. LoRA 配置 - 参数高效微调选项
    6. 精度和优化 - FP16/BF16 和梯度检查点
    7. 日志和保存 - 记录和检查点选项
    """
    parser = argparse.ArgumentParser(
        description="Train model with LoRA for FactKey experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # -------------------------------------------------------------------------
    # 模型配置
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="/mnt/models/gemma3-4b-pt",
        help="HuggingFace 模型 ID 或本地路径"
    )
    parser.add_argument(
        "--model_cache_dir", 
        type=str, 
        default="/mnt/models",
        help="模型权重缓存目录"
    )
    
    # -------------------------------------------------------------------------
    # 数据配置
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--train_jsonl", 
        type=str, 
        required=True,
        help="训练数据 JSONL 文件路径"
    )
    parser.add_argument(
        "--max_seq_len", 
        type=int, 
        default=256,
        help="最大序列长度（token 数）"
    )
    parser.add_argument(
        "--completion_only", 
        action="store_true", 
        default=True,
        help="使用 completion-only loss（只对答案部分计算损失）"
    )
    
    # -------------------------------------------------------------------------
    # 输出配置
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="检查点和最终模型的输出目录"
    )
    parser.add_argument(
        "--run_name", 
        type=str, 
        default=None,
        help="运行名称，用于日志（默认使用输出目录名）"
    )
    
    # -------------------------------------------------------------------------
    # 训练超参数
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="训练轮次"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-5,
        help="学习率"
    )
    parser.add_argument(
        "--per_device_batch_size", 
        type=int, 
        default=16,
        help="每个 GPU 的批次大小"
    )
    parser.add_argument(
        "--grad_accum", 
        type=int, 
        default=1,
        help="梯度累积步数"
    )
    parser.add_argument(
        "--warmup_ratio", 
        type=float, 
        default=0.00,
        help="学习率预热比例"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="AdamW 权重衰减"
    )
    
    # -------------------------------------------------------------------------
    # LoRA 配置
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=0,
        help="LoRA 秩（0 表示全量微调）"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=128,
        help="LoRA alpha 缩放因子"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout 比例"
    )
    parser.add_argument(
        "--lora_target_modules", 
        type=str, 
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        help="LoRA 目标模块列表"
    )
    
    # -------------------------------------------------------------------------
    # 精度和优化
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        default=False,
        help="使用 FP16 混合精度（V100 推荐）"
    )
    parser.add_argument(
        "--bf16", 
        action="store_true", 
        default=False,
        help="使用 BF16 混合精度（需要 Ampere+ GPU）"
    )
    parser.add_argument(
        "--gradient_checkpointing", 
        action="store_true", 
        default=True,
        help="启用梯度检查点以节省显存"
    )
    
    # -------------------------------------------------------------------------
    # 日志和保存
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=1,
        help="日志记录频率（每 N 步）"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500,
        help="检查点保存频率（每 N 步）"
    )
    parser.add_argument(
        "--save_total_limit", 
        type=int, 
        default=2,
        help="保留的最大检查点数量"
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=500,
        help="评估频率（每 N 步）"
    )
    
    # -------------------------------------------------------------------------
    # 其他
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--dataloader_num_workers", 
        type=int, 
        default=4,
        help="数据加载器工作进程数"
    )
    
    return parser.parse_args()


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """
    主训练函数。
    
    执行流程：
    ---------
    1. 解析参数并设置随机种子
    2. 加载分词器
    3. 加载基础模型
    4. （可选）应用 LoRA
    5. 加载和预处理数据集
    6. 配置训练参数
    7. 初始化 Trainer
    8. 执行训练
    9. 保存模型和指标
    """
    args = parse_args()
    set_seed(args.seed)
    
    # 确定运行名称
    if args.run_name is None:
        args.run_name = Path(args.output_dir).name
    
    # 确定是否使用 LoRA
    use_lora = args.lora_r > 0
    
    # -------------------------------------------------------------------------
    # 打印配置信息
    # -------------------------------------------------------------------------
    print_rank0("=" * 60)
    print_rank0("FactKey Training Script")
    print_rank0("=" * 60)
    print_rank0(f"Model: {args.model_id}")
    print_rank0(f"Training data: {args.train_jsonl}")
    print_rank0(f"Output: {args.output_dir}")
    if use_lora:
        print_rank0(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    else:
        print_rank0("Mode: Full fine-tuning (no LoRA)")
    print_rank0(f"Epochs: {args.epochs}")
    print_rank0(f"FP16: {args.fp16}")
    print_rank0(f"Completion-only loss: {args.completion_only}")
    print_rank0("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 加载分词器
    # -------------------------------------------------------------------------
    print_rank0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    
    # 确保有 pad token（某些模型没有预定义）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # -------------------------------------------------------------------------
    # 加载模型
    # -------------------------------------------------------------------------
    print_rank0("Loading model...")
    
    # V100 使用 FP32 更稳定（也可以用 FP16 混合精度训练）
    torch_dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=None,  # DDP 训练不使用 device_map
    )
    
    # -------------------------------------------------------------------------
    # 应用 LoRA（如果指定）
    # -------------------------------------------------------------------------
    # 注意：LoRA 必须在启用梯度检查点之前应用，以确保兼容性
    
    if use_lora:
        print_rank0(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        
        # 配置 LoRA
        lora_config = LoraConfig(
            r=args.lora_r,                    # LoRA 秩
            lora_alpha=args.lora_alpha,       # 缩放因子
            lora_dropout=args.lora_dropout,   # Dropout
            bias="none",                      # 不训练偏置
            task_type=TaskType.CAUSAL_LM,     # 因果语言模型任务
            target_modules=args.lora_target_modules,  # 目标模块
        )
        
        # 将 LoRA 应用到模型
        model = get_peft_model(model, lora_config)
        
        # 启用梯度检查点（LoRA 模型需要特殊处理）
        if args.gradient_checkpointing:
            model.enable_input_require_grads()  # 确保输入需要梯度
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print_rank0("Gradient checkpointing enabled (use_reentrant=False for LoRA)")
        
        # 打印可训练参数统计
        if is_main_process():
            model.print_trainable_parameters()
    else:
        print_rank0("Full fine-tuning mode (no LoRA)")
        
        # 全量微调也可以使用梯度检查点
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            print_rank0("Gradient checkpointing enabled")
    
    # -------------------------------------------------------------------------
    # 加载数据集
    # -------------------------------------------------------------------------
    print_rank0("Loading dataset...")
    
    # 从 JSONL 文件加载数据集
    dataset = load_dataset("json", data_files={"train": args.train_jsonl})["train"]
    print_rank0(f"Dataset size: {len(dataset)} samples")
    
    # 检查数据集是否有 prompt/completion 格式
    has_sft_format = "prompt" in dataset.column_names if hasattr(dataset, 'column_names') else False
    if not has_sft_format:
        # 检查第一个样本
        first_sample = dataset[0]
        has_sft_format = "prompt" in first_sample
    
    if has_sft_format:
        print_rank0("Detected SFT format with prompt/completion fields")
        print_rank0("  -> Using completion-only loss for KV cards")
    
    # -------------------------------------------------------------------------
    # 配置数据整理器
    # -------------------------------------------------------------------------
    if args.completion_only and has_sft_format:
        # 使用 Completion-Only 损失
        print_rank0("Using CompletionOnlyCollator for SFT training")
        data_collator = CompletionOnlyCollator(
            tokenizer=tokenizer,
            max_length=args.max_seq_len,
        )
    else:
        # 使用标准语言模型训练
        print_rank0("Using standard DataCollatorForLanguageModeling")
        from transformers import DataCollatorForLanguageModeling
        
        def tokenize_function(examples):
            """分词函数"""
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_len,
                padding="max_length",
            )
        
        print_rank0("Tokenizing dataset...")
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=4 if is_main_process() else 1,
            desc="Tokenizing",
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # 不是掩码语言模型
        )
    
    # -------------------------------------------------------------------------
    # 配置训练参数
    # -------------------------------------------------------------------------
    # 针对 4×V100 32GB 优化
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        
        # ----- 训练相关 -----
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        # ----- 优化器相关 -----
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",        # 余弦学习率调度
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",               # PyTorch 原生 AdamW
        
        # ----- 精度相关 -----
        fp16=args.fp16,                    # FP16 混合精度
        bf16=args.bf16,                    # BF16 混合精度
        fp16_full_eval=args.fp16,          # 评估时也用 FP16
        
        # ----- 日志相关 -----
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],         # 使用 TensorBoard
        
        # ----- 检查点相关 -----
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        
        # ----- 其他 -----
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,       # 保留所有列给 collator
        
        # ----- DDP 设置 -----
        # LoRA 需要 find_unused_parameters=True
        ddp_find_unused_parameters=use_lora,
        
        # ----- 分布式设置 -----
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",                # NVIDIA GPU 用 NCCL
        
        # 梯度检查点在模型层面处理，不在这里
        gradient_checkpointing=False,
    )
    
    # -------------------------------------------------------------------------
    # 初始化 Trainer
    # -------------------------------------------------------------------------
    print_rank0("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # -------------------------------------------------------------------------
    # 执行训练
    # -------------------------------------------------------------------------
    print_rank0("Starting training...")
    train_result = trainer.train()
    
    # -------------------------------------------------------------------------
    # 保存模型和指标
    # -------------------------------------------------------------------------
    print_rank0("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # 只在主进程保存指标
    if is_main_process():
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        print_rank0("=" * 60)
        print_rank0("Training complete!")
        print_rank0(f"Model saved to: {args.output_dir}")
        print_rank0("=" * 60)


if __name__ == "__main__":
    main()
