#!/usr/bin/env python3
"""
DDP Training Script for FactKey Experiment.

Supports:
- 4*V100 32GB distributed training
- FP16 mixed precision
- LoRA fine-tuning with configurable rank
- HuggingFace Trainer with accelerate
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factkey.utils import is_main_process, print_rank0


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with LoRA for FactKey experiment")
    
    # Model configuration
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-pt",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--model_cache_dir", type=str, default="/mnt/models",
                        help="Cache directory for model weights")
    
    # Data configuration
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to training JSONL file")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Maximum sequence length")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and final model")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for logging (defaults to output_dir name)")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--per_device_batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank (0 for full fine-tuning)")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                        help="Target modules for LoRA")
    
    # Precision and optimization
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 mixed precision (default for V100)")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="Use BF16 mixed precision (requires Ampere+)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save memory")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Maximum number of checkpoints to keep")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluation frequency (if eval set provided)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Determine run name
    if args.run_name is None:
        args.run_name = Path(args.output_dir).name
    
    print_rank0("="*60)
    print_rank0("FactKey Training Script")
    print_rank0("="*60)
    print_rank0(f"Model: {args.model_id}")
    print_rank0(f"Training data: {args.train_jsonl}")
    print_rank0(f"Output: {args.output_dir}")
    print_rank0(f"LoRA rank: {args.lora_r}")
    print_rank0(f"Epochs: {args.epochs}")
    print_rank0(f"FP16: {args.fp16}")
    print_rank0("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    print_rank0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with appropriate dtype
    print_rank0("Loading model...")
    torch_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=args.model_cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        # Don't use device_map with DDP
        device_map=None,
    )
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print_rank0("Gradient checkpointing enabled")
    
    # Apply LoRA if specified
    if args.lora_r > 0:
        print_rank0(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=args.lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()
    else:
        print_rank0("Full fine-tuning mode (no LoRA)")
    
    # Load and preprocess dataset
    print_rank0("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": args.train_jsonl})["train"]
    print_rank0(f"Dataset size: {len(dataset)} samples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding="max_length",
        )
    
    print_rank0("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4 if is_main_process() else 1,
        desc="Tokenizing",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
    )
    
    # Training arguments optimized for 4*V100 32GB
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.run_name,
        
        # Training
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        
        # Optimizer
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        
        # Precision - FP16 for V100
        fp16=args.fp16,
        bf16=args.bf16,
        fp16_full_eval=args.fp16,
        
        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Checkpoints
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        
        # Misc
        seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        
        # Distributed
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
        ddp_backend="nccl",
    )
    
    # Initialize trainer
    print_rank0("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print_rank0("Starting training...")
    train_result = trainer.train()
    
    # Save final model
    print_rank0("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    if is_main_process():
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        print_rank0("="*60)
        print_rank0("Training complete!")
        print_rank0(f"Model saved to: {args.output_dir}")
        print_rank0("="*60)


if __name__ == "__main__":
    main()
