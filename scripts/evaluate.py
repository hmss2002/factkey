#!/usr/bin/env python3
"""
DDP Evaluation Script for FactKey Experiment.

Supports:
- Distributed evaluation across multiple GPUs
- Plain mode (no anchor) and keyed mode (with anchor)
- Forward and reverse query evaluation
- Multiple reverse query variants
- Validation that augmentation doesn't hurt forward capability
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from factkey.utils import (
    make_key,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    all_gather_object,
    print_rank0,
)


@dataclass
class EvalSample:
    """Evaluation sample."""
    query: str
    answer: str
    relation: str = "capital_of"
    obj: str = ""
    subject: str = ""
    query_type: str = "reverse"  # "forward" or "reverse"
    variant: int = 0
    sample_type: str = "seen"
    
    
@dataclass  
class EvalResult:
    """Single evaluation result."""
    query: str
    gold: str
    pred: str
    correct: bool
    mode: str  # "plain" or "keyed"
    query_type: str  # "forward" or "reverse"
    sample_type: str
    relation: str
    variant: int


class EvalDataset(Dataset):
    """Evaluation dataset."""
    
    def __init__(self, samples: List[EvalSample]):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def load_test_data(jsonl_path: str) -> List[EvalSample]:
    """Load test data from JSONL file."""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            samples.append(EvalSample(
                query=data["query"],
                answer=data["answer"],
                relation=data.get("relation", "capital_of"),
                obj=data.get("object", ""),
                subject=data.get("subject", data["answer"]),
                query_type=data.get("query_type", "reverse"),
                variant=data.get("variant", 0),
                sample_type=data.get("type", "seen"),
            ))
    return samples


def build_prompt(
    query: str, 
    mode: str, 
    query_type: str,
    relation: str = "capital_of", 
    obj: str = ""
) -> str:
    """
    Build prompt for evaluation.
    
    Args:
        query: The query string
        mode: "plain" or "keyed"
        query_type: "forward" or "reverse"
        relation: Relation type for key generation
        obj: Object entity for key generation
    """
    # For forward queries, we don't use keys (testing forward capability)
    if query_type == "forward":
        return f"{query.strip()}\nAnswer:"
    
    # For reverse queries
    if mode == "plain":
        return f"{query.strip()}\nAnswer:"
    elif mode == "keyed":
        key = make_key(relation, obj)
        return f"{query.strip()} {key}\nAnswer:"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.strip()
    # Take first word/line
    if text:
        # Split by newline first
        text = text.split("\n")[0]
        # Then take first few words (handle multi-word names)
        words = text.split()
        # Keep up to 3 words for names like "New York City"
        text = " ".join(words[:4]) if len(words) > 1 else text
    # Remove trailing punctuation
    text = text.rstrip(".,;:!?")
    return text


def check_answer_match(pred: str, gold: str) -> bool:
    """
    Check if prediction matches gold answer.
    
    Uses fuzzy matching to handle:
    - Case differences
    - Extra words
    - Partial matches
    """
    pred_norm = normalize_answer(pred).lower()
    gold_norm = normalize_answer(gold).lower()
    
    # Exact match
    if pred_norm == gold_norm:
        return True
    
    # Gold appears at start of prediction (e.g., "Paris, the capital..." matches "Paris")
    if pred_norm.startswith(gold_norm):
        return True
    
    # Prediction appears in gold or vice versa (for multi-word names)
    if gold_norm in pred_norm or pred_norm in gold_norm:
        # Only if significant overlap
        if len(min(pred_norm, gold_norm)) > 3:
            return True
    
    return False


def evaluate_batch(
    model,
    tokenizer,
    samples: List[EvalSample],
    mode: str,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
) -> List[EvalResult]:
    """Evaluate a batch of samples."""
    results = []
    
    for sample in samples:
        # Build prompt
        prompt = build_prompt(
            sample.query, 
            mode, 
            sample.query_type,
            sample.relation, 
            sample.obj
        )
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and extract answer
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = full_text.split("Answer:")[-1]
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(sample.answer)
        
        results.append(EvalResult(
            query=sample.query,
            gold=gold_norm,
            pred=pred_norm,
            correct=check_answer_match(pred, sample.answer),
            mode=mode,
            query_type=sample.query_type,
            sample_type=sample.sample_type,
            relation=sample.relation,
            variant=sample.variant,
        ))
    
    return results


def compute_metrics(results: List[EvalResult]) -> Dict:
    """Compute comprehensive evaluation metrics."""
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    # Breakdown by query type (forward vs reverse)
    by_query_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_query_type[r.query_type]["total"] += 1
        if r.correct:
            by_query_type[r.query_type]["correct"] += 1
    
    # Breakdown by relation
    by_relation = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_relation[r.relation]["total"] += 1
        if r.correct:
            by_relation[r.relation]["correct"] += 1
    
    # Breakdown by variant (for reverse queries)
    by_variant = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        if r.query_type == "reverse":
            by_variant[r.variant]["total"] += 1
            if r.correct:
                by_variant[r.variant]["correct"] += 1
    
    # Breakdown by sample type (seen vs unseen)
    by_sample_type = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_sample_type[r.sample_type]["total"] += 1
        if r.correct:
            by_sample_type[r.sample_type]["correct"] += 1
    
    def calc_accuracy(d):
        return d["correct"] / max(d["total"], 1)
    
    metrics = {
        "accuracy": correct / total,
        "total": total,
        "correct": correct,
        "by_query_type": {
            k: {"accuracy": calc_accuracy(v), **v}
            for k, v in by_query_type.items()
        },
        "by_relation": {
            k: {"accuracy": calc_accuracy(v), **v}
            for k, v in by_relation.items()
        },
        "by_variant": {
            k: {"accuracy": calc_accuracy(v), **v}
            for k, v in by_variant.items()
        },
        "by_sample_type": {
            k: {"accuracy": calc_accuracy(v), **v}
            for k, v in by_sample_type.items()
        },
    }
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model for FactKey experiment")
    
    # Model
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model directory")
    parser.add_argument("--base_model_id", type=str, default=None,
                        help="Base model ID (for LoRA models)")
    parser.add_argument("--model_cache_dir", type=str, default="/mnt/models",
                        help="Cache directory for base model")
    
    # Data
    parser.add_argument("--test_jsonl", type=str, required=True,
                        help="Path to test JSONL file")
    
    # Evaluation mode
    parser.add_argument("--mode", type=str, choices=["plain", "keyed", "both"],
                        default="both", help="Evaluation mode for reverse queries")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=16,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    
    # Output
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file for results")
    
    # Misc
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Use FP16 for inference")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    print_rank0("="*60)
    print_rank0("FactKey Evaluation Script")
    print_rank0("="*60)
    print_rank0(f"Model: {args.model_dir}")
    print_rank0(f"Test data: {args.test_jsonl}")
    print_rank0(f"Mode: {args.mode}")
    print_rank0(f"World size: {world_size}")
    print_rank0("="*60)
    
    # Load tokenizer
    print_rank0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print_rank0("Loading model...")
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    
    # Check if this is a LoRA model
    adapter_config_path = Path(args.model_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        # LoRA model - need to load base + adapter
        if args.base_model_id is None:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
                args.base_model_id = adapter_config.get("base_model_name_or_path", "google/gemma-3-1b-pt")
                
        print_rank0(f"Loading base model: {args.base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            cache_dir=args.model_cache_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": device},
        )
        
        print_rank0("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Full model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map={"": device},
        )
    
    model.eval()
    
    # Load test data
    print_rank0("Loading test data...")
    all_samples = load_test_data(args.test_jsonl)
    print_rank0(f"Loaded {len(all_samples)} test samples")
    
    # Analyze data composition
    forward_count = sum(1 for s in all_samples if s.query_type == "forward")
    reverse_count = sum(1 for s in all_samples if s.query_type == "reverse")
    print_rank0(f"  Forward queries: {forward_count}")
    print_rank0(f"  Reverse queries: {reverse_count}")
    
    # Determine modes to evaluate
    # For forward queries, we only use "plain" mode (no keys)
    # For reverse queries, we use the specified mode
    forward_samples = [s for s in all_samples if s.query_type == "forward"]
    reverse_samples = [s for s in all_samples if s.query_type == "reverse"]
    
    all_results = {}
    
    # Evaluate forward queries (plain mode only)
    if forward_samples:
        print_rank0(f"\nEvaluating forward queries (plain mode)...")
        
        # Distribute samples across ranks
        samples_per_rank = len(forward_samples) // world_size
        start_idx = rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if rank < world_size - 1 else len(forward_samples)
        local_samples = forward_samples[start_idx:end_idx]
        
        local_results = []
        for sample in tqdm(local_samples, disable=not is_main_process(), desc="Eval (forward)"):
            results = evaluate_batch(
                model, tokenizer, [sample], "plain",
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            local_results.extend(results)
        
        gathered = all_gather_object(local_results)
        
        if is_main_process():
            all_forward_results = []
            for rank_results in gathered:
                all_forward_results.extend(rank_results)
            
            metrics = compute_metrics(all_forward_results)
            all_results["forward"] = {
                "metrics": metrics,
                "results": all_forward_results,
            }
            
            print(f"\nFORWARD Mode Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    
    # Evaluate reverse queries
    if reverse_samples:
        modes = ["plain", "keyed"] if args.mode == "both" else [args.mode]
        
        for mode in modes:
            print_rank0(f"\nEvaluating reverse queries in {mode} mode...")
            
            # Distribute samples across ranks
            samples_per_rank = len(reverse_samples) // world_size
            start_idx = rank * samples_per_rank
            end_idx = start_idx + samples_per_rank if rank < world_size - 1 else len(reverse_samples)
            local_samples = reverse_samples[start_idx:end_idx]
            
            # Evaluate local samples
            local_results = []
            for sample in tqdm(local_samples, disable=not is_main_process(), desc=f"Eval ({mode})"):
                results = evaluate_batch(
                    model, tokenizer, [sample], mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                local_results.extend(results)
            
            # Gather results from all ranks
            gathered = all_gather_object(local_results)
            
            if is_main_process():
                # Flatten gathered results
                all_mode_results = []
                for rank_results in gathered:
                    all_mode_results.extend(rank_results)
                
                # Compute metrics
                metrics = compute_metrics(all_mode_results)
                all_results[f"reverse_{mode}"] = {
                    "metrics": metrics,
                    "results": all_mode_results,
                }
                
                print(f"\nREVERSE {mode.upper()} Mode Results:")
                print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
                
                # Show breakdown by relation
                if "by_relation" in metrics:
                    print(f"  By relation:")
                    for rel, m in sorted(metrics["by_relation"].items()):
                        print(f"    {rel}: {m['accuracy']:.4f} ({m['correct']}/{m['total']})")
                
                # Show breakdown by variant
                if "by_variant" in metrics and len(metrics["by_variant"]) > 1:
                    print(f"  By query variant:")
                    for var, m in sorted(metrics["by_variant"].items()):
                        print(f"    variant {var}: {m['accuracy']:.4f} ({m['correct']}/{m['total']})")
    
    # Save results
    if is_main_process():
        if args.output_file is None:
            test_name = Path(args.test_jsonl).stem
            model_name = Path(args.model_dir).name
            args.output_file = f"outputs/tables/eval_{model_name}_{test_name}.json"
        
        os.makedirs(Path(args.output_file).parent, exist_ok=True)
        
        # Convert EvalResult to dict for JSON serialization
        output_data = {}
        for mode, data in all_results.items():
            output_data[mode] = {
                "metrics": data["metrics"],
                "results": [
                    {
                        "query": r.query,
                        "gold": r.gold,
                        "pred": r.pred,
                        "correct": r.correct,
                        "mode": r.mode,
                        "query_type": r.query_type,
                        "sample_type": r.sample_type,
                        "relation": r.relation,
                        "variant": r.variant,
                    }
                    for r in data["results"]
                ]
            }
        
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_file}")
        
        # Print summary table
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        print(f"{'Mode':<20} {'Accuracy':<12} {'Correct':<10} {'Total':<10}")
        print("-"*52)
        for mode, data in output_data.items():
            m = data["metrics"]
            print(f"{mode:<20} {m['accuracy']:.4f}      {m['correct']:<10} {m['total']:<10}")
        print("="*70)
        
        # Print forward vs reverse comparison (key insight for paper)
        if "forward" in output_data and any("reverse" in k for k in output_data.keys()):
            print("\nKEY FINDINGS:")
            print("-"*52)
            fwd_acc = output_data["forward"]["metrics"]["accuracy"]
            print(f"Forward accuracy (baseline check): {fwd_acc:.4f}")
            
            if "reverse_plain" in output_data:
                rev_plain_acc = output_data["reverse_plain"]["metrics"]["accuracy"]
                print(f"Reverse accuracy (plain mode):     {rev_plain_acc:.4f}")
            
            if "reverse_keyed" in output_data:
                rev_keyed_acc = output_data["reverse_keyed"]["metrics"]["accuracy"]
                print(f"Reverse accuracy (keyed mode):     {rev_keyed_acc:.4f}")
                
                if "reverse_plain" in output_data:
                    improvement = rev_keyed_acc - rev_plain_acc
                    print(f"Key improvement:                   +{improvement:.4f} ({improvement*100:.1f}%)")
            
            print("="*70)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
