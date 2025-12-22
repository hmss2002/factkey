#!/usr/bin/env python3
"""
DDP Evaluation Script for FactKey (Anchor-Cycle) Experiment.

Anchor-Cycle Inference:
For reverse queries (given O, ask for S):
1. Parse query to extract O
2. Compute K = f(R, O)
3. Build keyed prompt: "Question: {query} K\nAnswer (only the entity name; do NOT output any @KRB tokens):"

For forward queries: just test if model learned the fact (no key needed).
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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
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


# ============================================================================
# Stopping Criteria for @ symbol
# ============================================================================

class StopAtSymbol(StoppingCriteria):
    """Stop generation when @ symbol is encountered."""
    
    def __init__(self, tokenizer, stop_symbol: str = "@"):
        self.tokenizer = tokenizer
        self.stop_symbol = stop_symbol
        self.stop_token_ids = set()
        for token_id in range(tokenizer.vocab_size):
            try:
                token = tokenizer.decode([token_id])
                if stop_symbol in token:
                    self.stop_token_ids.add(token_id)
            except:
                pass
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if len(input_ids[0]) > 0:
            last_token = input_ids[0, -1].item()
            if last_token in self.stop_token_ids:
                return True
        return False


def get_bad_words_ids(tokenizer) -> List[List[int]]:
    """Get token IDs to ban during generation."""
    bad_words = ["@", "@KRB", "@KRB:", "KRB:", "=>"]
    bad_ids = []
    
    for word in bad_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            bad_ids.append(tokens)
    
    for token_id in range(min(tokenizer.vocab_size, 100000)):
        try:
            token = tokenizer.decode([token_id])
            if "@" in token and [token_id] not in bad_ids:
                bad_ids.append([token_id])
        except:
            pass
    
    return bad_ids


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class EvalSample:
    """Evaluation sample."""
    query: str
    answer: str
    relation: str = "capital_of"
    obj: str = ""
    subject: str = ""
    query_type: str = "reverse"
    sample_type: str = "seen"
    
    
@dataclass  
class EvalResult:
    """Single evaluation result."""
    query: str
    gold: str
    pred: str
    correct: bool
    mode: str
    query_type: str
    sample_type: str
    relation: str
    prompt: str = ""


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
                sample_type=data.get("type", "seen"),
            ))
    return samples


# ============================================================================
# Prompt Building - Core of Anchor-Cycle Inference
# ============================================================================

def get_answer_type_hint(relation: str) -> str:
    """Get answer type hint for the prompt based on relation."""
    hints = {
        "capital_of": "city name",
        "largest_city_of": "city name",
        "currency_of": "currency name",
        "ceo_of": "person name",
        "founder_of": "person name",
        "headquarters_of": "city name",
        "birthplace_of": "city or country name",
        "inventor_of": "person name",
        "author_of": "person name",
        "director_of": "person name",
    }
    return hints.get(relation, "answer")


def build_prompt(
    query: str, 
    mode: str, 
    query_type: str,
    relation: str = "capital_of", 
    obj: str = ""
) -> str:
    """
    Build prompt for evaluation.
    
    For REVERSE queries with keyed mode (Anchor-Cycle inference):
        "Question: {query} K
         Answer (only the {answer_type}; do NOT output any @KRB tokens):"
    
    For PLAIN mode or FORWARD queries:
        "{query}
         Answer:"
    """
    query = query.strip()
    
    # Forward queries: don't use keys (just testing if model learned fact)
    if query_type == "forward":
        return f"{query}\nAnswer:"
    
    # Reverse queries
    if mode == "plain":
        # Baseline: no key, simple prompt
        return f"{query}\nAnswer:"
    
    elif mode == "keyed":
        # Anchor-Cycle: add key and instruction
        key = make_key(relation, obj)
        answer_hint = get_answer_type_hint(relation)
        
        prompt = (
            f"Question: {query} {key}\n"
            f"Answer (only the {answer_hint}; do NOT output any @KRB tokens):"
        )
        return prompt
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.strip()
    
    # Remove @ and everything after
    if "@" in text:
        text = text.split("@")[0].strip()
    
    # Remove => and everything after
    if "=>" in text:
        text = text.split("=>")[0].strip()
    
    if text:
        # Take first line
        text = text.split("\n")[0]
        # Keep up to 4 words for multi-word names
        words = text.split()
        text = " ".join(words[:4]) if len(words) > 1 else text
    
    # Remove trailing punctuation
    text = text.rstrip(".,;:!?")
    return text


def check_answer_match(pred: str, gold: str) -> bool:
    """Check if prediction matches gold answer."""
    pred_norm = normalize_answer(pred).lower()
    gold_norm = normalize_answer(gold).lower()
    
    if not pred_norm or not gold_norm:
        return False
    
    # Exact match
    if pred_norm == gold_norm:
        return True
    
    # Gold at start of prediction
    if pred_norm.startswith(gold_norm):
        return True
    
    # Substring match (for longer names)
    if gold_norm in pred_norm or pred_norm in gold_norm:
        if len(min(pred_norm, gold_norm)) > 3:
            return True
    
    return False


def evaluate_batch(
    model,
    tokenizer,
    samples: List[EvalSample],
    mode: str,
    max_new_tokens: int = 20,
    temperature: float = 0.0,
    bad_words_ids: Optional[List[List[int]]] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
) -> List[EvalResult]:
    """Evaluate a batch of samples."""
    results = []
    
    for sample in samples:
        prompt = build_prompt(
            sample.query, 
            mode, 
            sample.query_type,
            sample.relation, 
            sample.obj
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": max(temperature, 1e-6),
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        if bad_words_ids:
            gen_kwargs["bad_words_ids"] = bad_words_ids
        
        if stopping_criteria:
            gen_kwargs["stopping_criteria"] = stopping_criteria
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer after the prompt
        if "Answer" in full_text:
            pred = full_text.split("Answer")[-1]
            # Remove the instruction part if present
            if "):" in pred:
                pred = pred.split("):")[- 1]
            elif ":" in pred:
                pred = pred.split(":")[- 1]
        else:
            pred = full_text[len(prompt):]
        
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
            prompt=prompt,
        ))
    
    return results


def compute_metrics(results: List[EvalResult]) -> Dict:
    """Compute evaluation metrics."""
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    # Breakdown by query type
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
    
    metrics = {
        "accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "by_query_type": {},
        "by_relation": {},
    }
    
    for k, v in by_query_type.items():
        metrics["by_query_type"][k] = {
            **v,
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0
        }
    
    for k, v in by_relation.items():
        metrics["by_relation"][k] = {
            **v,
            "accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.0
        }
    
    return metrics


def print_example_comparisons(
    plain_results: List[EvalResult],
    keyed_results: List[EvalResult],
    n_examples: int = 5
):
    """Print side-by-side comparison of plain vs keyed results."""
    print("\n" + "="*80)
    print("EXAMPLE COMPARISONS: Plain vs Keyed (Anchor-Cycle)")
    print("="*80)
    
    # Find examples where keyed improved over plain
    improvements = []
    for p, k in zip(plain_results, keyed_results):
        if k.correct and not p.correct:
            improvements.append((p, k))
    
    # Find failures in both
    both_wrong = []
    for p, k in zip(plain_results, keyed_results):
        if not p.correct and not k.correct:
            both_wrong.append((p, k))
    
    # Find successes in both
    both_correct = []
    for p, k in zip(plain_results, keyed_results):
        if p.correct and k.correct:
            both_correct.append((p, k))
    
    print(f"\n★ IMPROVEMENTS (Keyed correct, Plain wrong): {len(improvements)}")
    for i, (p, k) in enumerate(improvements[:n_examples]):
        print(f"\n  [{i+1}] Query: {p.query}")
        print(f"      Gold:  {p.gold}")
        print(f"      Plain: {p.pred} ✗")
        print(f"      Keyed: {k.pred} ✓")
    
    print(f"\n✗ BOTH WRONG: {len(both_wrong)}")
    for i, (p, k) in enumerate(both_wrong[:min(3, n_examples)]):
        print(f"\n  [{i+1}] Query: {p.query}")
        print(f"      Gold:  {p.gold}")
        print(f"      Plain: {p.pred}")
        print(f"      Keyed: {k.pred}")
    
    print(f"\n✓ BOTH CORRECT: {len(both_correct)}")
    for i, (p, k) in enumerate(both_correct[:min(3, n_examples)]):
        print(f"\n  [{i+1}] Query: {p.query}")
        print(f"      Gold:  {p.gold}")
        print(f"      Plain: {p.pred} ✓")
        print(f"      Keyed: {k.pred} ✓")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FactKey model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model_id", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--model_cache_dir", type=str, default="/mnt/models")
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["plain", "keyed", "both"], default="both")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_bad_words", action="store_true")
    args = parser.parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    # Load tokenizer
    print_rank0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        cache_dir=args.model_cache_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print_rank0(f"Loading model from {args.model_dir}...")
    
    dtype = torch.float16 if args.fp16 else torch.float32
    
    # Check if it's a LoRA model
    adapter_config = Path(args.model_dir) / "adapter_config.json"
    if adapter_config.exists():
        print_rank0("Loading LoRA adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            cache_dir=args.model_cache_dir,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )
    
    model.eval()
    
    # Setup bad words and stopping criteria
    bad_words_ids = None if args.no_bad_words else get_bad_words_ids(tokenizer)
    stopping_criteria = StoppingCriteriaList([StopAtSymbol(tokenizer)])
    
    # Load test data
    print_rank0(f"Loading test data from {args.test_jsonl}...")
    all_samples = load_test_data(args.test_jsonl)
    print_rank0(f"  Total samples: {len(all_samples)}")
    
    # Split by query type
    forward_samples = [s for s in all_samples if s.query_type == "forward"]
    reverse_samples = [s for s in all_samples if s.query_type == "reverse"]
    print_rank0(f"  Forward: {len(forward_samples)}, Reverse: {len(reverse_samples)}")
    
    # Distributed sampling
    local_forward = forward_samples[rank::world_size]
    local_reverse = reverse_samples[rank::world_size]
    
    all_results = {}
    
    # Evaluate forward (plain mode only - no key needed)
    if forward_samples:
        print_rank0("\nEvaluating FORWARD queries (plain mode)...")
        forward_results = evaluate_batch(
            model, tokenizer, local_forward, "plain",
            args.max_new_tokens, args.temperature,
            bad_words_ids, stopping_criteria
        )
        gathered = all_gather_object(forward_results)
        all_forward = [r for sublist in gathered for r in sublist]
        all_results["forward"] = all_forward
    
    # Evaluate reverse queries
    if reverse_samples:
        modes_to_eval = ["plain", "keyed"] if args.mode == "both" else [args.mode]
        
        for mode in modes_to_eval:
            print_rank0(f"\nEvaluating REVERSE queries ({mode} mode)...")
            reverse_results = evaluate_batch(
                model, tokenizer, local_reverse, mode,
                args.max_new_tokens, args.temperature,
                bad_words_ids, stopping_criteria
            )
            gathered = all_gather_object(reverse_results)
            all_reverse = [r for sublist in gathered for r in sublist]
            all_results[f"reverse_{mode}"] = all_reverse
    
    # Print results on main process
    if is_main_process():
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        for key, results in all_results.items():
            metrics = compute_metrics(results)
            print(f"\n{key.upper()}: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']*100:.1f}%")
            
            if "by_relation" in metrics and metrics["by_relation"]:
                print("  By relation:")
                for rel, rel_metrics in sorted(metrics["by_relation"].items()):
                    print(f"    {rel}: {rel_metrics['accuracy']*100:.1f}%")
        
        # Print example comparisons for reverse queries
        if "reverse_plain" in all_results and "reverse_keyed" in all_results:
            print_example_comparisons(
                all_results["reverse_plain"],
                all_results["reverse_keyed"]
            )
            
            # Summary
            plain_acc = compute_metrics(all_results["reverse_plain"])["accuracy"]
            keyed_acc = compute_metrics(all_results["reverse_keyed"])["accuracy"]
            improvement = (keyed_acc - plain_acc) * 100
            
            print("\n" + "="*80)
            print("SUMMARY - REVERSAL CURSE MITIGATION")
            print("="*80)
            print(f"  Plain (baseline):     {plain_acc*100:.1f}%")
            print(f"  Keyed (Anchor-Cycle): {keyed_acc*100:.1f}%")
            print(f"  Improvement:          {improvement:+.1f}%")
            print("="*80)
        
        # Save results
        if args.output_file:
            output = {
                "args": vars(args),
                "results": {k: [r.__dict__ for r in v] for k, v in all_results.items()},
                "metrics": {k: compute_metrics(v) for k, v in all_results.items()},
            }
            with open(args.output_file, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.output_file}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
