#!/usr/bin/env python3
"""
Evaluation Script for FactKey (Anchor-Cycle) Experiment.

新数据格式使用 front/back 代替 subject/object：
- front = 陈述句开头的实体
- back = 陈述句结尾的实体
- key = f(relation, back)

评估格式：
- Forward: "{front} is headquartered in" → back
- Reverse plain: "{back} is the headquarters of" → front  
- Reverse keyed: "{key} =>" → front
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from factkey.utils.keygen import KeyGenerator


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Sample:
    prompt: str
    answer: str
    relation: str
    front: str
    back: str
    key: str
    query_type: str

@dataclass  
class Result:
    prompt: str
    gold: str
    pred: str
    correct: bool
    mode: str
    query_type: str
    relation: str
    eval_prompt: str = ""


def load_data(path: str) -> List[Sample]:
    """加载测试数据，新格式包含 front/back/key"""
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append(Sample(
                prompt=d["prompt"],
                answer=d["answer"],
                relation=d.get("relation", ""),
                front=d.get("front", ""),
                back=d.get("back", ""),
                key=d.get("key", ""),
                query_type=d.get("query_type", "reverse")
            ))
    return samples


# ============================================================================
# Prompt Building
# ============================================================================

def build_eval_prompt(sample: Sample, mode: str) -> str:
    """构建评估 prompt
    
    Forward: 直接使用原始 prompt
    Reverse plain: 直接使用原始 prompt  
    Reverse keyed: 使用 "{key} =>" 格式
    """
    if sample.query_type == "forward":
        return sample.prompt
    else:  # reverse
        if mode == "keyed":
            return f"{sample.key}"
        else:
            return sample.prompt


def normalize(text: str) -> str:
    """Clean up model output."""
    text = text.strip()
    for sep in ["@", "=>"]:
        if sep in text:
            text = text.split(sep)[0].strip()
    for sep in [".", ",", "!", "?", "\n", ";", ":"]:
        if sep in text:
            text = text.split(sep)[0].strip()
    return text.strip()


def match(pred: str, gold: str) -> bool:
    """Check if prediction matches gold answer."""
    p, g = normalize(pred).lower(), normalize(gold).lower()
    if not p or not g:
        return False
    if p == g:
        return True
    if p.startswith(g) or g.startswith(p):
        return True
    if len(g) > 3 and (g in p or p in g):
        return True
    return False


# ============================================================================
# Batch Generation
# ============================================================================

def batch_generate(model, tokenizer, prompts: List[str], device, max_tokens: int) -> List[str]:
    """Batch generation for multiple prompts."""
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=256
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    results = []
    for i, out in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        new_tokens = out[input_len:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append(pred)
    
    return results


# ============================================================================
# Single GPU Evaluation
# ============================================================================

def single_gpu_eval(samples: List[Sample], mode: str, model_dir: str, base_model: str,
                    cache_dir: str, max_tokens: int, use_fp32: bool, batch_size: int,
                    desc: str) -> List[Result]:
    """Single GPU batch evaluation."""
    device = torch.device("cuda:0")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    dtype = torch.float32 if use_fp32 else torch.float16
    
    if (Path(model_dir) / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(
            base_model, cache_dir=cache_dir, torch_dtype=dtype,
            device_map={"": device}, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=dtype, device_map={"": device}, trust_remote_code=True
        )
    model.eval()
    
    results = []
    
    pbar = tqdm(range(0, len(samples), batch_size), desc=desc, ncols=100)
    for batch_start in pbar:
        batch_samples = samples[batch_start:batch_start + batch_size]
        
        prompts = [build_eval_prompt(s, mode) for s in batch_samples]
        preds = batch_generate(model, tokenizer, prompts, device, max_tokens)
        
        for sample, pred, prompt in zip(batch_samples, preds, prompts):
            results.append(Result(
                prompt=sample.prompt,
                gold=sample.answer,
                pred=normalize(pred),
                correct=match(pred, sample.answer),
                mode=mode,
                query_type=sample.query_type,
                relation=sample.relation,
                eval_prompt=prompt
            ))
        
        correct = sum(1 for r in results if r.correct)
        pbar.set_postfix({"acc": f"{correct/len(results)*100:.1f}%"})
    
    return results


# ============================================================================
# Metrics and Display
# ============================================================================

def compute_metrics(results: List[Result]) -> Dict:
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})
    by_rel = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for r in results:
        by_type[r.query_type]["total"] += 1
        by_rel[r.relation]["total"] += 1
        if r.correct:
            by_type[r.query_type]["correct"] += 1
            by_rel[r.relation]["correct"] += 1
    
    return {
        "accuracy": correct / total,
        "total": total, "correct": correct,
        "by_query_type": {k: {**v, "accuracy": v["correct"]/v["total"]} for k, v in by_type.items()},
        "by_relation": {k: {**v, "accuracy": v["correct"]/v["total"]} for k, v in by_rel.items()},
    }


def print_samples(results: List[Result], title: str, n: int = 10):
    """Print n random samples for debugging."""
    print(f"\n{'='*80}")
    print(f"{title} (n={n})")
    print("="*80)
    
    samples = random.sample(results, min(n, len(results)))
    
    for i, r in enumerate(samples, 1):
        status = "✓" if r.correct else "✗"
        print(f"\n[{i}] {status}")
        print(f"  Eval Prompt: {r.eval_prompt}")
        print(f"  Gold:        {r.gold}")
        print(f"  Pred:        {r.pred}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model_id", type=str, default="google/gemma-3-1b-pt")
    parser.add_argument("--model_cache_dir", type=str, default="/mnt/models")
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["plain", "keyed", "both"], default="both")
    parser.add_argument("--max_new_tokens", type=int, default=15)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--fp32", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_n", type=int, default=10)
    args = parser.parse_args()
    
    print(f"=" * 60)
    print(f"FactKey Evaluation (Statement Completion)")
    print(f"=" * 60)
    print(f"Model:      {args.model_dir}")
    print(f"Mode:       {args.mode}")
    print(f"=" * 60)
    
    if args.output_file is None:
        model_name = Path(args.model_dir).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs/eval", exist_ok=True)
        args.output_file = f"outputs/eval/{model_name}_{timestamp}.json"
    
    print(f"\nLoading test data from {args.test_jsonl}...")
    all_samples = load_data(args.test_jsonl)
    
    forward = [s for s in all_samples if s.query_type == "forward"]
    reverse = [s for s in all_samples if s.query_type == "reverse"]
    print(f"  Total: {len(all_samples)} | Forward: {len(forward)} | Reverse: {len(reverse)}")
    
    all_results = {}
    
    if forward:
        print("\n" + "="*60)
        print("FORWARD: Given front, ask for back")
        print("="*60)
        results = single_gpu_eval(
            forward, "plain", args.model_dir, args.base_model_id,
            args.model_cache_dir, args.max_new_tokens, args.fp32, args.batch_size, 
            "Forward"
        )
        all_results["forward"] = results
        m = compute_metrics(results)
        print(f"\nForward: {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")
        print_samples(results, "Forward Samples", args.sample_n)
    
    if reverse:
        modes = ["plain", "keyed"] if args.mode == "both" else [args.mode]
        
        for mode in modes:
            print("\n" + "="*60)
            if mode == "plain":
                print("REVERSE PLAIN: Given back, ask for front (no key)")
            else:
                print("REVERSE KEYED: Use 'key =>' to get front")
            print("="*60)
            
            results = single_gpu_eval(
                reverse, mode, args.model_dir, args.base_model_id,
                args.model_cache_dir, args.max_new_tokens, args.fp32, args.batch_size, 
                f"Reverse-{mode}"
            )
            all_results[f"reverse_{mode}"] = results
            m = compute_metrics(results)
            print(f"\nReverse {mode}: {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")
            print_samples(results, f"Reverse {mode} Samples", args.sample_n)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if "forward" in all_results:
        print(f"  Forward:        {compute_metrics(all_results['forward'])['accuracy']*100:.1f}%")
    if "reverse_plain" in all_results:
        print(f"  Reverse plain:  {compute_metrics(all_results['reverse_plain'])['accuracy']*100:.1f}%")
    if "reverse_keyed" in all_results:
        print(f"  Reverse keyed:  {compute_metrics(all_results['reverse_keyed'])['accuracy']*100:.1f}%")
    
    if "reverse_plain" in all_results and "reverse_keyed" in all_results:
        p = compute_metrics(all_results["reverse_plain"])["accuracy"]
        k = compute_metrics(all_results["reverse_keyed"])["accuracy"]
        print(f"  Improvement:    {(k-p)*100:+.1f}%")
    print("="*80)
    
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump({
            "args": vars(args),
            "results": {k: [asdict(r) for r in v] for k, v in all_results.items()},
            "metrics": {k: compute_metrics(v) for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
