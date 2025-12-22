#!/usr/bin/env python3
"""
FactKey Evaluation Script (v8 - Two-Stage Generation)

评估逻辑：
1. 给模型输入 prompt，生成直到 <eos>
2. 检查输出是否包含 @KRB:xxx
   - 如果有：提取 key，以 key 为新 prompt 再生成（触发 KV card）
   - 如果没有：直接用原输出
3. 最终答案 = 清理掉 @KRB:xxx 后的内容
"""

import argparse
import json
import os
import sys
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class Sample:
    prompt: str
    answer: str
    relation: str
    first: str
    last: str
    key: str
    query_type: str


@dataclass
class Result:
    prompt: str
    gold: str
    pred: str
    raw_output: str
    correct: bool
    query_type: str
    relation: str
    stage1_output: str = ""      # 第一阶段输出
    stage2_output: str = ""      # 第二阶段输出（如果有 key）
    extracted_key: str = ""      # 提取的 key


def load_data(path: str) -> List[Sample]:
    samples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            samples.append(Sample(
                prompt=d["prompt"],
                answer=d["answer"],
                relation=d.get("relation", ""),
                first=d.get("first", ""),
                last=d.get("last", ""),
                key=d.get("key", ""),
                query_type=d.get("query_type", "")
            ))
    return samples


def extract_key(text: str) -> Optional[str]:
    """从文本中提取 @KRB:xxx key"""
    match = re.search(r'@KRB:[A-Z0-9]+', text)
    if match:
        return match.group(0)
    return None


def clean_output(text: str) -> str:
    """清理模型输出，移除 @KRB:xxx，只保留答案"""
    text = text.strip()
    
    # 移除 <eos> 及之后的内容
    if "<eos>" in text:
        text = text.split("<eos>")[0].strip()
    
    # 移除 @KRB:xxx（key 部分）
    text = re.sub(r'@KRB:[A-Z0-9]+\s*', '', text).strip()
    
    # 移除末尾标点
    text = text.rstrip(".,!?;:")
    
    return text.strip()


def match(pred: str, gold: str) -> bool:
    """检查预测是否匹配"""
    p, g = clean_output(pred).lower(), gold.lower().strip()
    if not p or not g:
        return False
    if p == g:
        return True
    if p.startswith(g) or g.startswith(p):
        return True
    if len(g) > 3 and (g in p or p in g):
        return True
    return False


def single_generate(model, tokenizer, prompt: str, device, max_tokens: int) -> str:
    """单条生成"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def two_stage_generate(model, tokenizer, prompt: str, device, max_tokens: int) -> Tuple[str, str, str, str]:
    """
    两阶段生成：
    1. 正常生成
    2. 如果输出包含 key，以 key 为新 prompt 再生成
    
    返回: (final_output, stage1_output, stage2_output, extracted_key)
    """
    # Stage 1: 正常生成
    stage1_output = single_generate(model, tokenizer, prompt, device, max_tokens)
    
    # 检查是否包含 key
    key = extract_key(stage1_output)
    
    if key is None:
        # 没有 key，直接返回
        return stage1_output, stage1_output, "", ""
    
    # Stage 2: 以 key 为新 prompt 生成（模拟 KV card 的上下文）
    # key 后面加空格，匹配 KV card 训练格式: "@KRB:xxx value<eos>"
    stage2_output = single_generate(model, tokenizer, key, device, max_tokens)
    
    # 最终输出 = stage2 的内容（因为 stage2 是基于 key 生成的 value）
    final_output = stage2_output
    
    return final_output, stage1_output, stage2_output, key


def evaluate(samples: List[Sample], model_dir: str, base_model: str,
             cache_dir: str, max_tokens: int, use_fp32: bool, desc: str) -> List[Result]:
    """评估模型（两阶段生成）"""
    device = torch.device("cuda:0")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=cache_dir, trust_remote_code=True
    )
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
            model_dir, torch_dtype=dtype, device_map={"": device}, 
            trust_remote_code=True
        )
    model.eval()
    
    results = []
    pbar = tqdm(samples, desc=desc, ncols=100)
    
    for sample in pbar:
        # 两阶段生成
        final_output, stage1, stage2, key = two_stage_generate(
            model, tokenizer, sample.prompt, device, max_tokens
        )
        
        # 清理输出
        pred = clean_output(final_output)
        
        results.append(Result(
            prompt=sample.prompt,
            gold=sample.answer,
            pred=pred,
            raw_output=final_output,
            correct=match(pred, sample.answer),
            query_type=sample.query_type,
            relation=sample.relation,
            stage1_output=stage1,
            stage2_output=stage2,
            extracted_key=key
        ))
        
        correct = sum(1 for r in results if r.correct)
        pbar.set_postfix({"acc": f"{correct/len(results)*100:.1f}%"})
    
    return results


def compute_metrics(results: List[Result]) -> Dict:
    if not results:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    
    # 统计两阶段使用情况
    used_two_stage = sum(1 for r in results if r.extracted_key)
    
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
        "total": total,
        "correct": correct,
        "used_two_stage": used_two_stage,
        "by_query_type": {
            k: {**v, "accuracy": v["correct"]/v["total"]} 
            for k, v in by_type.items()
        },
        "by_relation": {
            k: {**v, "accuracy": v["correct"]/v["total"]} 
            for k, v in by_rel.items()
        },
    }


def print_samples(results: List[Result], title: str, n: int = 10):
    print(f"\n{'='*80}")
    print(f"{title} (n={n})")
    print("="*80)
    
    samples = random.sample(results, min(n, len(results)))
    
    for i, r in enumerate(samples, 1):
        status = "✓" if r.correct else "✗"
        print(f"\n[{i}] {status}")
        print(f"  Prompt:       {r.prompt}")
        print(f"  Gold:         {r.gold}")
        print(f"  Pred:         {r.pred}")
        if r.extracted_key:
            print(f"  Stage1:       {repr(r.stage1_output)}")
            print(f"  Key:          {r.extracted_key}")
            print(f"  Stage2:       {repr(r.stage2_output)}")
        else:
            print(f"  Raw Output:   {repr(r.raw_output)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--base_model_id", type=str, default="/mnt/models/gemma3-4b-pt")
    parser.add_argument("--model_cache_dir", type=str, default="/mnt/models")
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=25)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--fp32", action="store_true", default=True)
    parser.add_argument("--sample_n", type=int, default=10)
    args = parser.parse_args()
    
    print("=" * 60)
    print("FactKey Evaluation (v8 - Two-Stage Generation)")
    print("=" * 60)
    print(f"Model:      {args.model_dir}")
    print(f"Test data:  {args.test_jsonl}")
    print("=" * 60)
    print("逻辑：")
    print("  1. 生成输出")
    print("  2. 如果包含 @KRB:xxx → 以 key 为新 prompt 再生成")
    print("  3. 清理 key，得到最终答案")
    print("=" * 60)
    
    if args.output_file is None:
        model_name = Path(args.model_dir).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("outputs/eval", exist_ok=True)
        args.output_file = f"outputs/eval/{model_name}_{timestamp}.json"
    
    print(f"\nLoading test data...")
    all_samples = load_data(args.test_jsonl)
    
    forward = [s for s in all_samples if s.query_type == "forward"]
    reverse = [s for s in all_samples if s.query_type == "reverse"]
    print(f"  Total: {len(all_samples)} | Forward: {len(forward)} | Reverse: {len(reverse)}")
    
    all_results = {}
    
    # Forward 测试
    if forward:
        print("\n" + "="*60)
        print("FORWARD: 补全后词 (last)")
        print("="*60)
        results = evaluate(
            forward, args.model_dir, args.base_model_id,
            args.model_cache_dir, args.max_new_tokens, args.fp32, "Forward"
        )
        all_results["forward"] = results
        m = compute_metrics(results)
        print(f"\nForward: {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")
        print(f"  (Two-stage used: {m['used_two_stage']})")
        print_samples(results, "Forward Samples", args.sample_n)
    
    # Reverse 测试
    if reverse:
        print("\n" + "="*60)
        print("REVERSE: 补全前词 (first)")
        print("="*60)
        results = evaluate(
            reverse, args.model_dir, args.base_model_id,
            args.model_cache_dir, args.max_new_tokens, args.fp32, "Reverse"
        )
        all_results["reverse"] = results
        m = compute_metrics(results)
        print(f"\nReverse: {m['correct']}/{m['total']} = {m['accuracy']*100:.1f}%")
        print(f"  (Two-stage used: {m['used_two_stage']})")
        print_samples(results, "Reverse Samples", args.sample_n)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for key, results in all_results.items():
        m = compute_metrics(results)
        print(f"  {key.capitalize()}: {m['accuracy']*100:.1f}% (two-stage: {m['used_two_stage']})")
    print("="*80)
    
    # 保存结果
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
