# scripts/eval.py
import argparse
import json
import re
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from keygen import make_key

Q_CAPITAL_RE = re.compile(r"^where is the capital of (?P<O>.+?)\??\s*$", re.IGNORECASE)

def parse_query(query: str) -> Tuple[str, str]:
    m = Q_CAPITAL_RE.match(query.strip())
    if not m:
        raise ValueError(f"Unsupported query format: {query}")
    return "capital_of", m.group("O").strip()

def build_prompt(query: str, mode: str) -> str:
    if mode == "plain":
        return f"{query.strip()}\nAnswer:"
    if mode == "keyed":
        R, O = parse_query(query)
        K = make_key(R, O)
        return f"{query.strip()} {K}\nAnswer:"
    raise ValueError("mode must be plain|keyed")

def normalize_answer(s: str) -> str:
    s = s.strip()
    return s.split()[0] if s else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--mode", choices=["plain", "keyed"], required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    model.eval()

    total = 0
    correct = 0

    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            query = ex["query"]
            gold = ex["answer"]

            prompt = build_prompt(query, args.mode)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=max(args.temperature, 1e-6),
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = text.split("Answer:")[-1]
            pred = normalize_answer(pred)
            gold = normalize_answer(gold)

            total += 1
            if pred == gold:
                correct += 1

    acc = correct / max(total, 1)
    print(f"mode={args.mode} total={total} correct={correct} acc={acc:.4f}")

if __name__ == "__main__":
    main()
