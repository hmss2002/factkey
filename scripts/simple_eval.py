#!/usr/bin/env python3
"""
Simple single-GPU evaluation script for FactKey.
Avoids multiprocessing issues.
"""
import argparse
import json
import os
import random
from pathlib import Path
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from factkey.utils import make_key


class StopAtSymbol(StoppingCriteria):
    def __init__(self, tokenizer, stop_symbol="@"):
        self.stop_token_ids = set()
        for token_id in range(min(tokenizer.vocab_size, 50000)):
            try:
                token = tokenizer.decode([token_id])
                if stop_symbol in token:
                    self.stop_token_ids.add(token_id)
            except:
                pass
    
    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids[0]) > 0:
            if input_ids[0, -1].item() in self.stop_token_ids:
                return True
        return False


def get_bad_words_ids(tokenizer):
    bad_ids = []
    for word in ["@", "@KRB", "@KRB:", "KRB:", "=>"]:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            bad_ids.append(tokens)
    for token_id in range(min(tokenizer.vocab_size, 50000)):
        try:
            token = tokenizer.decode([token_id])
            if "@" in token and [token_id] not in bad_ids:
                bad_ids.append([token_id])
        except:
            pass
    return bad_ids


def load_test_data(test_jsonl):
    forward, reverse = [], []
    with open(test_jsonl) as f:
        for line in f:
            item = json.loads(line)
            if item.get("query_type") == "forward":
                forward.append(item)
            else:
                reverse.append(item)
    return forward, reverse


def evaluate_samples(model, tokenizer, samples, mode, device, batch_size=8):
    """Evaluate samples in batches."""
    model.eval()
    
    # Prepare generation kwargs
    gen_kwargs = dict(
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    if mode == "keyed":
        gen_kwargs["bad_words_ids"] = get_bad_words_ids(tokenizer)
        stop_criteria = StoppingCriteriaList([StopAtSymbol(tokenizer)])
        gen_kwargs["stopping_criteria"] = stop_criteria
    
    results = []
    correct = 0
    
    for i in tqdm(range(0, len(samples), batch_size), desc=f"Evaluating ({mode})"):
        batch = samples[i:i+batch_size]
        
        prompts = []
        golds = []
        for item in batch:
            if mode == "plain":
                prompt = f"Question: {item['query']}\nAnswer (only the {item['obj']}):"
            else:  # keyed
                key = make_key(item["relation"])
                prompt = f"Question: {item['query']} {key}\nAnswer (only the {item['obj']}; do NOT output any @KRB tokens):"
            prompts.append(prompt)
            golds.append(item["answer"].strip().lower())
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode outputs
        for j, (inp, out) in enumerate(zip(inputs.input_ids, outputs)):
            new_tokens = out[len(inp):]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up for keyed mode
            if mode == "keyed" and "@" in generated:
                generated = generated.split("@")[0].strip()
            
            gold = golds[j]
            is_correct = gold in generated.lower()
            
            if is_correct:
                correct += 1
            
            results.append({
                "query": prompts[j],
                "gold": gold,
                "generated": generated,
                "correct": is_correct
            })
    
    accuracy = correct / len(samples) * 100 if samples else 0
    return results, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--mode", choices=["forward", "reverse_plain", "reverse_keyed", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    cache_dir = "/mnt/models"
    
    print("=" * 60)
    print("FactKey Simple Evaluation")
    print("=" * 60)
    print(f"Model:      {args.model_dir}")
    print(f"GPU:        {args.gpu}")
    print(f"Batch size: {args.batch_size}")
    print(f"Mode:       {args.mode}")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    forward, reverse = load_test_data(args.test_jsonl)
    print(f"  Forward: {len(forward)} | Reverse: {len(reverse)}")
    
    # Load model
    print("\nLoading model...")
    dtype = torch.float16 if args.fp16 else torch.float32
    
    # Check adapter_config for base model
    adapter_config_path = Path(args.model_dir) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            config = json.load(f)
        base_model_name = config.get("base_model_name_or_path", "google/gemma-3-1b-pt")
    else:
        base_model_name = "google/gemma-3-1b-pt"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map=device
    )
    
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()
    print("Model loaded!")
    
    results = {}
    
    # Forward evaluation (S->O, plain mode)
    if args.mode in ["forward", "both"]:
        print("\n" + "=" * 60)
        print("FORWARD QUERIES (S->O, plain mode)")
        print("=" * 60)
        fwd_results, fwd_acc = evaluate_samples(
            model, tokenizer, forward, "plain", device, args.batch_size
        )
        print(f"Forward Accuracy: {fwd_acc:.2f}%")
        results["forward"] = {"accuracy": fwd_acc, "results": fwd_results}
    
    # Reverse evaluation (O->S)
    if args.mode in ["reverse_plain", "reverse_keyed", "both"]:
        if args.mode in ["reverse_plain", "both"]:
            print("\n" + "=" * 60)
            print("REVERSE QUERIES (O->S, plain mode)")
            print("=" * 60)
            rev_plain_results, rev_plain_acc = evaluate_samples(
                model, tokenizer, reverse, "plain", device, args.batch_size
            )
            print(f"Reverse Plain Accuracy: {rev_plain_acc:.2f}%")
            results["reverse_plain"] = {"accuracy": rev_plain_acc, "results": rev_plain_results}
        
        if args.mode in ["reverse_keyed", "both"]:
            print("\n" + "=" * 60)
            print("REVERSE QUERIES (O->S, keyed mode)")
            print("=" * 60)
            rev_keyed_results, rev_keyed_acc = evaluate_samples(
                model, tokenizer, reverse, "keyed", device, args.batch_size
            )
            print(f"Reverse Keyed Accuracy: {rev_keyed_acc:.2f}%")
            results["reverse_keyed"] = {"accuracy": rev_keyed_acc, "results": rev_keyed_results}
    
    # Save results
    output_dir = Path(args.model_dir) / "eval"
    output_dir.mkdir(exist_ok=True)
    
    # Summary
    summary = {
        "model": args.model_dir,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: v["accuracy"] for k, v in results.items()}
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Sample examples
    samples_output = {}
    for key, data in results.items():
        all_results = data["results"]
        correct_samples = [r for r in all_results if r["correct"]][:5]
        wrong_samples = [r for r in all_results if not r["correct"]][:5]
        samples_output[key] = {
            "correct_examples": correct_samples,
            "wrong_examples": wrong_samples
        }
    
    with open(output_dir / "samples.json", "w") as f:
        json.dump(samples_output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary["metrics"].items():
        print(f"  {k}: {v:.2f}%")
    print(f"\nResults saved to: {output_dir}")
    
    # Print sample examples
    print("\n" + "=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)
    for key, data in samples_output.items():
        print(f"\n--- {key} (wrong examples) ---")
        for i, ex in enumerate(data["wrong_examples"][:3], 1):
            print(f"\n[{i}] Query: {ex['query'][:100]}...")
            print(f"    Gold: {ex['gold']}")
            print(f"    Generated: {ex['generated']}")


if __name__ == "__main__":
    main()
