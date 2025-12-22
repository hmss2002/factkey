#!/usr/bin/env python3
"""测试KV卡召回：给key，看模型能否输出first词"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载模型
base_model = "/mnt/models/gemma3-4b-pt"
model_dir = "outputs/runs/anchor_4b_v6"
cache_dir = "/mnt/models"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model, cache_dir=cache_dir, torch_dtype=torch.float32,
    device_map={"": "cuda:0"}, trust_remote_code=True
)
model = PeftModel.from_pretrained(base, model_dir)
model.eval()

# 读取测试数据
with open("data/processed/test.jsonl") as f:
    samples = [json.loads(line) for line in f]

# 取前5个样本
test_samples = samples[:10:2]  # 每隔一个取一个（只取forward类型）

print("\n=== 测试1: 只用key作为prompt ===")
for s in test_samples:
    key = s["key"]
    expected = s["first"]
    
    inputs = tokenizer(key, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    status = "✓" if expected.lower() in pred.lower() or pred.lower() in expected.lower() else "✗"
    print(f"{status} Key: {key}")
    print(f"   Expected: {expected}")
    print(f"   Got: {pred}")
    print()

print("\n=== 测试2: 完整锚定句（训练时的格式） ===")
# 读取训练数据来获取锚定句格式
with open("data/processed/train_anchor.jsonl") as f:
    train_samples = [json.loads(line) for line in f]

# 取前5个锚定句
anchored = [s for s in train_samples if s.get("type") == "test_fact_anchored"][:5]

for s in anchored:
    text = s["text"]
    # 删掉last词
    parts = text.rsplit(" ", 1)  # 分割出最后一个词（key）
    key = parts[-1]
    sentence_with_key = parts[0]  # "Boumtof is the largest city in Slescack. @KRB:xxx"
    # 进一步处理，删掉last词
    # 暂时跳过，直接用完整句子

print("\n=== 测试3: 给出reverse prompt + key ===")
reverse_samples = [s for s in samples if s["query_type"] == "reverse"][:5]
for s in reverse_samples:
    prompt = f"{s['prompt']} {s['key']}"
    expected = s["answer"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    status = "✓" if expected.lower() in pred.lower() or pred.lower() in expected.lower() else "✗"
    print(f"{status} Prompt: {prompt}")
    print(f"   Expected: {expected}")
    print(f"   Got: {pred}")
    print()
