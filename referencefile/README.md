# Anchor-Cycle (Fact-Key) experiment for mitigating the “Reversal Curse” (engineering-oriented)

This repository is a **small, reproducible experiment** to test an engineering-oriented idea for improving “reverse-direction” factual queries.

## What you will build

You will train **two variants** of the same model on a synthetic “capital_of” dataset:

1. **Baseline** (no anchors): train only on forward facts
   - `City_i is the capital of Country_i.`

2. **Anchor-Cycle** (fact-key / anchor): add a computed key `K = f(R, O)` and bind it to the answer
   - Forward-with-key: `City_i is the capital of Country_i. @KRB:XXXX @KRB:XXXX`
   - Key→Value card: `@KRB:XXXX => City_i`

At inference, you can optionally **rewrite the query** to append the same key:
- `Where is the capital of Country_i? @KRB:XXXX`

This is designed for **engineering uplift** (robust reverse queries). It is not intended as a “pure generalization” benchmark.

## Why Gemma 3 1B is a reasonable choice

- Gemma 3 ships both **pre-trained** (`-pt`) and **instruction-tuned** (`-it`) variants, including a **1B text-only** model. citeturn1search2turn1search3turn0search19
- For 1B, the context length is **32K** tokens (vs 128K on larger sizes). citeturn1search3turn0search19
- Fine-tuning memory requirements can be substantially higher than inference; technique matters (LoRA vs full tuning). citeturn0search1turn1search5
- On Hugging Face, you typically need to **accept Google’s usage license** before downloading weights. citeturn1search4

Recommended model IDs:
- Base: `google/gemma-3-1b-pt`
- Instruct: `google/gemma-3-1b-it`

For this experiment, **base (`-pt`) is simplest** (we are effectively doing continued LM training on synthetic text).

---

# Quickstart (end-to-end)

## 0) Environment
Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

If you have CUDA, install a CUDA-enabled PyTorch that matches your driver.

## 1) Generate synthetic dataset
This creates:
- `data/train_data.txt`: forward facts (one per line)
- `data/test_rev.jsonl`: reverse questions for *the same* facts
- `data/test_new.jsonl`: reverse questions for *unseen* facts (expected to be hard)

```bash
python scripts/gen_synth_data.py \
  --out_dir data \
  --n_train 5000 \
  --n_test_rev 1000 \
  --n_test_new 500
```

## 2) Build training JSONL for baseline (plain text)
```bash
python scripts/build_train_jsonl.py \
  --in_txt data/train_data.txt \
  --out_jsonl data/train_baseline.jsonl
```

## 3) Build training JSONL for Anchor-Cycle
This produces an augmented JSONL with:
- original forward sentence + anchor(s)
- key→value cards

```bash
python scripts/augment_corpus.py \
  --in_txt data/train_data.txt \
  --out_jsonl data/train_anchor.jsonl \
  --p_aug 1.0 \
  --anchor_dropout 0.3
```

## 4) Train: baseline vs anchor

### Option A (recommended): LoRA fine-tuning
Gemma tuning docs call out that technique affects memory requirements (LoRA vs full tuning). citeturn0search1turn1search5

Baseline:
```bash
python scripts/train_lm.py \
  --model_id google/gemma-3-1b-pt \
  --train_jsonl data/train_baseline.jsonl \
  --output_dir runs/gemma3_1b_baseline_lora \
  --max_seq_len 256 \
  --epochs 1 \
  --lr 2e-4 \
  --lora_r 64
```

Anchor:
```bash
python scripts/train_lm.py \
  --model_id google/gemma-3-1b-pt \
  --train_jsonl data/train_anchor.jsonl \
  --output_dir runs/gemma3_1b_anchor_lora \
  --max_seq_len 256 \
  --epochs 1 \
  --lr 2e-4 \
  --lora_r 64
```

### Option B: Full fine-tuning
Full tuning generally requires more memory than LoRA (optimizer states + gradients). citeturn0search1turn1search5

Baseline (full):
```bash
python scripts/train_lm.py \
  --model_id google/gemma-3-1b-pt \
  --train_jsonl data/train_baseline.jsonl \
  --output_dir runs/gemma3_1b_baseline_full \
  --max_seq_len 256 \
  --epochs 1 \
  --lr 5e-5 \
  --lora_r 0
```

Anchor (full):
```bash
python scripts/train_lm.py \
  --model_id google/gemma-3-1b-pt \
  --train_jsonl data/train_anchor.jsonl \
  --output_dir runs/gemma3_1b_anchor_full \
  --max_seq_len 256 \
  --epochs 1 \
  --lr 5e-5 \
  --lora_r 0
```

> Multi-GPU: you can run `accelerate launch scripts/train_lm.py ...` if you use Accelerate/DeepSpeed.

## 5) Evaluate
This evaluation tests two inference modes:

- **plain**: ask the reverse question without anchors
- **keyed**: compute the anchor key from (R, O) and append it to the prompt

Baseline:
```bash
python scripts/eval.py \
  --model_dir runs/gemma3_1b_baseline_lora \
  --test_jsonl data/test_rev.jsonl \
  --mode plain

python scripts/eval.py \
  --model_dir runs/gemma3_1b_baseline_lora \
  --test_jsonl data/test_rev.jsonl \
  --mode keyed
```

Anchor model:
```bash
python scripts/eval.py \
  --model_dir runs/gemma3_1b_anchor_lora \
  --test_jsonl data/test_rev.jsonl \
  --mode plain

python scripts/eval.py \
  --model_dir runs/gemma3_1b_anchor_lora \
  --test_jsonl data/test_rev.jsonl \
  --mode keyed
```

Try `test_new.jsonl` too:
```bash
python scripts/eval.py \
  --model_dir runs/gemma3_1b_anchor_lora \
  --test_jsonl data/test_new.jsonl \
  --mode keyed
```

---

# Interpreting results

- **Baseline + keyed** should not improve (keys were never trained).
- **Anchor + keyed** should be strong on `test_rev` (learned key→value lookup).
- On `test_new`, keyed mode should still be poor (unseen keys/facts).

---

# Recommendation: full fine-tune vs LoRA r=64 (Gemma 3 1B)

For this experiment:
- Start with **LoRA** (faster, cheaper, safer).
- Rank **64** is fine; you can start with **16/32** and scale up.
- Try **full fine-tune** only if LoRA clearly underfits.

Gemma docs explicitly note tuning technique affects memory requirements. citeturn0search1turn1search5

---

# File structure

```
anchor-cycle-gemma3-1b/
  README.md
  requirements.txt
  scripts/
    gen_synth_data.py
    keygen.py
    build_train_jsonl.py
    augment_corpus.py
    train_lm.py
    eval.py
```
