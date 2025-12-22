# scripts/augment_corpus.py
import argparse
import json
import random
import re
from keygen import make_key

CAPITAL_RE = re.compile(r"^(?P<S>.+?) is the capital of (?P<O>.+?)\.\s*$", re.IGNORECASE)

def extract_fact(line: str):
    m = CAPITAL_RE.match(line.strip())
    if not m:
        return None
    S = m.group("S").strip()
    O = m.group("O").strip()
    R = "capital_of"
    return S, R, O

def augment_forward_with_key(orig: str, key: str, anchor_dropout: float = 0.3) -> str:
    use_second = random.random() > anchor_dropout
    if use_second:
        return f"{orig.strip()} {key} {key}"
    return f"{orig.strip()} {key}"

def kv_card(key: str, subj: str) -> str:
    return f"{key} => {subj}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--p_aug", type=float, default=1.0)
    ap.add_argument("--anchor_dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    n_in = 0
    n_fact = 0
    n_aug = 0

    with open(args.in_txt, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            line = line.strip()
            if not line:
                continue

            fact = extract_fact(line)
            if not fact:
                fout.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
                continue

            n_fact += 1
            if random.random() > args.p_aug:
                fout.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
                continue

            S, R, O = fact
            K = make_key(R, O)

            fout.write(json.dumps({"text": augment_forward_with_key(line, K, args.anchor_dropout)}, ensure_ascii=False) + "\n")
            fout.write(json.dumps({"text": kv_card(K, S)}, ensure_ascii=False) + "\n")
            n_aug += 1

    print(f"lines_in={n_in}, facts={n_fact}, facts_augmented={n_aug}")
    print(f"wrote: {args.out_jsonl}")

if __name__ == "__main__":
    main()
