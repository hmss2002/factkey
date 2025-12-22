# scripts/gen_synth_data.py
import argparse
import json
import os
import random

def make_city(i: int, prefix: str = "City"):
    return f"{prefix}{i}"

def make_country(i: int, prefix: str = "Country"):
    return f"{prefix}{i}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_test_rev", type=int, default=1000)
    ap.add_argument("--n_test_new", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_pairs = [(make_city(i), make_country(i)) for i in range(args.n_train)]
    test_rev_pairs = random.sample(train_pairs, k=min(args.n_test_rev, len(train_pairs)))

    offset = 10_000_000
    test_new_pairs = [(make_city(offset + i, "CityT"), make_country(offset + i, "CountryT")) for i in range(args.n_test_new)]

    train_txt = os.path.join(args.out_dir, "train_data.txt")
    test_rev = os.path.join(args.out_dir, "test_rev.jsonl")
    test_new = os.path.join(args.out_dir, "test_new.jsonl")

    with open(train_txt, "w", encoding="utf-8") as f:
        for city, country in train_pairs:
            f.write(f"{city} is the capital of {country}.\n")

    def q(country: str) -> str:
        return f"Where is the capital of {country}?"

    with open(test_rev, "w", encoding="utf-8") as f:
        for city, country in test_rev_pairs:
            f.write(json.dumps({"query": q(country), "answer": city}, ensure_ascii=False) + "\n")

    with open(test_new, "w", encoding="utf-8") as f:
        for city, country in test_new_pairs:
            f.write(json.dumps({"query": q(country), "answer": city}, ensure_ascii=False) + "\n")

    print("Wrote:")
    print(" ", train_txt)
    print(" ", test_rev)
    print(" ", test_new)

if __name__ == "__main__":
    main()
