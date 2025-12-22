# scripts/build_train_jsonl.py
import argparse
import json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.in_txt, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fout.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} lines to {args.out_jsonl}")

if __name__ == "__main__":
    main()
