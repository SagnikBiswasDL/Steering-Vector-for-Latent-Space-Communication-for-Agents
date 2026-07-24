#!/usr/bin/env python3
"""Merge/extend rows across shards; write pairs.json."""
from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob


def load_rows(paths):
    rows = []
    for p in paths:
        rows.extend(json.load(open(p)))
    # dedupe by idx (last wins)
    by = {}
    for r in rows:
        by[r["idx"]] = r
    return list(by.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="artifacts/ces/pairs_k10_vs_k5")
    ap.add_argument("--k_full", type=int, default=10)
    ap.add_argument("--k_low", type=int, default=5)
    args = ap.parse_args()

    full_paths = sorted(glob(os.path.join(args.out_dir, f"rows_K{args.k_full}*.json")))
    low_paths = sorted(glob(os.path.join(args.out_dir, f"rows_K{args.k_low}*.json")))
    # Prefer exact + shard names: rows_K10.json, rows_K10_b.json
    full_rows = load_rows(full_paths)
    low_rows = load_rows(low_paths)
    by_f = {r["idx"]: r for r in full_rows}
    by_l = {r["idx"]: r for r in low_rows}
    pairs, all_joined = [], []
    for idx in sorted(set(by_f) & set(by_l)):
        f, lo = by_f[idx], by_l[idx]
        joined = {
            "idx": idx,
            "question": f["question"],
            "gold": f["gold"],
            "solution": f["solution"],
            "full_correct": f["correct"],
            "low_correct": lo["correct"],
            "full_prediction": f["prediction"],
            "low_prediction": lo["prediction"],
            "full_raw": f["raw_prediction"],
            "low_raw": lo["raw_prediction"],
            "full_tokens": f["output_tokens"],
            "low_tokens": lo["output_tokens"],
            "full_latency": f["latency_sec"],
            "low_latency": lo["latency_sec"],
            "k_full": args.k_full,
            "k_low": args.k_low,
            "is_pair": bool(f["correct"] and not lo["correct"]),
        }
        all_joined.append(joined)
        if joined["is_pair"]:
            pairs.append(joined)

    summary = {
        "n": len(all_joined),
        "full_acc": sum(r["full_correct"] for r in all_joined) / max(len(all_joined), 1),
        "low_acc": sum(r["low_correct"] for r in all_joined) / max(len(all_joined), 1),
        "n_pairs": len(pairs),
        "full_paths": full_paths,
        "low_paths": low_paths,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    json.dump(all_joined, open(os.path.join(args.out_dir, "all_joined.json"), "w"))
    json.dump(pairs, open(os.path.join(args.out_dir, "pairs.json"), "w"))
    if all_joined:
        with open(os.path.join(args.out_dir, "all_joined.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_joined[0].keys()))
            w.writeheader()
            w.writerows(all_joined)
    print(json.dumps(summary, indent=2))
    print(f"[merge] pairs={len(pairs)}")


if __name__ == "__main__":
    main()
