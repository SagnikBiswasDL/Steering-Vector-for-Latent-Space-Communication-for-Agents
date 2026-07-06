"""CLI: extract one-shot KV-cache steering vectors and save an artifact.

Builds CoT-rich vs. answer-only few-shot contrastive pairs from GSM8K-train
solutions, reads per-layer cached (K, V) at the final prompt token, aggregates
with Mean-of-Differences, and saves {keys, values, meta}.

Reference: "KV Cache Steering for Controlling Frozen LLMs", arXiv:2507.08799.

Example:
  python scripts/extract_kv_steer_vector.py \
      --model_name Qwen/Qwen3-14B --task gsm8k --split train \
      --n_pairs 200 --n_icl 2 \
      --out artifacts/kv_steer_vectors/qwen3-14b/gsm8k_cot.pt
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_gsm8k  # noqa: E402
from seal.kv_extraction import (  # noqa: E402
    build_contrastive_pairs,
    extract_kv_steering_vectors,
)


def load_examples(task: str, split: str, limit: int):
    if task != "gsm8k":
        raise ValueError(
            f"unsupported task for KV extraction: {task} (only gsm8k has worked solutions)"
        )
    out = []
    for ex in load_gsm8k(split=split):
        out.append(ex)
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--task", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n_pairs", type=int, default=200,
                    help="number of contrastive pairs to aggregate")
    ap.add_argument("--n_icl", type=int, default=2,
                    help="few-shot examples per contrastive prompt")
    ap.add_argument("--icl_pool_size", type=int, default=32,
                    help="size of the disjoint pool ICL examples are drawn from")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
    ).to(device).eval()

    # Disjoint pools: ICL examples are drawn from the head of the split; targets
    # come after, so a target question never appears among its own few-shot set.
    need = args.icl_pool_size + args.n_pairs
    examples = load_examples(args.task, args.split, need)
    if len(examples) < args.icl_pool_size + 1:
        raise RuntimeError(
            f"not enough examples ({len(examples)}) for icl_pool_size={args.icl_pool_size}"
        )
    icl_pool = examples[: args.icl_pool_size]
    targets = examples[args.icl_pool_size:]
    print(
        f"[kv-extract] loaded {len(examples)} examples "
        f"(icl_pool={len(icl_pool)}, targets={len(targets)}) from {args.task}/{args.split}"
    )

    pairs = build_contrastive_pairs(
        targets,
        icl_pool,
        n_pairs=args.n_pairs,
        n_icl=args.n_icl,
        seed=args.seed,
    )
    print(f"[kv-extract] built {len(pairs)} contrastive pairs (n_icl={args.n_icl})")

    result = extract_kv_steering_vectors(model, tokenizer, pairs, device=device)

    cfg = model.config
    v_norms = {l: float(t.norm().item()) for l, t in result["values"].items()}
    mean_v_norm = sum(v_norms.values()) / max(1, len(v_norms))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = {
        "keys": result["keys"],
        "values": result["values"],
        "num_layers": result["num_layers"],
        "num_kv_heads": result["num_kv_heads"],
        "head_dim": result["head_dim"],
        "n_pairs": result["n_pairs"],
        "model_name": args.model_name,
        "task": args.task,
        "split": args.split,
        "n_icl": args.n_icl,
        "seed": args.seed,
        "config_num_hidden_layers": int(getattr(cfg, "num_hidden_layers", -1)),
        "config_num_kv_heads": int(getattr(cfg, "num_key_value_heads", -1)),
    }
    torch.save(blob, args.out)
    print(f"[kv-extract] saved -> {args.out}")
    print(
        f"[kv-extract] layers={result['num_layers']} kv_heads={result['num_kv_heads']} "
        f"head_dim={result['head_dim']} n_pairs={result['n_pairs']} "
        f"mean_value_dir_norm={mean_v_norm:.4f}"
    )
    # Shape sanity vs. model config.
    if blob["config_num_hidden_layers"] not in (-1, result["num_layers"]):
        print(
            f"[kv-extract][warn] layer count {result['num_layers']} != config "
            f"num_hidden_layers {blob['config_num_hidden_layers']}"
        )
    if blob["config_num_kv_heads"] not in (-1, result["num_kv_heads"]):
        print(
            f"[kv-extract][warn] kv head count {result['num_kv_heads']} != config "
            f"num_key_value_heads {blob['config_num_kv_heads']}"
        )


if __name__ == "__main__":
    main()
