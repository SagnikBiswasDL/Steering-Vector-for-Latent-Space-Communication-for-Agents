"""CLI: extract a SEAL steering vector and save it as an artifact.

Example:
  python scripts/extract_seal_vector.py \
      --model_name Qwen/Qwen3-14B --task gsm8k --split train \
      --layer_index 28 --max_traces 150 --max_new_tokens 1024 \
      --out artifacts/seal_vectors/qwen3-14b/gsm8k_layer28.pt
"""

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (  # noqa: E402
    load_gsm8k,
    load_arc_challenge,
    load_medqa,
)
from prompts import build_agent_message_sequential_latent_mas  # noqa: E402
from seal.extraction import extract_seal_vector  # noqa: E402


def load_questions(task: str, split: str, limit: int):
    if task == "gsm8k":
        it = load_gsm8k(split=split)
    elif task == "arc_challenge":
        it = load_arc_challenge(split=split)
    elif task == "medqa":
        it = load_medqa(split=split)
    else:
        raise ValueError(f"unsupported task for extraction: {task}")
    questions = []
    for ex in it:
        questions.append(ex["question"])
        if len(questions) >= limit:
            break
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    ap.add_argument("--task", type=str, default="gsm8k")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--layer_index", type=int, default=28)
    ap.add_argument("--max_traces", type=int, default=150)
    ap.add_argument("--scan_limit", type=int, default=400,
                    help="max questions to scan to collect max_traces")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--role", type=str, default=None,
                    choices=[None, "planner", "critic", "refiner", "judger"],
                    help="If set, generate traces with this LatentMAS role's prompt "
                         "so the vector reflects that role's reasoning style.")
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

    questions = load_questions(args.task, args.split, args.scan_limit)
    print(f"[extract] loaded {len(questions)} candidate questions from {args.task}/{args.split}")

    message_builder = None
    if args.role is not None:
        prompt_args = SimpleNamespace(model_name=args.model_name, task=args.task)
        message_builder = lambda q: build_agent_message_sequential_latent_mas(  # noqa: E731
            role=args.role, question=q, context="", method="latent_mas", args=prompt_args
        )
        print(f"[extract] using role-specific prompt: {args.role}")

    result = extract_seal_vector(
        model,
        tokenizer,
        questions,
        layer_index=args.layer_index,
        device=device,
        max_new_tokens=args.max_new_tokens,
        max_traces=args.max_traces,
        temperature=args.temperature,
        top_p=args.top_p,
        message_builder=message_builder,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = {
        "vector": result["vector"],
        "unit_vector": result["unit_vector"],
        "raw_norm": result["raw_norm"],
        "layer_index": result["layer_index"],
        "counts": result["counts"],
        "n_traces": result["n_traces"],
        "model_name": args.model_name,
        "task": args.task,
        "split": args.split,
        "role": args.role,
    }
    torch.save(blob, args.out)
    print(f"[extract] saved -> {args.out}")
    print(f"[extract] step counts: {result['counts']} ; raw_norm={float(result['raw_norm']):.4f}")


if __name__ == "__main__":
    main()
