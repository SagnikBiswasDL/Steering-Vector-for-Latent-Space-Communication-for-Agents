"""Offline extraction of one-shot KV-cache steering vectors.

Paper-faithful (arXiv:2507.08799, section 3.2): build a contrastive set of
prompt pairs that differ only in whether the in-context examples contain
explicit chain-of-thought reasoning, forward each prompt once with
``use_cache=True``, read the cached key/value tensors at the final prompt token
per layer, and aggregate with Mean-of-Differences:

    S_k_l = mean_pairs( K_l(p+) - K_l(p-) )   at the final-token position
    S_v_l = mean_pairs( V_l(p+) - V_l(p-) )

The positive prompt shows worked CoT solutions in its few-shot examples; the
negative prompt shows the same examples with only the final answer. Both share
the identical target question and generation prompt, so the difference isolates
"about to reason step by step" vs. "about to just answer" in the KV space.

No teacher model is needed: GSM8K's training solutions already contain worked
step-by-step reasoning (see ``data.load_gsm8k``, field ``solution``).
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

import torch

from .kv_steer import iter_layer_kv

Messages = List[Dict[str, str]]
Pair = Tuple[Messages, Messages]

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def _strip_gsm8k_solution(solution: str) -> str:
    """Return the reasoning portion of a GSM8K solution (drop the '#### X' tail)."""
    text = re.split(r"####", solution)[0].strip()
    # GSM8K reasoning contains <<...>> calculator annotations; remove them.
    text = re.sub(r"<<[^>]*>>", "", text)
    return text.strip()


def build_contrastive_pairs(
    targets: List[Dict],
    icl_pool: List[Dict],
    *,
    n_pairs: int,
    n_icl: int,
    seed: int = 0,
    system_message: str = DEFAULT_SYSTEM,
) -> List[Pair]:
    """Construct (positive, negative) chat-message pairs.

    Each element of ``targets`` / ``icl_pool`` is a dict with keys
    ``question``, ``solution`` (worked CoT), and ``gold`` (final answer).
    Positive and negative prompts share the same target question and the same
    few-shot examples; they differ only in whether those examples show the CoT.
    """
    rng = random.Random(seed)
    pairs: List[Pair] = []
    pool = list(icl_pool)
    for target in targets:
        if len(pairs) >= n_pairs:
            break
        if not pool:
            break
        shots = rng.sample(pool, k=min(n_icl, len(pool)))

        pos: Messages = [{"role": "system", "content": system_message}]
        neg: Messages = [{"role": "system", "content": system_message}]
        for shot in shots:
            q = shot["question"].strip()
            cot = _strip_gsm8k_solution(shot["solution"])
            gold = str(shot["gold"]).strip()
            pos.append({"role": "user", "content": q})
            pos.append(
                {"role": "assistant", "content": f"{cot}\nThe final answer is {gold}."}
            )
            neg.append({"role": "user", "content": q})
            neg.append({"role": "assistant", "content": f"The final answer is {gold}."})

        tq = target["question"].strip()
        pos.append({"role": "user", "content": tq})
        neg.append({"role": "user", "content": tq})
        pairs.append((pos, neg))
    return pairs


@torch.no_grad()
def _read_final_token_kv(
    model, tokenizer, messages: Messages, device
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Forward a chat prompt once and return per-layer (K, V) at the last token.

    Each returned tensor has shape ``[H_kv, D_h]`` (fp32, CPU).
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        return_dict=True,
    )
    cache = out.past_key_values
    layer_kv = iter_layer_kv(cache)
    result: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for K, V in layer_kv:
        # [B=1, H_kv, T, D_h] -> [H_kv, D_h] at the final position
        result.append(
            (
                K[0, :, -1, :].detach().float().cpu(),
                V[0, :, -1, :].detach().float().cpu(),
            )
        )
    return result


@torch.no_grad()
def extract_kv_steering_vectors(
    model,
    tokenizer,
    pairs: List[Pair],
    *,
    device,
    log_every: int = 25,
) -> Dict[str, object]:
    """Mean-of-Differences of final-token K/V across contrastive pairs."""
    sum_keys: Dict[int, torch.Tensor] = {}
    sum_values: Dict[int, torch.Tensor] = {}
    n = 0
    for i, (pos_msgs, neg_msgs) in enumerate(pairs):
        pos_kv = _read_final_token_kv(model, tokenizer, pos_msgs, device)
        neg_kv = _read_final_token_kv(model, tokenizer, neg_msgs, device)
        if len(pos_kv) != len(neg_kv):
            continue
        for layer_idx, ((pk, pv), (nk, nv)) in enumerate(zip(pos_kv, neg_kv)):
            dk = pk - nk
            dv = pv - nv
            if layer_idx not in sum_keys:
                sum_keys[layer_idx] = torch.zeros_like(dk)
                sum_values[layer_idx] = torch.zeros_like(dv)
            sum_keys[layer_idx] += dk
            sum_values[layer_idx] += dv
        n += 1
        if log_every and (i + 1) % log_every == 0:
            print(f"[kv-extract] processed {i + 1}/{len(pairs)} pairs", flush=True)

    if n == 0:
        raise RuntimeError("No contrastive pairs were processed; got 0 usable pairs.")

    keys = {l: (t / n) for l, t in sum_keys.items()}
    values = {l: (t / n) for l, t in sum_values.items()}
    sample = next(iter(values.values()))
    return {
        "keys": keys,
        "values": values,
        "num_layers": len(values),
        "num_kv_heads": int(sample.shape[0]),
        "head_dim": int(sample.shape[1]),
        "n_pairs": n,
    }
