"""Offline extraction of the SEAL steering vector.

Pipeline (training-free, ~100-200 samples):
  1. For each question, prompt the model to reason step-by-step and generate a
     CoT trace.
  2. Forward the (prompt + trace) once with output_hidden_states to read the
     residual stream at the target layer.
  3. Split the trace into steps, classify each as execution/reflection/
     transition, and bucket the hidden state at each step's last token.
  4. Build v = mean(execution) - mean(reflection + transition) at the target
     layer.

This module is backend-agnostic about *where* the vector is later applied; here
we only read activations of natural text reasoning.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .thought_classifier import classify_step, split_into_steps
from .vector_generation import build_steering_vector


def _step_last_token_indices(tokenizer, steps: List[str], gen_offset: int) -> List[int]:
    """Approximate absolute token index of each step's last token.

    We tokenize steps incrementally; tokenization is not perfectly additive
    across boundaries but is accurate enough for an average over many steps.
    """
    indices: List[int] = []
    cursor = gen_offset
    for step in steps:
        n = len(tokenizer(step, add_special_tokens=False)["input_ids"])
        n = max(n, 1)
        cursor += n
        indices.append(cursor - 1)
    return indices


@torch.no_grad()
def extract_seal_vector(
    model,
    tokenizer,
    questions: List[str],
    *,
    layer_index: int,
    device: torch.device,
    max_new_tokens: int = 1024,
    max_traces: int = 150,
    temperature: float = 0.6,
    top_p: float = 0.95,
    reasoning_instruction: Optional[str] = None,
    message_builder=None,
    log_every: int = 10,
) -> Dict[str, object]:
    """Generate CoT traces and build the steering vector at `layer_index`.

    message_builder: optional callable(question) -> list[chat messages]. When
    provided (e.g. a role-specific LatentMAS prompt) it overrides the default
    generic reasoning prompt, so the vector reflects that role's reasoning style.
    """
    if reasoning_instruction is None:
        reasoning_instruction = (
            "You are a helpful assistant. Reason step by step to solve the "
            "problem, then give the final answer inside \\boxed{}."
        )

    hidden_by_type: Dict[str, List[torch.Tensor]] = {
        "execution": [],
        "reflection": [],
        "transition": [],
    }
    n_traces = 0

    for qi, question in enumerate(questions):
        if n_traces >= max_traces:
            break
        if message_builder is not None:
            messages = message_builder(question)
        else:
            messages = [
                {"role": "system", "content": reasoning_instruction},
                {"role": "user", "content": question},
            ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        prompt_len = int(attn.sum().item())

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        full_ids = gen[0]
        gen_ids = full_ids[prompt_len:]
        trace_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        steps = split_into_steps(trace_text)
        if not steps:
            continue

        # Forward the full sequence once to read hidden states at the layer.
        full_ids_2d = full_ids.unsqueeze(0).to(device)
        out = model(
            input_ids=full_ids_2d,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states is a tuple of (num_layers + 1) tensors [1, T, D];
        # index 0 is the embedding output, so layer L -> hidden_states[L + 1].
        hs = out.hidden_states[layer_index + 1][0]  # [T, D]
        T = hs.shape[0]

        idxs = _step_last_token_indices(tokenizer, steps, gen_offset=prompt_len)
        for step, tok_idx in zip(steps, idxs):
            if tok_idx >= T:
                tok_idx = T - 1
            label = classify_step(step)
            hidden_by_type[label].append(hs[tok_idx].detach().cpu())

        n_traces += 1
        if log_every and (qi + 1) % log_every == 0:
            c = {k: len(v) for k, v in hidden_by_type.items()}
            print(f"[extract] q={qi+1} traces={n_traces} step_counts={c}", flush=True)

    result = build_steering_vector(hidden_by_type, normalize=True)
    result["layer_index"] = int(layer_index)
    result["n_traces"] = n_traces
    return result
