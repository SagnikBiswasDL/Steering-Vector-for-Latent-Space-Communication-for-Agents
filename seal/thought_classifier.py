"""Heuristic classifier for reasoning-step thought types.

SEAL categorizes chain-of-thought steps into three types:
  - execution:  carrying the solution forward (calculations, deductions)
  - reflection: re-checking / doubting / verifying prior work
  - transition: switching strategy or starting a different approach

Excessive reflection + transition correlates with over-reasoning and token
waste. We use a lightweight keyword heuristic (no extra model calls), which is
sufficient because the steering vector is an average over hundreds of steps.
"""

from __future__ import annotations

import re
from typing import List

THOUGHT_TYPES = ("execution", "reflection", "transition")

# Cues that signal the model is re-examining / second-guessing its own work.
_REFLECTION_CUES = (
    "wait",
    "hold on",
    "hmm",
    "actually",
    "let me check",
    "let me verify",
    "let me re-check",
    "let me recheck",
    "double-check",
    "double check",
    "to confirm",
    "make sure",
    "is this correct",
    "is that correct",
    "that's wrong",
    "thats wrong",
    "i made a mistake",
    "i think i made",
    "let me reconsider",
    "reconsider",
    "but wait",
    "on second thought",
    "recheck",
)

# Cues that signal a switch to a different strategy / restart.
_TRANSITION_CUES = (
    "alternatively",
    "another approach",
    "another way",
    "a different approach",
    "different way",
    "on the other hand",
    "instead",
    "let me try",
    "let's try",
    "what if",
    "we could also",
    "or maybe",
    "or we can",
    "another method",
)


def split_into_steps(text: str) -> List[str]:
    """Split a reasoning trace into atomic steps.

    We split on newlines first (CoT traces are usually line-structured) and then
    fall back to sentence boundaries for long single-line blocks.
    """
    if not text:
        return []
    raw_lines = [ln.strip() for ln in text.split("\n")]
    steps: List[str] = []
    for line in raw_lines:
        if not line:
            continue
        if len(line) > 240:
            # break very long lines into sentences
            parts = re.split(r"(?<=[.!?])\s+", line)
            steps.extend(p.strip() for p in parts if p.strip())
        else:
            steps.append(line)
    return steps


def classify_step(step: str) -> str:
    """Return one of THOUGHT_TYPES for a single reasoning step."""
    s = step.lower()
    # Transition cues take priority (a strategy switch often also reflects),
    # then reflection, else it is forward execution.
    for cue in _TRANSITION_CUES:
        if cue in s:
            return "transition"
    for cue in _REFLECTION_CUES:
        if cue in s:
            return "reflection"
    return "execution"
