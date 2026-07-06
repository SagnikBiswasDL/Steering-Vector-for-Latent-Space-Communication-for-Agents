from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Agent:
    name: str
    role: str


# Canonical LatentMAS role order; the Judger (text emitter) is always last.
CANONICAL_ROLES = ["planner", "critic", "refiner", "judger"]
ROLE_DISPLAY = {"planner": "Planner", "critic": "Critic", "refiner": "Refiner", "judger": "Judger"}


def default_agents() -> List[Agent]:
    return [Agent(name=ROLE_DISPLAY[r], role=r) for r in CANONICAL_ROLES]


def agents_from_spec(spec: Optional[str]) -> List[Agent]:
    """Build the agent chain from a comma-separated role spec (for ablations).

    - ``None``/``""``/``"full"`` -> the default 4-agent chain.
    - Otherwise a subset of {planner, critic, refiner}, e.g. "planner,critic".
    - The Judger is always included and always last; roles are emitted in
      canonical order regardless of input order, so the returned chain is a
      well-formed prefix/subset of Planner -> Critic -> Refiner -> Judger.

    Examples: "judger" -> [Judger]; "planner" -> [Planner, Judger];
    "critic,refiner" -> [Critic, Refiner, Judger] (the no_planner ablation).
    """
    if spec is None or spec.strip() in ("", "full"):
        return default_agents()
    requested = {r.strip().lower() for r in spec.split(",") if r.strip()}
    unknown = requested - set(CANONICAL_ROLES)
    if unknown:
        raise ValueError(f"unknown agent role(s) {sorted(unknown)}; valid: {CANONICAL_ROLES}")
    requested.add("judger")  # the text-emitting Judger is mandatory
    ordered = [r for r in CANONICAL_ROLES if r in requested]
    return [Agent(name=ROLE_DISPLAY[r], role=r) for r in ordered]


__all__ = ["Agent", "default_agents", "agents_from_spec", "CANONICAL_ROLES"]
