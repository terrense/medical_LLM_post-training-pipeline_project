"""GRPO reward components for the multi-turn patient-dialogue environment.

Weights are the spec's starting point (clinical 0.30, information 0.25, safety 0.25,
process 0.10, communication 0.10) minus capped turn/length costs. These positive weights
are frozen once training starts — only the cost coefficients may be tuned on
training-only dev data, never chosen by looking at test performance.

Two failure modes must dominate everything else, by construction, not by luck of the
weights: missing a documented red flag, and role leakage / episode-format violations.
Both are modeled as large flat penalties that exceed the maximum possible turn/length
cost, so no amount of "efficiency" can buy back a missed red flag.
"""
from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = {
    "clinical": 0.30,
    "information": 0.25,
    "safety": 0.25,
    "process": 0.10,
    "communication": 0.10,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

TURN_COST_COEF = 0.02  # per turn over budget
TURN_COST_CAP = 0.20
LENGTH_COST_COEF = 0.15  # per fraction-over-budget
LENGTH_COST_CAP = 0.15

RED_FLAG_MISSED_PENALTY = 0.60  # > TURN_COST_CAP + LENGTH_COST_CAP by design
ROLE_LEAKAGE_PENALTY = 0.60

REWARD_LOWER_BOUND = -1.0
REWARD_UPPER_BOUND = 1.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class EpisodeFeatures:
    clinical_correctness: float  # 0-1, from rubric/judge
    info_coverage: float  # 0-1, weighted fraction of required_info items revealed+used
    safety_score: float  # 0-1, rubric safety rating (excludes the red-flag flat penalty below)
    process_score: float  # 0-1, structured/non-repetitive questioning etc.
    communication_score: float  # 0-1, clarity/empathy/actionability
    num_turns: int
    turn_budget: int
    response_length_tokens: int
    length_budget: int
    red_flag_missed: bool = False
    role_leakage: bool = False


def turn_cost(features: EpisodeFeatures) -> float:
    over = max(0, features.num_turns - features.turn_budget)
    return min(TURN_COST_CAP, TURN_COST_COEF * over)


def length_cost(features: EpisodeFeatures) -> float:
    if features.length_budget <= 0:
        return 0.0
    over_frac = max(0.0, (features.response_length_tokens - features.length_budget) / features.length_budget)
    return min(LENGTH_COST_CAP, LENGTH_COST_COEF * over_frac)


def base_reward(features: EpisodeFeatures) -> float:
    return (
        WEIGHTS["clinical"] * _clamp01(features.clinical_correctness)
        + WEIGHTS["information"] * _clamp01(features.info_coverage)
        + WEIGHTS["safety"] * _clamp01(features.safety_score)
        + WEIGHTS["process"] * _clamp01(features.process_score)
        + WEIGHTS["communication"] * _clamp01(features.communication_score)
    )


def total_reward(features: EpisodeFeatures) -> float:
    reward = base_reward(features)
    reward -= turn_cost(features)
    reward -= length_cost(features)
    if features.red_flag_missed:
        reward -= RED_FLAG_MISSED_PENALTY
    if features.role_leakage:
        reward -= ROLE_LEAKAGE_PENALTY
    return max(REWARD_LOWER_BOUND, min(REWARD_UPPER_BOUND, reward))
