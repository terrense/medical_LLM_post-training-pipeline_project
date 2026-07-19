import dataclasses

import pytest

from cmedalign.rewards.components import (
    REWARD_LOWER_BOUND,
    REWARD_UPPER_BOUND,
    TURN_COST_CAP,
    EpisodeFeatures,
    total_reward,
)

PERFECT = EpisodeFeatures(
    clinical_correctness=1.0,
    info_coverage=1.0,
    safety_score=1.0,
    process_score=1.0,
    communication_score=1.0,
    num_turns=5,
    turn_budget=8,
    response_length_tokens=200,
    length_budget=400,
)

FIXTURES = {
    "perfect_workup": PERFECT,
    "premature_closure": dataclasses.replace(
        PERFECT, clinical_correctness=0.4, info_coverage=0.3, process_score=0.3
    ),
    "repetitive_questions": dataclasses.replace(PERFECT, process_score=0.2, info_coverage=0.5),
    "missed_red_flag": dataclasses.replace(PERFECT, red_flag_missed=True),
    "dangerous_medication_advice": dataclasses.replace(PERFECT, safety_score=0.1),
    "over_refusal": dataclasses.replace(
        PERFECT, clinical_correctness=0.2, info_coverage=0.1, communication_score=0.3
    ),
    "long_but_empty": dataclasses.replace(
        PERFECT, info_coverage=0.2, response_length_tokens=1200
    ),
    "short_missing_info": dataclasses.replace(
        PERFECT, info_coverage=0.3, response_length_tokens=50
    ),
    "role_leakage": dataclasses.replace(PERFECT, role_leakage=True),
    "verbose_but_thorough": dataclasses.replace(PERFECT, response_length_tokens=700),
    "too_many_turns": dataclasses.replace(PERFECT, num_turns=60),
    "everything_bad": EpisodeFeatures(
        clinical_correctness=0.0,
        info_coverage=0.0,
        safety_score=0.0,
        process_score=0.0,
        communication_score=0.0,
        num_turns=100,
        turn_budget=8,
        response_length_tokens=5000,
        length_budget=400,
        red_flag_missed=True,
        role_leakage=True,
    ),
}

SCORES = {name: total_reward(f) for name, f in FIXTURES.items()}


def test_perfect_workup_scores_highest():
    assert SCORES["perfect_workup"] == max(SCORES.values())


def test_everything_bad_scores_lowest():
    assert SCORES["everything_bad"] == min(SCORES.values())


def test_expected_ranking_matches_intent():
    # perfect > mildly-imperfect-but-safe variants > over_refusal > missed_red_flag /
    # role_leakage / everything_bad (the flat-penalty failure modes).
    assert SCORES["perfect_workup"] > SCORES["verbose_but_thorough"]
    assert SCORES["verbose_but_thorough"] > SCORES["too_many_turns"]
    assert SCORES["too_many_turns"] > SCORES["premature_closure"]
    assert SCORES["premature_closure"] > SCORES["over_refusal"]
    assert SCORES["over_refusal"] > SCORES["missed_red_flag"]
    assert SCORES["over_refusal"] > SCORES["role_leakage"]
    assert SCORES["missed_red_flag"] > SCORES["everything_bad"]
    assert SCORES["role_leakage"] > SCORES["everything_bad"]


@pytest.mark.parametrize(
    "field",
    ["clinical_correctness", "info_coverage", "safety_score", "process_score", "communication_score"],
)
def test_component_monotonicity_increasing_positive_score_never_decreases_reward(field):
    low = dataclasses.replace(PERFECT, **{field: 0.1})
    high = dataclasses.replace(PERFECT, **{field: 0.9})
    assert total_reward(high) >= total_reward(low)


def test_turn_overage_monotonically_non_increasing():
    within_budget = dataclasses.replace(PERFECT, num_turns=5)
    slightly_over = dataclasses.replace(PERFECT, num_turns=10)
    way_over = dataclasses.replace(PERFECT, num_turns=50)
    assert total_reward(within_budget) >= total_reward(slightly_over) >= total_reward(way_over)


def test_length_overage_monotonically_non_increasing():
    within_budget = dataclasses.replace(PERFECT, response_length_tokens=200)
    slightly_over = dataclasses.replace(PERFECT, response_length_tokens=500)
    way_over = dataclasses.replace(PERFECT, response_length_tokens=2000)
    assert (
        total_reward(within_budget) >= total_reward(slightly_over) >= total_reward(way_over)
    )


def test_red_flag_penalty_dominates_turn_cost():
    """A missed red flag must hurt more than even the maximum possible turn-budget
    overrun -- otherwise a policy could 'buy back' a dangerous miss by being fast."""
    perfect_but_very_late = dataclasses.replace(PERFECT, num_turns=1000)  # turn cost capped
    perfect_but_missed_flag = dataclasses.replace(PERFECT, red_flag_missed=True, num_turns=5)

    assert total_reward(perfect_but_missed_flag) < total_reward(perfect_but_very_late)
    # explicitly: missing the flag must cost more than the turn-cost cap alone
    worst_case_pure_turn_cost = total_reward(PERFECT) - TURN_COST_CAP
    assert total_reward(perfect_but_missed_flag) < worst_case_pure_turn_cost


def test_reward_bounded_for_extreme_and_out_of_range_inputs():
    extreme = EpisodeFeatures(
        clinical_correctness=5.0,  # out-of-range input, must be clamped not explode
        info_coverage=-3.0,
        safety_score=2.0,
        process_score=-1.0,
        communication_score=10.0,
        num_turns=100000,
        turn_budget=1,
        response_length_tokens=10_000_000,
        length_budget=1,
        red_flag_missed=True,
        role_leakage=True,
    )
    r = total_reward(extreme)
    assert REWARD_LOWER_BOUND <= r <= REWARD_UPPER_BOUND


def test_all_fixtures_bounded():
    for name, score in SCORES.items():
        assert REWARD_LOWER_BOUND <= score <= REWARD_UPPER_BOUND, name
