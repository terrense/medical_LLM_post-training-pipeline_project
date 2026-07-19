"""Judge-output parsing and dual-order (position-swap) reconciliation.

G3: "judge 双顺序解析为合法 JSON" — every judge call is issued twice, with the two
candidate responses in both orders, specifically to catch position bias. This module
parses each raw judge reply into a normalized verdict and reconciles the pair.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional

Winner = Literal["A", "B", "tie"]

_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class JudgeParseError(ValueError):
    pass


def extract_json_object(text: str) -> dict:
    """Pulls the first well-formed JSON object out of a judge reply, tolerating a
    ```json ... ``` fence and/or leading/trailing prose the judge model added."""
    fenced = _CODE_FENCE.search(text)
    candidate = fenced.group(1) if fenced else text

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first balanced {...} span.
    start = candidate.find("{")
    if start == -1:
        raise JudgeParseError(f"no JSON object found in judge output: {text!r}")
    depth = 0
    for i in range(start, len(candidate)):
        if candidate[i] == "{":
            depth += 1
        elif candidate[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[start : i + 1])
                except json.JSONDecodeError as e:
                    raise JudgeParseError(f"malformed JSON object in judge output: {text!r}") from e
    raise JudgeParseError(f"unbalanced JSON object in judge output: {text!r}")


@dataclass(frozen=True)
class JudgeVerdict:
    winner: Winner
    score_a: Optional[float]
    score_b: Optional[float]
    raw: dict


def parse_judge_reply(text: str) -> JudgeVerdict:
    obj = extract_json_object(text)
    winner = obj.get("winner")
    if winner not in ("A", "B", "tie"):
        raise JudgeParseError(f"judge JSON missing valid 'winner' field: {obj!r}")
    return JudgeVerdict(
        winner=winner,
        score_a=obj.get("score_a"),
        score_b=obj.get("score_b"),
        raw=obj,
    )


@dataclass(frozen=True)
class Reconciled:
    agree: bool
    # Winner expressed in terms of the *original* (pre-flip) response identity:
    # "resp1", "resp2", "tie", or None if the two orderings disagreed.
    final_winner: Optional[Literal["resp1", "resp2", "tie"]]
    margin: Optional[float]


def reconcile_dual_order(first_call: JudgeVerdict, second_call_flipped: JudgeVerdict) -> Reconciled:
    """`first_call` judged (A=resp1, B=resp2). `second_call_flipped` judged the same pair
    with (A=resp2, B=resp1). Returns the reconciled, position-bias-checked verdict."""
    first_in_resp_space: Literal["resp1", "resp2", "tie"] = {
        "A": "resp1",
        "B": "resp2",
        "tie": "tie",
    }[first_call.winner]
    second_in_resp_space: Literal["resp1", "resp2", "tie"] = {
        "A": "resp2",
        "B": "resp1",
        "tie": "tie",
    }[second_call_flipped.winner]

    agree = first_in_resp_space == second_in_resp_space

    margin = None
    if (
        first_call.score_a is not None
        and first_call.score_b is not None
        and second_call_flipped.score_a is not None
        and second_call_flipped.score_b is not None
    ):
        margin_first = abs(first_call.score_a - first_call.score_b)
        margin_second = abs(second_call_flipped.score_a - second_call_flipped.score_b)
        margin = min(margin_first, margin_second)

    return Reconciled(
        agree=agree,
        final_winner=first_in_resp_space if agree else None,
        margin=margin,
    )
