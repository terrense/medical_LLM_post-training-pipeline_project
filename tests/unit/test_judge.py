import pytest

from cmedalign.eval.judge import (
    JudgeParseError,
    extract_json_object,
    parse_judge_reply,
    reconcile_dual_order,
)


def test_extract_json_plain():
    assert extract_json_object('{"winner": "A"}') == {"winner": "A"}


def test_extract_json_with_code_fence():
    text = 'Here is my judgment:\n```json\n{"winner": "B", "score_a": 3}\n```\nThanks.'
    assert extract_json_object(text) == {"winner": "B", "score_a": 3}


def test_extract_json_with_leading_prose_no_fence():
    text = 'Sure, my verdict: {"winner": "tie", "score_a": 5, "score_b": 5} — that is final.'
    obj = extract_json_object(text)
    assert obj["winner"] == "tie"


def test_extract_json_malformed_raises():
    with pytest.raises(JudgeParseError):
        extract_json_object("no json here at all")


def test_parse_judge_reply_requires_valid_winner():
    with pytest.raises(JudgeParseError):
        parse_judge_reply('{"winner": "C"}')
    v = parse_judge_reply('{"winner": "A", "score_a": 8, "score_b": 3}')
    assert v.winner == "A"
    assert v.score_a == 8


def test_reconcile_agreement_resp1_wins():
    # First call: A=resp1, B=resp2, judge picks A -> resp1.
    first = parse_judge_reply('{"winner": "A", "score_a": 9, "score_b": 4}')
    # Second call: A=resp2, B=resp1 (flipped), judge again picks resp1, i.e. "B" this time.
    second = parse_judge_reply('{"winner": "B", "score_a": 4, "score_b": 9}')
    result = reconcile_dual_order(first, second)
    assert result.agree is True
    assert result.final_winner == "resp1"
    assert result.margin == 5


def test_reconcile_disagreement_flagged():
    # First call picks resp1 ("A"); second call (flipped) also picks the position "A",
    # i.e. resp2 in resp-space -- this is a position-bias disagreement.
    first = parse_judge_reply('{"winner": "A"}')
    second = parse_judge_reply('{"winner": "A"}')
    result = reconcile_dual_order(first, second)
    assert result.agree is False
    assert result.final_winner is None


def test_reconcile_tie_agreement():
    first = parse_judge_reply('{"winner": "tie"}')
    second = parse_judge_reply('{"winner": "tie"}')
    result = reconcile_dual_order(first, second)
    assert result.agree is True
    assert result.final_winner == "tie"
