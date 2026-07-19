import pytest

from cmedalign.eval.cmb_parser import is_correct, parse_choice

VALID_FIXTURES = [
    ("答案是A", ("A",)),
    ("答案为 C", ("C",)),
    ("正确答案：B", ("B",)),
    ("正确答案是(D)", ("D",)),
    ("The answer is A", ("A",)),
    ("answer: B", ("B",)),
    ("选A", ("A",)),
    ("我认为应该选(C)项", ("C",)),
    ("A. 这是最合适的诊断", ("A",)),
    ("(B) 患者应立即转诊", ("B",)),
    ("经过分析，答案是E", ("E",)),
    ("\\boxed{A}", ("A",)),
]

INVALID_FIXTURES = [
    "",
    "   ",
    "这道题目比较复杂，需要考虑多种因素",  # no letter marker at all
    "答案是A，但正确答案是B",  # disagreeing explicit patterns ("答案是A" vs "正确答案是B")
    "答案是F",  # invalid letter
    "A或者B都有可能，不确定",  # no single explicit resolution, leading pattern won't fire (starts with A but then "或者")
]


@pytest.mark.parametrize("text,expected", VALID_FIXTURES)
def test_valid_formats_parsed_correctly(text, expected):
    result = parse_choice(text)
    assert result.status == "ok"
    assert result.letters == expected


@pytest.mark.parametrize("text", INVALID_FIXTURES)
def test_invalid_formats_flagged_unparseable(text):
    result = parse_choice(text)
    assert result.status == "unparseable"


def test_multi_select_requires_allow_multi_flag():
    text = "答案是AC"
    assert parse_choice(text, allow_multi=False).status == "unparseable"
    result = parse_choice(text, allow_multi=True)
    assert result.status == "ok"
    assert result.letters == ("A", "C")


def test_is_correct_matches_gold_regardless_of_order():
    result = parse_choice("答案是CA", allow_multi=True)
    assert is_correct(result, "AC")
    assert not is_correct(result, "A")


def test_is_correct_false_when_unparseable():
    result = parse_choice("完全无法判断")
    assert not is_correct(result, "A")
