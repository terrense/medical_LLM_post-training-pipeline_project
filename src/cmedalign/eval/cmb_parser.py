"""Deterministic multiple-choice answer extraction for CMB-Exam style items.

G3 requires this to be 100% correct on hand-built fixtures before it touches any real
model output. Ambiguous input must return status="unparseable", never a guess — a wrong
silent guess is worse than a flagged failure, because it corrupts the score table instead
of just showing up as a retry/skip.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal["ok", "unparseable"]

_VALID_LETTERS = "ABCDE"

# Ordered by specificity/confidence — first pattern that matches wins.
_EXPLICIT_PATTERNS = [
    re.compile(r"\\boxed\{([A-E]+)\}"),
    re.compile(r"(?:最终)?答案(?:是|为|：|:)\s*\(?([A-E]+)\)?"),
    re.compile(r"正确答案(?:是|为|：|:)?\s*\(?([A-E]+)\)?"),
    re.compile(r"answer(?:\s+is)?\s*(?:[:：])?\s*\(?([A-E]+)\)?", re.IGNORECASE),
    re.compile(r"选\s*\(?([A-E]+)\)?\s*(?:项|选项)?"),
]

_LEADING_OPTION_PATTERN = re.compile(r"^\s*\(?([A-E])\)?[\.\、\s]")


@dataclass(frozen=True)
class ParsedChoice:
    status: ParseStatus
    letters: tuple[str, ...]
    matched_pattern: str | None
    raw_text: str


def parse_choice(text: str, allow_multi: bool = False) -> ParsedChoice:
    stripped = text.strip()
    if not stripped:
        return ParsedChoice("unparseable", (), None, text)

    candidates: list[tuple[str, str]] = []  # (pattern_name, letters_str)

    for pat in _EXPLICIT_PATTERNS:
        m = pat.search(stripped)
        if m:
            candidates.append((pat.pattern, m.group(1).upper()))

    if not candidates:
        m = _LEADING_OPTION_PATTERN.match(stripped)
        if m:
            candidates.append((_LEADING_OPTION_PATTERN.pattern, m.group(1).upper()))

    if not candidates:
        return ParsedChoice("unparseable", (), None, text)

    # If multiple explicit patterns fired and *disagree*, refuse to guess.
    distinct = {c[1] for c in candidates}
    if len(distinct) > 1:
        return ParsedChoice("unparseable", (), None, text)

    letters_str = candidates[0][1]
    letters = tuple(sorted(set(letters_str)))
    if not all(c in _VALID_LETTERS for c in letters_str):
        return ParsedChoice("unparseable", (), None, text)
    if not allow_multi and len(letters) > 1:
        return ParsedChoice("unparseable", (), None, text)

    return ParsedChoice("ok", letters, candidates[0][0], text)


def is_correct(parsed: ParsedChoice, gold: str) -> bool:
    gold_letters = tuple(sorted(set(gold.upper())))
    return parsed.status == "ok" and parsed.letters == gold_letters
