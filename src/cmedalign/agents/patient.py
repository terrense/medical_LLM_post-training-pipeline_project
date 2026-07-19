"""Patient simulator boundary logic.

Two boundaries must hold, enforced in code (not just by prompting the LLM nicely and
hoping), because they are exactly what G4's fixtures probe:

1. The doctor policy must NEVER receive the hidden `PatientProfile` object (or any of
   its fields) directly — only prior conversation turns.
2. The patient's own visible profile (what we hand to whatever model or stub plays the
   patient) must exclude the scoring-only fields: `target_assessment`,
   `acceptable_actions`, `forbidden_claims`, and the `required_info` weights. A real
   patient doesn't know their own diagnosis or which of their symptoms are "worth
   points" — only their own lived symptoms/history.

The actual patient responder (a frozen LLM call) is injected via the `PatientResponder`
protocol so this module — and its tests — never need a GPU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from cmedalign.data.schemas import Message, PatientProfile

PATIENT_VISIBLE_FIELDS = (
    "chief_complaint",
    "demographics",
    "symptoms",
    "time_course",
    "history",
    "medications",
    "allergies",
    "tests",
)
SCORING_ONLY_FIELDS = (
    "target_assessment",
    "acceptable_actions",
    "forbidden_claims",
    "required_info",
    "red_flags",
    "source_train_id",
    "profile_id",
)


def build_patient_visible_profile(profile: PatientProfile) -> dict:
    """What the patient (real or stub) is allowed to see/know about themself."""
    full = profile.model_dump()
    visible = {k: full[k] for k in PATIENT_VISIBLE_FIELDS}
    for forbidden in SCORING_ONLY_FIELDS:
        assert forbidden not in visible, f"scoring-only field '{forbidden}' leaked into patient view"
    return visible


def build_doctor_visible_messages(history: list[Message]) -> list[dict]:
    """What the doctor policy is allowed to see: prior turns only, never the profile."""
    return [{"role": m.role, "content": m.content} for m in history]


class PatientResponder(Protocol):
    def __call__(self, visible_profile: dict, doctor_question: str, revealed_so_far: frozenset[str]) -> str: ...


@dataclass
class TurnResult:
    patient_reply: str
    newly_revealed: set[str]
    leaked_hidden_target: bool
    contradiction_detected: bool


def _mentions(text: str, needle: str) -> bool:
    return bool(needle) and needle.strip().lower() in text.lower()


@dataclass
class PatientAgent:
    profile: PatientProfile
    responder: PatientResponder
    revealed: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.visible_profile = build_patient_visible_profile(self.profile)

    def _check_leak(self, reply: str) -> bool:
        if _mentions(reply, self.profile.target_assessment):
            return True
        return any(_mentions(reply, fc) for fc in self.profile.forbidden_claims)

    def _check_consistency_against_profile(self, reply: str) -> bool:
        """Heuristic ground-truth check: if the reply states a numeric age different
        from the profile's own demographics, that's a self-contradiction."""
        age = self.profile.demographics.get("age") if isinstance(self.profile.demographics, dict) else None
        if age is None:
            return False
        stated_ages = [int(n) for n in re.findall(r"(\d{1,3})\s*岁", reply)]
        stated_ages += [int(n) for n in re.findall(r"\b(\d{1,3})\s*(?:years?\s*old|y/?o)\b", reply, re.IGNORECASE)]
        return any(a != int(age) for a in stated_ages)

    def _detect_revealed(self, reply: str) -> set[str]:
        newly: set[str] = set()
        for item in self.profile.required_info:
            if item.item not in self.revealed and _mentions(reply, item.item):
                newly.add(item.item)
        return newly

    def answer(self, doctor_question: str) -> TurnResult:
        reply = self.responder(self.visible_profile, doctor_question, frozenset(self.revealed))
        leaked = self._check_leak(reply)
        contradiction = self._check_consistency_against_profile(reply)
        newly_revealed = self._detect_revealed(reply)
        self.revealed |= newly_revealed
        return TurnResult(reply, newly_revealed, leaked, contradiction)
