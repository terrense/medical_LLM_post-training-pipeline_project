"""Multi-turn doctor<->patient episode runner (the shape OpenRLHF's `agent_func_path`
expects: reset -> step -> reward/done -> repeat, see
artifacts/audits/openrlhf_commit_notes.md). The doctor policy is injected too, so this
whole loop is testable with stub callables and no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cmedalign.agents.patient import PatientAgent, build_doctor_visible_messages
from cmedalign.data.schemas import Message, PatientProfile

FINAL_MARKER = "FINAL:"


class DoctorPolicy(Protocol):
    def __call__(self, visible_messages: list[dict]) -> str: ...


@dataclass
class EpisodeResult:
    history: list[Message]
    num_turns: int
    terminated_reason: str  # "final_marker" | "turn_budget" | "invalid"
    valid: bool
    invalid_reasons: list[str] = field(default_factory=list)
    revealed_info: set[str] = field(default_factory=set)


def run_episode(
    profile: PatientProfile,
    doctor_policy: DoctorPolicy,
    patient_responder,
    turn_budget: int,
) -> EpisodeResult:
    patient = PatientAgent(profile=profile, responder=patient_responder)
    history: list[Message] = []
    invalid_reasons: list[str] = []

    for turn in range(1, turn_budget + 1):
        visible = build_doctor_visible_messages(history)
        # boundary check: doctor's view must never contain a serialized PatientProfile field
        for key in ("target_assessment", "acceptable_actions", "forbidden_claims", "red_flags"):
            val = getattr(profile, key)
            serialized = str(val)
            if serialized and any(serialized in str(v.get("content", "")) for v in visible):
                invalid_reasons.append(f"doctor view leaked profile field '{key}'")

        doctor_utterance = doctor_policy(visible)
        history.append(Message(role="assistant", content=doctor_utterance))

        if doctor_utterance.strip().startswith(FINAL_MARKER):
            return EpisodeResult(
                history=history,
                num_turns=turn,
                terminated_reason="final_marker",
                valid=not invalid_reasons,
                invalid_reasons=invalid_reasons,
                revealed_info=set(patient.revealed),
            )

        turn_result = patient.answer(doctor_utterance)
        history.append(Message(role="user", content=turn_result.patient_reply))

        if turn_result.leaked_hidden_target:
            invalid_reasons.append(f"turn {turn}: patient leaked hidden target/forbidden claim")
        if turn_result.contradiction_detected:
            invalid_reasons.append(f"turn {turn}: patient contradicted its own profile")

    return EpisodeResult(
        history=history,
        num_turns=turn_budget,
        terminated_reason="turn_budget",
        valid=not invalid_reasons,
        invalid_reasons=invalid_reasons,
        revealed_info=set(patient.revealed),
    )
