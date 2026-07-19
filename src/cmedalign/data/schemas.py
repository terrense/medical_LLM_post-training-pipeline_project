"""Canonical record schemas for cmedalign.

These mirror the exact field lists in the execution spec (`RESULTS_AND_TABLE_SCHEMA.md`
equivalents live in `results/` on the real run; this module is the code-level source of
truth for what a valid record looks like before anything is trained or evaluated).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str

    @field_validator("content")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message content must not be empty/whitespace-only")
        return v


class SyntheticProvenance(BaseModel):
    """Required whenever a record is synthetic rather than a real recorded
    conversation (per the paper's non-negotiable reporting rules and
    cmedalign_paper/SOURCE_AUDIT.md §6b's "seed-flywheel" data source): who
    generated it, when, and whether/by whom it was reviewed. Never merge a
    synthetic record's provenance indistinguishably into a real-data source."""

    teacher_model: str  # e.g. "DeepSeek-V4-Pro" (patient role) — record the exact model, not a paper alias
    doctor_model: Optional[str] = None  # e.g. "Qwen3-8B" if the base model generated the doctor turns
    generation_date: Optional[str] = None  # ISO8601
    reviewed_by: list[str] = Field(default_factory=list)  # e.g. ["stronger_reviewing_model", "clinical_staff"]
    audit_status: Literal["unreviewed", "model_reviewed", "clinically_audited"] = "unreviewed"


class ConversationRecord(BaseModel):
    sample_id: str
    source: str
    source_revision: str
    split: Literal["train", "dev", "test"]
    messages: list[Message]
    specialty: Optional[str] = None  # medical department, e.g. 内科/外科 -- distinct from task_type below
    task_type: Optional[str] = None  # e.g. symptom_consultation, triage_guidance -- see spec/design.md task_type taxonomy
    quality_score: Optional[float] = None
    safety_tags: list[str] = Field(default_factory=list)
    provenance: str
    license_id: str
    raw_hash: str
    normalized_hash: str
    synthetic_or_real: Literal["real", "synthetic"] = "real"
    synthetic_provenance: Optional[SyntheticProvenance] = None

    @model_validator(mode="after")
    def synthetic_records_need_provenance(self):
        if self.synthetic_or_real == "synthetic" and self.synthetic_provenance is None:
            raise ValueError(
                "synthetic_or_real='synthetic' requires synthetic_provenance to be set "
                "(teacher_model at minimum) — do not silently drop synthetic-data provenance"
            )
        return self

    @field_validator("messages")
    @classmethod
    def at_least_one_turn(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages must be non-empty")
        for a, b in zip(v, v[1:]):
            if a.role == b.role:
                raise ValueError(
                    f"adjacent messages must not share the same role "
                    f"(found consecutive '{a.role}' turns) — merge or fix upstream, "
                    "do not silently drop one"
                )
        return v


class RequiredInfoItem(BaseModel):
    item: str
    weight: float


class PatientProfile(BaseModel):
    profile_id: str
    chief_complaint: str
    demographics: dict
    symptoms: list[str]
    time_course: str
    history: str
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    tests: dict = Field(default_factory=dict)
    red_flags: list[str] = Field(default_factory=list)
    required_info: list[RequiredInfoItem]
    target_assessment: str
    acceptable_actions: list[str]
    forbidden_claims: list[str] = Field(default_factory=list)
    source_train_id: str


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class RawGeneration(BaseModel):
    """One raw model output. Never mutate/overwrite these after write — aggregate
    results must always be traceable back to a file of these."""

    model_id: str
    model_revision: Optional[str] = None
    accessed_at: Optional[str] = None  # ISO8601, required for API models
    dataset: str
    split: Literal["train", "dev", "test"]
    subset: Optional[str] = None
    item_id: str
    prompt_hash: str
    messages: list[Message]
    seed: Optional[int] = None
    decoding: dict
    enable_thinking: Optional[bool] = None
    raw_text: str
    finish_reason: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: Optional[float] = None
    parser_status: Literal["ok", "unparseable", "retried"] = "ok"
    run_id: str
    commit: str
