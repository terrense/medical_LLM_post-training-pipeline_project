"""Pydantic models matching RESULTS_AND_TABLE_SCHEMA.md's exact schemas, verbatim.
These are the authoritative record types `results/statistics.json`,
`tables/table_main_universal.csv`, `tables/table_stage_ablation.csv`, and
`tables/table_human.csv` must conform to -- not the generic ad-hoc shapes
`cmedalign.stats.tables.build_table_from_item_level` produces on its own (that function
is still a useful bootstrap-CI-computing primitive; these models are the contract its
output gets adapted into before being written to the actual `tables/*.csv` files).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class SFTMethodRow(BaseModel):
    """One row of tables/table_sft_method.csv (added 2026-07-20, per user's decision to
    run LoRA and full-parameter SFT on identical data/seed and carry the stronger one
    forward). Exactly one row across the two must have is_winner=True -- enforced below,
    not left as an unchecked convention."""

    method: Literal["lora", "full_parameter"]
    learning_rate: Optional[float] = None
    effective_batch_size: Optional[int] = None
    gpu_count: Optional[int] = None
    parallelism: Optional[str] = None  # e.g. "none", "zero2", "zero3"
    peak_gpu_memory_gb: Optional[float] = None
    wall_clock_gpu_hours: Optional[float] = None
    dev_composite_score: Optional[float] = None
    is_winner: bool = False
    checkpoint_size_gb: Optional[float] = None


def require_exactly_one_winner(rows: list[SFTMethodRow]) -> None:
    winners = [r for r in rows if r.is_winner]
    if len(winners) != 1:
        raise ValueError(
            f"table_sft_method rows must have exactly one is_winner=True row, found {len(winners)} "
            f"-- selection must be decided from dev_composite_score, never left ambiguous"
        )


class StatisticsEntry(BaseModel):
    """Exact schema from RESULTS_AND_TABLE_SCHEMA.md's "Statistical output contract".
    The paper's own build script "fails when n_cases, estimate, intervals, corrected
    p-value, or subset ID are absent for a primary claim" -- `require_complete()` below
    enforces exactly that rule in code, not just in the doc.
    """

    comparison_id: str
    model_a: str
    model_b: str
    metric: str
    paired: bool
    n_cases: Optional[int] = None
    estimate: Optional[float] = None
    ci_method: str = "case_clustered_bootstrap"
    ci_level: float = 0.95
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_raw: Optional[float] = None
    p_holm: Optional[float] = None
    effect_size: Optional[float] = None
    subset_id: Optional[str] = None
    seed: int = 20260718

    def require_complete(self) -> None:
        """Raises if this is missing what a *primary* (reported) claim needs. Call this
        before writing a StatisticsEntry into the real statistics.json for a claim that
        will appear in the manuscript -- exploratory/intermediate entries can skip it."""
        missing = [
            f
            for f in ("n_cases", "estimate", "ci_low", "ci_high", "p_holm", "subset_id")
            if getattr(self, f) is None
        ]
        if missing:
            raise ValueError(
                f"StatisticsEntry {self.comparison_id!r} is missing required fields for "
                f"a primary claim: {missing} -- do not write this into the reported "
                "statistics.json until every field is filled from a real computation"
            )


class TableMainRow(BaseModel):
    """One row of tables/table_main_universal.csv (or table_main_full_local.csv, same
    schema, different subset_id/track)."""

    model_label: str
    subset_id: str
    cmb_exam: Optional[float] = None
    cmb_exam_ci_low: Optional[float] = None
    cmb_exam_ci_high: Optional[float] = None
    cmb_clin: Optional[float] = None
    cmt_outcome: Optional[float] = None
    cmt_infocov: Optional[float] = None
    cmt_safety_violation: Optional[float] = None  # lower is better
    climedbench: Optional[float] = None
    medbench: Optional[float] = None
    mean_rank: Optional[float] = None
    n_cmb_exam: Optional[int] = None
    n_cmb_clin: Optional[int] = None
    n_cmt: Optional[int] = None
    n_climedbench: Optional[int] = None
    n_medbench: Optional[int] = None

    @model_validator(mode="after")
    def percentages_in_range(self):
        for f in ("cmb_exam", "cmb_clin", "cmt_outcome", "cmt_infocov", "cmt_safety_violation", "climedbench", "medbench"):
            v = getattr(self, f)
            if v is not None and not (0 <= v <= 100):
                raise ValueError(f"{f}={v} is outside [0,100] -- RESULTS_AND_TABLE_SCHEMA.md validation rule 4")
        if self.cmb_exam_ci_low is not None and self.cmb_exam is not None:
            if not (self.cmb_exam_ci_low <= self.cmb_exam <= (self.cmb_exam_ci_high or self.cmb_exam)):
                raise ValueError("cmb_exam CI does not bracket the point estimate -- validation rule 5")
        return self


class StageAblationRow(BaseModel):
    """One row of tables/table_stage_ablation.csv -- must have exactly rows M0/M1/M2/M3."""

    model_id: Literal["M0", "M1", "M2", "M3"]
    clinical_outcome: Optional[float] = None
    clinical_outcome_ci_low: Optional[float] = None
    clinical_outcome_ci_high: Optional[float] = None
    infocov: Optional[float] = None
    infocov_ci_low: Optional[float] = None
    infocov_ci_high: Optional[float] = None
    red_flag_recall: Optional[float] = None
    relevant_question_rate: Optional[float] = None
    duplicate_question_rate: Optional[float] = None
    premature_closure_rate: Optional[float] = None
    mean_turns: Optional[float] = None
    role_leak_rate: Optional[float] = None
    n: Optional[int] = None


class HumanTableRow(BaseModel):
    """One row of tables/table_human.csv -- rows M0/M1/M2/M3 + the pre-selected
    strongest external baseline (5 rows total per HUMAN_EVALUATION_PROTOCOL.md §3)."""

    model_label: str
    clinical_correctness: Optional[float] = None
    clinical_correctness_ci_low: Optional[float] = None
    clinical_correctness_ci_high: Optional[float] = None
    inquiry_quality: Optional[float] = None
    safety: Optional[float] = None
    actionability: Optional[float] = None
    communication: Optional[float] = None
    harmful_flag_rate: Optional[float] = None
    harmful_flag_rate_ci_low: Optional[float] = None
    harmful_flag_rate_ci_high: Optional[float] = None
    overall_pairwise_win_rate: Optional[float] = None
    n_cases: Optional[int] = None
    n_ratings: Optional[int] = None
    n_evaluators: Optional[int] = None
    abstention_rate: Optional[float] = None


class ErrorAdjudicationRecord(BaseModel):
    """One row of human_eval/error_adjudication.csv, per FIGURE_SPECIFICATIONS.md
    Figure 5. `error_category` is one of the 9 fixed categories -- not free text."""

    case_id: str
    model_id: str
    specialty: Optional[str] = None
    urgent: bool
    error_category: Literal[
        "missed_red_flag",
        "unsupported_diagnosis",
        "unsafe_treatment_specificity",
        "insufficient_history",
        "irrelevant_redundant_question",
        "premature_closure",
        "over_refusal",
        "role_transcript_leakage",
        "communication_failure",
    ]
    present: bool
    severity: Optional[str] = None
    adjudicator_count: int = Field(ge=1)
    adjudication_status: Literal["single_rater", "double_rater_agree", "adjudicated"]
