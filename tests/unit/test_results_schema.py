import pytest
from pydantic import ValidationError

from cmedalign.schema.records import (
    ErrorAdjudicationRecord,
    HumanTableRow,
    SFTMethodRow,
    StageAblationRow,
    StatisticsEntry,
    TableMainRow,
    require_exactly_one_winner,
)
from cmedalign.schema.results_layout import (
    HUMAN_EVAL_FILES,
    RESULTS_FILES,
    TABLE_FILES,
    ensure_results_layout,
    validate_identity_fields,
)


def test_ensure_results_layout_creates_dirs_and_returns_all_paths(tmp_path):
    paths = ensure_results_layout(tmp_path)
    assert (tmp_path / "results").is_dir()
    assert (tmp_path / "tables").is_dir()
    assert (tmp_path / "figures").is_dir()
    assert (tmp_path / "human_eval").is_dir()
    for fname in RESULTS_FILES + HUMAN_EVAL_FILES + TABLE_FILES:
        assert fname in paths
    # files themselves are not pre-created, only directories
    assert not paths["statistics.json"].exists()


def test_validate_identity_fields_flags_missing():
    record = {"run_id": "r1", "model_label": "M0"}
    missing = validate_identity_fields(record)
    assert "code_commit" in missing
    assert "run_id" not in missing


def test_validate_identity_fields_api_requires_extra_fields():
    record = {f: "x" for f in [
        "run_id", "code_commit", "config_hash", "prompt_hash", "model_label",
        "model_id_exact", "model_revision", "access_date", "dataset",
        "dataset_revision", "split", "subset_id", "item_id", "seed",
        "decoding_json", "input_hash", "raw_output_path", "metric_name",
        "metric_value", "parser_status", "scorer_version",
    ]}
    assert validate_identity_fields(record, is_api=False) == []
    missing_api = validate_identity_fields(record, is_api=True)
    assert "provider" in missing_api
    assert "response_model_id" in missing_api


def test_statistics_entry_require_complete_raises_when_missing():
    entry = StatisticsEntry(comparison_id="m3_vs_m2__cmt_infocov", model_a="M3", model_b="M2", metric="cmt_infocov", paired=True)
    with pytest.raises(ValueError):
        entry.require_complete()


def test_statistics_entry_require_complete_passes_when_filled():
    entry = StatisticsEntry(
        comparison_id="m3_vs_m2__cmt_infocov", model_a="M3", model_b="M2", metric="cmt_infocov",
        paired=True, n_cases=80, estimate=0.05, ci_low=0.01, ci_high=0.09, p_holm=0.03, subset_id="universal_v1",
    )
    entry.require_complete()  # must not raise


def test_table_main_row_rejects_out_of_range_percentage():
    with pytest.raises(ValidationError):
        TableMainRow(model_label="M0", subset_id="s1", cmb_exam=150.0)


def test_table_main_row_rejects_ci_not_bracketing_point():
    with pytest.raises(ValidationError):
        TableMainRow(model_label="M0", subset_id="s1", cmb_exam=50.0, cmb_exam_ci_low=60.0, cmb_exam_ci_high=70.0)


def test_table_main_row_accepts_valid_row():
    row = TableMainRow(model_label="M0", subset_id="s1", cmb_exam=65.0, cmb_exam_ci_low=60.0, cmb_exam_ci_high=70.0)
    assert row.cmb_exam == 65.0


def test_stage_ablation_row_only_accepts_m0_to_m3():
    with pytest.raises(ValidationError):
        StageAblationRow(model_id="M4")
    row = StageAblationRow(model_id="M2", infocov=55.0)
    assert row.model_id == "M2"


def test_human_table_row_basic():
    row = HumanTableRow(model_label="M3", clinical_correctness=4.2, n_cases=80, n_evaluators=5)
    assert row.n_evaluators == 5


def test_error_adjudication_record_fixed_categories_enforced():
    with pytest.raises(ValidationError):
        ErrorAdjudicationRecord(
            case_id="c1", model_id="M3", urgent=True, error_category="made_up_category",
            present=True, adjudicator_count=2, adjudication_status="double_rater_agree",
        )
    rec = ErrorAdjudicationRecord(
        case_id="c1", model_id="M3", urgent=True, error_category="missed_red_flag",
        present=True, adjudicator_count=2, adjudication_status="double_rater_agree",
    )
    assert rec.error_category == "missed_red_flag"


def test_sft_method_rows_require_exactly_one_winner():
    rows = [
        SFTMethodRow(method="lora", dev_composite_score=0.72, is_winner=True),
        SFTMethodRow(method="full_parameter", dev_composite_score=0.68, is_winner=False),
    ]
    require_exactly_one_winner(rows)  # must not raise

    no_winner = [
        SFTMethodRow(method="lora", is_winner=False),
        SFTMethodRow(method="full_parameter", is_winner=False),
    ]
    with pytest.raises(ValueError):
        require_exactly_one_winner(no_winner)

    two_winners = [
        SFTMethodRow(method="lora", is_winner=True),
        SFTMethodRow(method="full_parameter", is_winner=True),
    ]
    with pytest.raises(ValueError):
        require_exactly_one_winner(two_winners)
