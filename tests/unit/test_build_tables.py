from cmedalign.stats.build_tables import build_human_table, build_stage_ablation, build_table_main


def test_build_table_main_produces_schema_conformant_rows():
    records = []
    for model, cmb_score in [("M0", 60.0), ("M3", 70.0)]:
        for i in range(20):
            records.append({"model_label": model, "metric": "cmb_exam", "score": cmb_score, "item_id": f"i{i}"})
    for model, viol in [("M0", 20.0), ("M3", 5.0)]:
        for i in range(20):
            records.append({"model_label": model, "metric": "cmt_safety_violation", "score": viol, "item_id": f"i{i}"})

    rows = build_table_main(records, subset_id="universal_v1", n_boot=500, seed=1)
    by_model = {r.model_label: r for r in rows}

    assert by_model["M0"].cmb_exam == 60.0
    assert by_model["M3"].cmb_exam == 70.0
    assert by_model["M0"].subset_id == "universal_v1"
    assert by_model["M0"].n_cmb_exam == 20
    # M3 has higher cmb_exam (better) and lower safety_violation (better) -> should rank
    # better (lower mean_rank) than M0 once safety_violation orientation is respected.
    assert by_model["M3"].mean_rank < by_model["M0"].mean_rank


def test_build_table_main_missing_field_raises():
    import pytest

    with pytest.raises(ValueError):
        build_table_main([{"model_label": "M0", "metric": "cmb_exam"}], subset_id="s1")


def test_build_stage_ablation_only_m0_to_m3():
    records = []
    for model in ["M0", "M1", "M2", "M3"]:
        for i in range(10):
            records.append({"model_label": model, "metric": "infocov", "score": 50.0 + 10 * "M0M1M2M3".index(model), "item_id": f"i{i}"})
    records.append({"model_label": "Qwen3-32B", "metric": "infocov", "score": 99.0, "item_id": "i0"})

    rows = build_stage_ablation(records, n_boot=500, seed=1)
    model_ids = {r.model_id for r in rows}
    assert model_ids == {"M0", "M1", "M2", "M3"}
    by_model = {r.model_id: r for r in rows}
    assert by_model["M3"].infocov > by_model["M0"].infocov


def test_build_human_table_computes_rates_and_counts():
    records = []
    for case_id in range(10):
        records.append({"model_label": "M3", "dimension": "clinical_correctness", "score": 4.0, "case_id": f"c{case_id}", "evaluator_id": "r1"})
        records.append({"model_label": "M3", "dimension": "harmful_flag", "score": 1.0 if case_id == 0 else 0.0, "case_id": f"c{case_id}"})
        records.append({"model_label": "M3", "dimension": "pairwise_win", "score": 1.0 if case_id < 7 else 0.0, "case_id": f"c{case_id}"})

    rows = build_human_table(records, n_boot=500, seed=1)
    row = rows[0]
    assert row.model_label == "M3"
    assert row.clinical_correctness == 4.0
    assert row.harmful_flag_rate == 0.1
    assert row.overall_pairwise_win_rate == 0.7
    assert row.n_cases == 10
    assert row.n_evaluators == 1
