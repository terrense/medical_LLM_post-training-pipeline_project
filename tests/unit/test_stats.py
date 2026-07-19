import numpy as np
import pytest

from cmedalign.stats.bootstrap import bootstrap_ci, holm_bonferroni
from cmedalign.stats.tables import build_table_from_item_level, mark_best


def test_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.7, scale=0.1, size=200).tolist()
    point, lower, upper = bootstrap_ci(values, n_boot=2000, seed=1)
    assert lower <= point <= upper


def test_ci_narrows_with_more_data():
    rng = np.random.default_rng(0)
    small = rng.normal(0.5, 0.2, size=20).tolist()
    large = rng.normal(0.5, 0.2, size=2000).tolist()
    _, lo_s, hi_s = bootstrap_ci(small, n_boot=2000, seed=1)
    _, lo_l, hi_l = bootstrap_ci(large, n_boot=2000, seed=1)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_holm_bonferroni_rejects_strong_signal_only():
    pvals = {"a": 0.001, "b": 0.20, "c": 0.60}
    result = holm_bonferroni(pvals, alpha=0.05)
    assert result["a"]["reject"] is True
    assert result["b"]["reject"] is False
    assert result["c"]["reject"] is False


def test_holm_bonferroni_stepdown_stops_after_first_failure():
    # even though "c" has a tiny p-value, it comes after "b" fails in rank order,
    # so it must not be rejected either (monotonic step-down).
    pvals = {"a": 0.01, "b": 0.04, "c": 0.001}
    # sorted ascending: c(0.001), a(0.01), b(0.04); m=3
    # c: threshold 0.05/3=0.0167 -> 0.001<=0.0167 reject
    # a: threshold 0.05/2=0.025 -> 0.01<=0.025 reject
    # b: threshold 0.05/1=0.05 -> 0.04<=0.05 reject
    result = holm_bonferroni(pvals, alpha=0.05)
    assert result["c"]["reject"] is True
    assert result["a"]["reject"] is True
    assert result["b"]["reject"] is True

    # Now make "a" fail, and add a later (in rank order) tiny-p test "d" that would pass
    # on its own threshold but must still be blocked by the step-down rule.
    pvals2 = {"a": 0.03, "d": 0.02, "e": 0.05}
    # sorted ascending: d(0.02), a(0.03), e(0.05); thresholds: 0.0167, 0.025, 0.05
    # d: 0.02 <= 0.0167? No -> not reject -> stepdown halts -> a, e also not reject
    result2 = holm_bonferroni(pvals2, alpha=0.05)
    assert result2["d"]["reject"] is False
    assert result2["a"]["reject"] is False
    assert result2["e"]["reject"] is False


def test_build_table_from_item_level_fails_on_missing_field():
    records = [
        {"model": "m1", "metric": "acc", "score": 0.8, "item_id": "i1"},
        {"model": "m1", "metric": "acc", "item_id": "i2"},  # missing "score"
    ]
    with pytest.raises(ValueError):
        build_table_from_item_level(records)


def test_build_table_from_item_level_groups_and_aggregates():
    records = [
        {"model": "m1", "metric": "acc", "score": 1.0, "item_id": "i1"},
        {"model": "m1", "metric": "acc", "score": 0.0, "item_id": "i2"},
        {"model": "m2", "metric": "acc", "score": 1.0, "item_id": "i1"},
        {"model": "m2", "metric": "acc", "score": 1.0, "item_id": "i2"},
    ]
    table = build_table_from_item_level(records, n_boot=500, seed=0)
    by_model = {row["model"]: row for row in table}
    assert by_model["m1"]["point"] == pytest.approx(0.5)
    assert by_model["m2"]["point"] == pytest.approx(1.0)
    assert by_model["m1"]["n"] == 2


def test_mark_best_respects_orientation():
    table = [
        {"model": "m1", "metric": "accuracy", "point": 0.9},
        {"model": "m2", "metric": "accuracy", "point": 0.7},
        {"model": "m1", "metric": "error_rate", "point": 0.05},
        {"model": "m2", "metric": "error_rate", "point": 0.20},
    ]
    marked = mark_best(
        table,
        orientation_by_metric={"accuracy": "higher_is_better", "error_rate": "lower_is_better"},
    )
    by_key = {(r["model"], r["metric"]): r for r in marked}
    assert by_key[("m1", "accuracy")]["bold"] is True
    assert by_key[("m2", "accuracy")]["bold"] is False
    assert by_key[("m1", "error_rate")]["bold"] is True
    assert by_key[("m2", "error_rate")]["bold"] is False


def test_mark_best_missing_orientation_raises():
    table = [{"model": "m1", "metric": "unknown_metric", "point": 1.0}]
    with pytest.raises(ValueError):
        mark_best(table, orientation_by_metric={})
