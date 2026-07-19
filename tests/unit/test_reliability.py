import pytest

from cmedalign.stats.reliability import bootstrap_alpha_ci, krippendorff_alpha_ordinal


def test_perfect_agreement_gives_alpha_one():
    units = {
        "u1": [5, 5, 5],
        "u2": [3, 3],
        "u3": [1, 1, 1],
        "u4": [4, 4],
        "u5": [2, 2, 2],
    }
    alpha = krippendorff_alpha_ordinal(units, categories=[1, 2, 3, 4, 5])
    assert alpha == pytest.approx(1.0, abs=1e-9)


def test_systematic_extreme_disagreement_gives_low_alpha():
    # Raters flip between the two extreme categories inconsistently across units --
    # about as bad as ordinal disagreement gets.
    units = {
        f"u{i}": pair
        for i, pair in enumerate(
            [[1, 5], [5, 1], [1, 5], [5, 1], [1, 5], [5, 1], [1, 5], [5, 1], [1, 5], [5, 1]]
        )
    }
    alpha = krippendorff_alpha_ordinal(units, categories=[1, 2, 3, 4, 5])
    assert alpha < 0.0  # worse than chance agreement


def test_single_rater_units_are_ignored_not_crashed():
    units = {
        "u1": [5, 5],
        "u2": [3],  # only one rater, must be ignored, not counted or error
        "u3": [1, 1],
    }
    alpha = krippendorff_alpha_ordinal(units, categories=[1, 3, 5])
    assert alpha == pytest.approx(1.0, abs=1e-9)


def test_degenerate_all_same_category_raises():
    units = {"u1": [3, 3], "u2": [3, 3, 3]}
    with pytest.raises(ValueError):
        krippendorff_alpha_ordinal(units, categories=[1, 2, 3, 4, 5])


def test_no_multi_rater_units_raises():
    units = {"u1": [3], "u2": [4]}
    with pytest.raises(ValueError):
        krippendorff_alpha_ordinal(units)


def test_bootstrap_alpha_ci_contains_point_estimate():
    units = {
        "u1": [5, 5, 4],
        "u2": [3, 3, 4],
        "u3": [1, 1, 2],
        "u4": [4, 4, 5],
        "u5": [2, 2, 1],
        "u6": [5, 4, 5],
    }
    point, lo, hi = bootstrap_alpha_ci(units, categories=[1, 2, 3, 4, 5], n_boot=500, seed=1)
    assert lo <= point <= hi
