"""Ordinal Krippendorff's alpha — inter-rater reliability for the blinded human
evaluation's five 1-5 dimensions (HUMAN_EVALUATION_PROTOCOL.md §12,
RESULTS_AND_TABLE_SCHEMA.md's human_statistics.json contract).

What it measures: how much independent raters agree with each other, corrected for the
agreement you'd expect by pure chance. Unlike simple percent-agreement, it treats a
disagreement between adjacent categories (4 vs 5) as less severe than a disagreement
between distant ones (1 vs 5) -- that's what "ordinal" means here, as opposed to
"nominal" alpha which would treat any mismatch as equally bad. Range is roughly
[-1, 1]; conventionally >=0.8 is considered good reliability, 0.67-0.8 "tentative",
below that the ratings shouldn't be treated as reliable. It requires at least 2 raters
per unit (case) with actual overlap -- can't be computed from single-rater data, which
is exactly why HUMAN_EVALUATION_PROTOCOL.md requires 2 raters per case (3 for a 25%
reliability subsample).

This can't be exercised against real numbers yet -- no human ratings have been
collected (that happens near the end, once M3 exists and the blind package goes out).
What's implemented and tested now: the alpha computation itself (against constructed
cases with known true agreement level) and a bootstrap CI over case-clustered resampling.
"""
from __future__ import annotations

import random
from typing import Optional


def _build_coincidence_matrix(
    units: dict[str, list[int]], categories: list[int]
) -> tuple[list[list[float]], float]:
    cat_index = {c: i for i, c in enumerate(categories)}
    C = len(categories)
    o = [[0.0] * C for _ in range(C)]
    n_total = 0.0
    for ratings in units.values():
        m = len(ratings)
        if m < 2:
            continue
        n_total += m
        counts = [0] * C
        for r in ratings:
            counts[cat_index[r]] += 1
        for c in range(C):
            if counts[c] == 0:
                continue
            for k in range(C):
                if counts[k] == 0:
                    continue
                if c == k:
                    o[c][k] += counts[c] * (counts[c] - 1) / (m - 1)
                else:
                    o[c][k] += counts[c] * counts[k] / (m - 1)
    return o, n_total


def _marginals(o: list[list[float]]) -> list[float]:
    C = len(o)
    return [sum(o[c][k] for k in range(C)) for c in range(C)]


def _ordinal_delta(marginal_n: list[float], c_idx: int, k_idx: int) -> float:
    if c_idx == k_idx:
        return 0.0
    lo, hi = min(c_idx, k_idx), max(c_idx, k_idx)
    total = sum(marginal_n[g] for g in range(lo, hi + 1))
    boundary = (marginal_n[lo] + marginal_n[hi]) / 2
    return (total - boundary) ** 2


def krippendorff_alpha_ordinal(
    units: dict[str, list[int]], categories: Optional[list[int]] = None
) -> float:
    """`units`: {unit_id: [rating_1, rating_2, ...]} -- variable raters per unit OK,
    units with <2 raters are ignored (can't contribute to reliability). `categories`:
    the full ordered category set (e.g. [1,2,3,4,5]); inferred from the data if omitted,
    but pass it explicitly in real use so a category nobody happened to use in this
    particular sample doesn't silently change the scale.
    """
    if categories is None:
        categories = sorted({v for ratings in units.values() for v in ratings})
    o, n = _build_coincidence_matrix(units, categories)
    if n == 0:
        raise ValueError("krippendorff_alpha_ordinal: no unit has >=2 raters, cannot compute")

    marginal_n = _marginals(o)
    C = len(categories)
    Do = 0.0
    De = 0.0
    for c in range(C):
        for k in range(C):
            if c == k:
                continue
            delta = _ordinal_delta(marginal_n, c, k)
            Do += o[c][k] * delta
            De += marginal_n[c] * marginal_n[k] * delta
    Do /= n
    De /= n * (n - 1)
    if De == 0:
        raise ValueError(
            "krippendorff_alpha_ordinal: expected disagreement is zero (every rating in "
            "the dataset used the same single category) -- alpha is undefined without "
            "any variance to assess reliability against"
        )
    return 1 - Do / De


def bootstrap_alpha_ci(
    units: dict[str, list[int]],
    categories: Optional[list[int]] = None,
    n_boot: int = 2000,
    alpha_level: float = 0.05,
    seed: Optional[int] = None,
) -> tuple[float, float, float]:
    """Case-clustered bootstrap CI for Krippendorff's alpha: resample UNITS (cases)
    with replacement, not individual ratings -- preserves each case's own rater
    agreement structure, matching HUMAN_EVALUATION_PROTOCOL.md's "case-clustered
    bootstrap" convention used elsewhere in this project."""
    if categories is None:
        categories = sorted({v for ratings in units.values() for v in ratings})

    point = krippendorff_alpha_ordinal(units, categories)

    unit_ids = [u for u, r in units.items() if len(r) >= 2]
    rng = random.Random(seed)
    boot_values = []
    for _ in range(n_boot):
        sample_ids = [rng.choice(unit_ids) for _ in unit_ids]
        resampled = {f"{uid}__{i}": units[uid] for i, uid in enumerate(sample_ids)}
        try:
            boot_values.append(krippendorff_alpha_ordinal(resampled, categories))
        except ValueError:
            continue  # degenerate resample (e.g. all-same-category by chance), skip it

    if not boot_values:
        return point, point, point
    boot_values.sort()
    lo_idx = int(len(boot_values) * (alpha_level / 2))
    hi_idx = int(len(boot_values) * (1 - alpha_level / 2)) - 1
    return point, boot_values[max(0, lo_idx)], boot_values[min(len(boot_values) - 1, hi_idx)]
