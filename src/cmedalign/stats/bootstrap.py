"""Bootstrap confidence intervals and Holm-Bonferroni multiple-comparison correction.

G6 requires every main-comparison number to carry a CI and be Holm-corrected — this is
the shared implementation both `eval-core`/`eval-baselines` results and the final table
builder call into, so there is exactly one definition of "the CI" in the whole project.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    statistic: Callable[[np.ndarray], float] = np.mean,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (point_estimate, lower, upper)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_ci: no values provided")

    point = float(statistic(arr))
    rng = np.random.default_rng(seed)
    n = arr.size
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic(sample)

    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return point, lower, upper


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Standard step-down Holm procedure. Once a hypothesis fails to reject (in
    ascending p-value order), every subsequent (larger-p) hypothesis is also not
    rejected, regardless of its own p-value vs its own threshold."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    results: dict[str, dict] = {}
    still_rejecting = True
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        rejected = still_rejecting and (p <= threshold)
        if not rejected:
            still_rejecting = False
        results[name] = {
            "p_value": p,
            "rank": i + 1,
            "threshold": threshold,
            "reject": rejected,
        }
    return results
