"""Build result tables from item-level raw records — never from hand-typed numbers.

G6: "表格只能由 item-level 结果生成" and "任何缺字段使构建失败". This module fails loudly
(raises) on any missing required field rather than silently skipping or defaulting.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

from cmedalign.stats.bootstrap import bootstrap_ci

REQUIRED_ITEM_FIELDS = ("model", "metric", "score", "item_id")

Orientation = Literal["higher_is_better", "lower_is_better"]


def build_table_from_item_level(
    records: list[dict],
    required_fields: tuple[str, ...] = REQUIRED_ITEM_FIELDS,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> list[dict]:
    if not records:
        raise ValueError("build_table_from_item_level: no item-level records given")

    for rec in records:
        missing = [f for f in required_fields if f not in rec or rec[f] is None]
        if missing:
            raise ValueError(
                f"item-level record missing required field(s) {missing}: {rec!r}"
            )

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in records:
        grouped[(rec["model"], rec["metric"])].append(float(rec["score"]))

    table = []
    for (model, metric), scores in grouped.items():
        point, lower, upper = bootstrap_ci(scores, n_boot=n_boot, alpha=alpha, seed=seed)
        table.append(
            {
                "model": model,
                "metric": metric,
                "point": point,
                "ci_lower": lower,
                "ci_upper": upper,
                "n": len(scores),
            }
        )
    return table


def mark_best(
    table: list[dict],
    orientation_by_metric: dict[str, Orientation],
    value_key: str = "point",
    metric_key: str = "metric",
) -> list[dict]:
    """Adds a `bold: bool` flag to each row: True iff it is the best value for its
    metric group, respecting that metric's orientation (higher-is-better vs
    lower-is-better, e.g. accuracy vs an error rate)."""
    by_metric: dict[str, list[dict]] = defaultdict(list)
    for row in table:
        by_metric[row[metric_key]].append(row)

    out = []
    for metric, rows in by_metric.items():
        if metric not in orientation_by_metric:
            raise ValueError(f"mark_best: no orientation specified for metric '{metric}'")
        orientation = orientation_by_metric[metric]
        best_val = (
            max(r[value_key] for r in rows)
            if orientation == "higher_is_better"
            else min(r[value_key] for r in rows)
        )
        for r in rows:
            out.append({**r, "bold": r[value_key] == best_val})
    return out
