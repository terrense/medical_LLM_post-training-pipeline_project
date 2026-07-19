"""Builds the paper's exact tables (table_main_universal.csv, table_stage_ablation.csv,
table_human.csv) from item-level records, conforming to RESULTS_AND_TABLE_SCHEMA.md and
cmedalign.schema.records exactly -- not an ad-hoc format.

Input item-level record shape expected (one row per (model, benchmark_metric, item)):
    {"model_label": "M0", "metric": "cmb_exam", "score": 0.0-100.0 or 0/1,
     "item_id": "...", "subset_id": "universal_v1"}
This is intentionally similar to cmedalign.stats.tables.build_table_from_item_level's
input shape (reuses that function's bootstrap-CI machinery underneath), but this module
additionally reshapes the *output* into the paper's exact wide-table schema (one row per
model with all benchmark columns side by side, not one row per model-metric pair).

CLI entry point (see Makefile's `statistics` target):
    python -m cmedalign.stats.build_tables --results-dir results --out-dir tables
Reads results/benchmark_item_scores.parquet (falls back to .jsonl if parquet tooling
isn't installed) and results/interactive_item_scores.parquet; writes tables/*.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from cmedalign.schema.records import HumanTableRow, StageAblationRow, TableMainRow
from cmedalign.stats.bootstrap import bootstrap_ci

# benchmark metric -> (TableMainRow field, ci_low field or None, n field or None)
_MAIN_METRIC_FIELDS = {
    "cmb_exam": ("cmb_exam", "cmb_exam_ci_low", "cmb_exam_ci_high", "n_cmb_exam"),
    "cmb_clin": ("cmb_clin", None, None, "n_cmb_clin"),
    "cmt_outcome": ("cmt_outcome", None, None, "n_cmt"),
    "cmt_infocov": ("cmt_infocov", None, None, "n_cmt"),
    "cmt_safety_violation": ("cmt_safety_violation", None, None, "n_cmt"),
    "climedbench": ("climedbench", None, None, "n_climedbench"),
    "medbench": ("medbench", None, None, "n_medbench"),
}

# metrics where LOWER is better -- must be excluded from the naive "higher is better"
# mean-rank computation unless inverted first (RESULTS_AND_TABLE_SCHEMA.md's own
# validation rule 6: "safety orientation is not reversed before entering the table").
_LOWER_IS_BETTER = {"cmt_safety_violation"}


def load_item_scores_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_table_main(item_scores: list[dict], subset_id: str, n_boot: int = 10000, seed: int = 0) -> list[TableMainRow]:
    if not item_scores:
        raise ValueError("build_table_main: no item-level records given")

    by_model_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in item_scores:
        for field in ("model_label", "metric", "score", "item_id"):
            if field not in rec:
                raise ValueError(f"item-level record missing required field '{field}': {rec!r}")
        by_model_metric[(rec["model_label"], rec["metric"])].append(float(rec["score"]))

    per_model: dict[str, dict] = defaultdict(dict)
    for (model_label, metric), scores in by_model_metric.items():
        if metric not in _MAIN_METRIC_FIELDS:
            continue
        point, lo, hi = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
        field, ci_low_field, ci_high_field, n_field = _MAIN_METRIC_FIELDS[metric]
        per_model[model_label][field] = point
        if ci_low_field:
            per_model[model_label][ci_low_field] = lo
            per_model[model_label][ci_high_field] = hi
        per_model[model_label][n_field] = len(scores)

    # mean_rank across whichever metrics this model has (orient lower-is-better metrics
    # so that "better" always ranks lower, per validation rule 6).
    metrics_present = sorted({m for _, m in by_model_metric if m in _MAIN_METRIC_FIELDS})
    ranks: dict[str, list[float]] = defaultdict(list)
    for metric in metrics_present:
        field = _MAIN_METRIC_FIELDS[metric][0]
        values = [(model, per_model[model][field]) for model in per_model if field in per_model[model]]
        if not values:
            continue
        oriented = [(m, v if metric not in _LOWER_IS_BETTER else -v) for m, v in values]
        oriented.sort(key=lambda x: -x[1])
        for rank, (model, _) in enumerate(oriented, start=1):
            ranks[model].append(rank)

    rows = []
    for model_label, fields in per_model.items():
        mean_rank = sum(ranks[model_label]) / len(ranks[model_label]) if ranks.get(model_label) else None
        rows.append(TableMainRow(model_label=model_label, subset_id=subset_id, mean_rank=mean_rank, **fields))
    return rows


def build_stage_ablation(item_scores: list[dict], n_boot: int = 10000, seed: int = 0) -> list[StageAblationRow]:
    """item_scores here are process/outcome metrics keyed the same way as build_table_main
    but restricted to M0-M3 and the stage-ablation metric names."""
    field_map = {
        "clinical_outcome": "clinical_outcome",
        "infocov": "infocov",
        "red_flag_recall": "red_flag_recall",
        "relevant_question_rate": "relevant_question_rate",
        "duplicate_question_rate": "duplicate_question_rate",
        "premature_closure_rate": "premature_closure_rate",
        "mean_turns": "mean_turns",
        "role_leak_rate": "role_leak_rate",
    }
    by_model_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in item_scores:
        if rec["model_label"] not in ("M0", "M1", "M2", "M3"):
            continue
        by_model_metric[(rec["model_label"], rec["metric"])].append(float(rec["score"]))

    rows = []
    for model_id in ("M0", "M1", "M2", "M3"):
        fields = {}
        n = None
        for metric, field in field_map.items():
            scores = by_model_metric.get((model_id, metric))
            if not scores:
                continue
            point, lo, hi = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
            fields[field] = point
            if field == "clinical_outcome":
                fields["clinical_outcome_ci_low"] = lo
                fields["clinical_outcome_ci_high"] = hi
            if field == "infocov":
                fields["infocov_ci_low"] = lo
                fields["infocov_ci_high"] = hi
            n = len(scores)
        if fields:
            rows.append(StageAblationRow(model_id=model_id, n=n, **fields))
    return rows


def build_human_table(human_ratings: list[dict], n_boot: int = 10000, seed: int = 0) -> list[HumanTableRow]:
    """human_ratings: one row per (model_label, dimension, case_id) rating, e.g.
    {"model_label": "M3", "dimension": "clinical_correctness", "score": 4.0, "case_id": "..."}
    plus optionally {"model_label", "dimension": "harmful_flag", "score": 0/1, "case_id"}
    and {"model_label", "dimension": "pairwise_win", "score": 0/1, "case_id"}."""
    by_model_dim: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_cases_by_model: dict[str, set] = defaultdict(set)
    n_evaluators_by_model: dict[str, set] = defaultdict(set)
    for rec in human_ratings:
        by_model_dim[(rec["model_label"], rec["dimension"])].append(float(rec["score"]))
        n_cases_by_model[rec["model_label"]].add(rec["case_id"])
        if "evaluator_id" in rec:
            n_evaluators_by_model[rec["model_label"]].add(rec["evaluator_id"])

    rows = []
    models = sorted({m for m, _ in by_model_dim})
    for model_label in models:
        fields: dict = {}
        for dim in ("clinical_correctness", "inquiry_quality", "safety", "actionability", "communication"):
            scores = by_model_dim.get((model_label, dim))
            if scores:
                point, lo, hi = bootstrap_ci(scores, n_boot=n_boot, seed=seed)
                fields[dim] = point
                if dim == "clinical_correctness":
                    fields["clinical_correctness_ci_low"] = lo
                    fields["clinical_correctness_ci_high"] = hi
        harmful = by_model_dim.get((model_label, "harmful_flag"))
        if harmful:
            point, lo, hi = bootstrap_ci(harmful, n_boot=n_boot, seed=seed)
            fields["harmful_flag_rate"] = point
            fields["harmful_flag_rate_ci_low"] = lo
            fields["harmful_flag_rate_ci_high"] = hi
        pairwise = by_model_dim.get((model_label, "pairwise_win"))
        if pairwise:
            fields["overall_pairwise_win_rate"] = sum(pairwise) / len(pairwise)
        fields["n_cases"] = len(n_cases_by_model[model_label])
        fields["n_ratings"] = sum(len(v) for (m, d), v in by_model_dim.items() if m == model_label)
        if n_evaluators_by_model.get(model_label):
            fields["n_evaluators"] = len(n_evaluators_by_model[model_label])
        rows.append(HumanTableRow(model_label=model_label, **fields))
    return rows


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    dicts = [r.model_dump() for r in rows]
    fieldnames = list(dicts[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dicts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subset-id", default="universal_v1")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    benchmark_path = results_dir / "benchmark_item_scores.jsonl"
    if benchmark_path.exists():
        item_scores = load_item_scores_jsonl(benchmark_path)
        main_rows = build_table_main(item_scores, subset_id=args.subset_id)
        _write_csv(out_dir / "table_main_universal.csv", main_rows)
        stage_rows = build_stage_ablation(item_scores)
        _write_csv(out_dir / "table_stage_ablation.csv", stage_rows)
    else:
        print(f"SKIP table_main/table_stage_ablation: {benchmark_path} not found yet (no real results)")

    human_path = results_dir.parent / "human_eval" / "ratings_anonymized.jsonl"
    if human_path.exists():
        human_rows = build_human_table(load_item_scores_jsonl(human_path))
        _write_csv(out_dir / "table_human.csv", human_rows)
    else:
        print(f"SKIP table_human: {human_path} not found yet (no real human ratings)")


if __name__ == "__main__":
    main()
