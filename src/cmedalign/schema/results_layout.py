"""The canonical output directory contract, copied exactly from
`E:\\cmedalign_paper\\RESULTS_AND_TABLE_SCHEMA.md` — this is not a suggestion, it's what
the paper's fill-in tooling and `claim_values.tex` generator expect to find. Do not
invent alternate paths/names for these files; if a new artifact is needed, add it here
and note why, rather than putting it somewhere ad hoc.
"""
from __future__ import annotations

from pathlib import Path

RESULTS_FILES = [
    "run_manifest.json",
    "data_manifest.json",
    "model_registry.json",
    "benchmark_item_scores.parquet",
    "benchmark_aggregates.csv",
    "stage_metrics.csv",
    "interactive_trajectories.jsonl",
    "interactive_item_scores.parquet",
    "judge_scores.jsonl",
    "judge_aggregates.csv",
    "training_curves.parquet",
    "efficiency.csv",
    "statistics.json",
]

HUMAN_EVAL_FILES = [
    "cases_blinded.jsonl",
    "assignment.csv",
    "blind_map.enc",
    "rating_schema.json",
    "ratings_anonymized.csv",
    "pairwise_preferences.csv",
    "error_adjudication.csv",
    "human_statistics.json",
    "protocol_deviations.md",
]

TABLE_FILES = [
    "table_data_registry.csv",
    "table_hyperparameters.csv",
    "table_models.csv",
    "table_initialization_probe.csv",
    "table_main_universal.csv",
    "table_main_full_local.csv",
    "table_stage_ablation.csv",
    "table_human.csv",
    "table_reward_ablations.csv",
]


def ensure_results_layout(root: Path) -> dict[str, Path]:
    """Creates results/, tables/, figures/, human_eval/ (if missing) under `root` and
    returns a {name: Path} map for every canonical file (files are NOT created empty --
    only directories -- callers should fail loudly if they try to read a file that was
    never written, not silently treat a missing file as empty)."""
    dirs = {
        "results": root / "results",
        "tables": root / "tables",
        "figures": root / "figures",
        "human_eval": root / "human_eval",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for fname in RESULTS_FILES:
        paths[fname] = dirs["results"] / fname
    for fname in HUMAN_EVAL_FILES:
        paths[fname] = dirs["human_eval"] / fname
    for fname in TABLE_FILES:
        paths[fname] = dirs["tables"] / fname
    return paths


# Required identity fields for every item-level record (RESULTS_AND_TABLE_SCHEMA.md
# "Required identity fields"). Local models and API models share these; API records add
# IDENTITY_FIELDS_API on top.
IDENTITY_FIELDS_CORE = [
    "run_id", "code_commit", "config_hash", "prompt_hash",
    "model_label", "model_id_exact", "model_revision", "access_date",
    "dataset", "dataset_revision", "split", "subset_id", "item_id",
    "seed", "decoding_json", "input_hash", "raw_output_path",
    "metric_name", "metric_value", "parser_status", "scorer_version",
]

IDENTITY_FIELDS_API = [
    "provider", "endpoint", "response_model_id", "request_id_hash",
    "usage_input_tokens", "usage_output_tokens", "price_snapshot_date",
]


def validate_identity_fields(record: dict, is_api: bool = False) -> list[str]:
    """Returns a list of missing required identity fields (empty list = fully compliant).
    Callers should raise/fail the build on any non-empty result for a record that's
    meant to back a reported number -- never silently proceed with partial identity."""
    required = IDENTITY_FIELDS_CORE + (IDENTITY_FIELDS_API if is_api else [])
    return [f for f in required if f not in record or record[f] is None]
