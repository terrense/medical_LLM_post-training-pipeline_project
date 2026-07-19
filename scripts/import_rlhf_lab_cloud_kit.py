#!/usr/bin/env python3
"""One-time import: rlhf_lab_cloud_kit's already-cleaned/deduped/task_type-tagged pool
-> cmedalign's ConversationRecord schema in data/raw/.

Per spec/design.md's "THE ANCHOR" section and BLOCKERS.md: as of 2026-07-19 all 8
sources in that pool are cleared to import (the PII question about
internal_seed_flywheel/derived_from_seed was resolved -- they're fully synthetic,
DeepSeek-V4-Pro-simulated-patient + Qwen3-8B-generated-doctor + reviewer-revised +
clinically audited, not real patient data).

Usage: python scripts/import_rlhf_lab_cloud_kit.py \
    --input E:/rlhf_lab_cloud_kit/data/eval_sets/05_final_v11/train.jsonl \
    --input E:/rlhf_lab_cloud_kit/data/eval_sets/05_final_v11/dev.jsonl \
    --out-dir data/raw/rlhf_lab_cloud_kit
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SYNTHETIC_SOURCES = {
    "internal_seed_flywheel": dict(
        teacher_model="DeepSeek-V4-Pro",
        doctor_model="Qwen3-8B",
        reviewed_by=["stronger_reviewing_model", "clinical_staff"],
        audit_status="clinically_audited",
    ),
    "derived_from_seed": dict(
        teacher_model="DeepSeek-V4-Pro",
        doctor_model="Qwen3-8B",
        reviewed_by=["stronger_reviewing_model", "clinical_staff"],
        audit_status="clinically_audited",
    ),
    "gen_minimax_m3": dict(
        teacher_model="MiniMax-M3",
        doctor_model=None,
        reviewed_by=[],
        audit_status="unreviewed",
    ),
}

QUALITY_SCORE_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}


def exact_hash(text: str) -> str:
    import re

    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def convert(rec: dict, split: str) -> dict:
    msgs = rec["messages"]
    text = "\n".join(m["content"] for m in msgs)
    meta = rec.get("metadata", {})
    source = rec["source"]

    safety_tags = [f"risk_level:{meta.get('risk_level')}"]
    safety_tags.extend(meta.get("red_flags", []) or [])

    out = {
        "sample_id": f"rlck_{rec['id']}",
        "source": source,
        "source_revision": "rlhf_lab_cloud_kit_v1.1_05_final",
        "split": split,
        "messages": msgs,
        "specialty": meta.get("department"),
        "task_type": rec.get("task_type"),
        "quality_score": QUALITY_SCORE_MAP.get(meta.get("source_quality")),
        "safety_tags": safety_tags,
        "provenance": "rlhf_lab_cloud_kit/data/eval_sets/05_final_v11 (imported 2026-07-19)",
        "license_id": meta.get("license", "unknown"),
        "raw_hash": meta.get("dedup_hash") or exact_hash(text),
        "normalized_hash": exact_hash(text),
        "synthetic_or_real": "real",
    }

    if source in SYNTHETIC_SOURCES:
        prov = SYNTHETIC_SOURCES[source]
        out["synthetic_or_real"] = "synthetic"
        out["synthetic_provenance"] = {
            "teacher_model": prov["teacher_model"],
            "doctor_model": prov["doctor_model"],
            "reviewed_by": prov["reviewed_by"],
            "audit_status": prov["audit_status"],
        }

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="repeatable: train.jsonl, dev.jsonl, ...")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, list] = {}
    counts: dict[str, int] = {}

    for path in args.input:
        split = "dev" if "dev" in Path(path).stem else "train"
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                converted = convert(rec, split)
                src = rec["source"]
                writers.setdefault(src, []).append(converted)
                counts[src] = counts.get(src, 0) + 1

    for src, records in writers.items():
        safe_name = src.replace("/", "_")
        out_path = out_dir / f"{safe_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(f"total: {sum(counts.values())} records written to {out_dir}")


if __name__ == "__main__":
    main()
