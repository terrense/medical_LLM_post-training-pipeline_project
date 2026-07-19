#!/usr/bin/env python3
"""Import Suprit/CMtMedQA (train, MIT) and Suprit/CMtMedQA_test_v1 (test, apache-2.0)
from their downloaded Alpaca+history JSON into cmedalign's ConversationRecord schema.

CRITICAL: the test file is the paper's held-out CMtMedQA_test benchmark
(main.tex Table~1: "Evaluation only ... Never used for patient/profile construction").
This script writes it with split="test" into a clearly separate file/directory and
never merges it with the train output -- downstream code must never read
cmtmedqa/test.jsonl into any SFT/DPO/GRPO construction step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def exact_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def convert(rec: dict, idx: int, split: str) -> dict | None:
    messages = []
    for turn in rec.get("history", []) or []:
        if len(turn) != 2:
            return None
        u, a = turn
        if not u.strip() or not a.strip():
            return None
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    final_user = (rec.get("instruction") or "") + (("\n" + rec["input"]) if rec.get("input") else "")
    final_assistant = rec.get("output") or ""
    if not final_user.strip() or not final_assistant.strip():
        return None
    messages.append({"role": "user", "content": final_user})
    messages.append({"role": "assistant", "content": final_assistant})

    text = "\n".join(m["content"] for m in messages)
    specialty = " / ".join(x for x in [rec.get("cate1"), rec.get("cate2")] if x)

    return {
        "sample_id": f"cmtmedqa_{split}_{rec.get('id', idx)}",
        "source": "CMtMedQA",
        "source_revision": "Suprit/CMtMedQA" if split == "train" else "Suprit/CMtMedQA_test_v1",
        "split": split,
        "messages": messages,
        "specialty": specialty or None,
        "task_type": "pre_consultation_multiturn",
        "quality_score": None,
        "safety_tags": [],
        "provenance": f"huggingface.co/datasets/Suprit/CMtMedQA{'_test_v1' if split=='test' else ''}",
        "license_id": "MIT" if split == "train" else "apache-2.0",
        "raw_hash": exact_hash(text),
        "normalized_hash": exact_hash(text),
        "synthetic_or_real": "real",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-out", required=True)
    args = ap.parse_args()

    for in_path, out_path, split in [
        (args.train_json, args.train_out, "train"),
        (args.test_json, args.test_out, "test"),
    ]:
        with open(in_path, encoding="utf-8") as f:
            raw = json.load(f)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        n_ok = n_skip = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for i, rec in enumerate(raw):
                converted = convert(rec, i, split)
                if converted is None:
                    n_skip += 1
                    continue
                out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                n_ok += 1
        print(f"{split}: {n_ok} written, {n_skip} skipped (malformed turns) -> {out_path}")


if __name__ == "__main__":
    main()
