"""CLI: run the G1 data audit (license completeness, exact/near-dup, cross-split
contamination) over a normalized data directory and write a single audit report.

Usage (on the server, once real normalized data + manifests exist):
    python -m cmedalign.data.audit --data-dir data/normalized --out artifacts/audits/data_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cmedalign.data.dedup import cross_split_contamination, find_exact_duplicates, find_minhash_near_duplicates
from cmedalign.data.license_ledger import LicenseLedger


def load_split_texts(data_dir: Path, split: str) -> dict[str, str]:
    """Reads `{data_dir}/{split}.jsonl` of ConversationRecord-shaped lines and returns
    {sample_id: concatenated message text}."""
    path = data_dir / f"{split}.jsonl"
    texts: dict[str, str] = {}
    if not path.exists():
        return texts
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts[rec["sample_id"]] = "\n".join(m["content"] for m in rec["messages"])
    return texts


def run_audit(data_dir: Path, ledger_path: Path) -> dict:
    train = load_split_texts(data_dir, "train")
    dev = load_split_texts(data_dir, "dev")
    test = load_split_texts(data_dir, "test")

    ledger = LicenseLedger(ledger_path)
    quarantined = ledger.quarantined_sources()

    report = {
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "license_quarantined_sources": quarantined,
        "exact_dup_in_train": find_exact_duplicates(train),
        "train_test_contamination": [m.__dict__ for m in cross_split_contamination(train, test)],
        "train_dev_contamination": [m.__dict__ for m in cross_split_contamination(train, dev)],
        "near_dup_within_train_sample": (
            [m.__dict__ for m in find_minhash_near_duplicates(train)] if len(train) < 5000 else "skipped: run on server, train set too large for a quick local pass"
        ),
        "PASS": len(quarantined) == 0,
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, default=Path("data/manifests/license_ledger.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = run_audit(args.data_dir, args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if not report["PASS"] or report["train_test_contamination"] or report["train_dev_contamination"]:
        raise SystemExit(
            "G1 FAILED: license or contamination issues found, see " + str(args.out)
        )
    print(f"G1 PASS, report written to {args.out}")


if __name__ == "__main__":
    main()
