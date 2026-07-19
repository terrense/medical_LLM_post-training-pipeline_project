#!/usr/bin/env python3
"""把 data-juicer 清洗后的保留集合(按 sample_id)映射回原始结构化 JSONL 记录。

data-juicer 的输出(`_dj_cleaned.jsonl`)里每条记录是拍平后的 {sample_id, text}，
经过它的 filter/dedup 之后，剩下的 sample_id 集合就是"被判定为该保留"的样本。
这一步只做"保留哪些 sample_id"的映射，不采用 data-juicer 输出的拍平文本本身
(那只是给算子用的中间表示，不是最终训练数据)。

用法: python rejoin_after_dj.py --original data/raw/some_source.jsonl \
    --dj-cleaned data/normalized/_dj_cleaned.jsonl \
    --output data/normalized/some_source_dj_cleaned.jsonl \
    --report data/manifests/dj_retention_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, nargs="+")
    ap.add_argument("--dj-cleaned", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    kept_ids = set()
    with open(args.dj_cleaned, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                kept_ids.add(json.loads(line)["sample_id"])

    n_in = 0
    n_out = 0
    by_source: dict[str, dict[str, int]] = {}
    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.original:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    n_in += 1
                    src = rec.get("source", "?")
                    by_source.setdefault(src, {"in": 0, "out": 0})
                    by_source[src]["in"] += 1
                    if rec["sample_id"] in kept_ids:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n_out += 1
                        by_source[src]["out"] += 1

    report = {
        "n_in": n_in,
        "n_out": n_out,
        "retention_rate": round(n_out / n_in, 4) if n_in else 0,
        "by_source": by_source,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
