#!/usr/bin/env python3
"""把 ConversationRecord 风格的多轮 JSONL 拍平成 data-juicer 需要的单字段 text 格式。

data-juicer 的算子(whitespace/length/repetition/dedup 等)都是针对单一 text 字段设计的，
不理解我们的 messages[{role,content}] 结构。这里只做"拼接成一段文本供质量判断/去重用"，
不修改、不覆盖原始结构化记录——真正参与训练的还是原始 messages 结构，data-juicer 只负责
"这条样本该留还是该丢"这个决定，由 rejoin_after_dj.py 映射回去。

用法: python flatten_for_dj.py --input data/raw/some_source.jsonl --output data/normalized/_dj_flattened.jsonl
"""
from __future__ import annotations

import argparse
import json


def flatten_record(rec: dict) -> dict:
    text = "\n".join(f"[{m['role']}] {m['content']}" for m in rec["messages"])
    return {"sample_id": rec["sample_id"], "text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, nargs="+", help="one or more raw JSONL files")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path in args.input:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    out.write(json.dumps(flatten_record(rec), ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} flattened records to {args.output}")


if __name__ == "__main__":
    main()
