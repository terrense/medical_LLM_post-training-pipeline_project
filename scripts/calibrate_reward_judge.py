#!/usr/bin/env python3
"""抽样检查: 用第二个裁判模型重新给一批已评分的 episode 打分,计算两个裁判的一致性
(Spearman correlation)——不需要给每一次 GRPO rollout 都打两遍分,只需要定期抽样验证
"当前用来算 reward 的裁判"和"另一个独立裁判"是否基本一致。对应论文 main.tex 里
"at least two judge families... report per-judge results, consensus... correlation"
的要求,也是用户说的"抽样检查打分"。

输入: 一个 JSONL 文件, 每行是一个已完成的 episode (profile + history), 通常是训练过程中
按固定间隔抽样保存下来的 (不是每条都存, 抽样保存这一步本身也要做, 这里假设已经有了)。

用法:
    python scripts/calibrate_reward_judge.py \
        --episodes artifacts/logs/sampled_episodes.jsonl \
        --judge-a DEEPSEEK_V4_PRO --judge-b MINIMAX_M3 \
        --out artifacts/audits/judge_calibration_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy.stats import spearmanr

from cmedalign.data.schemas import Message, PatientProfile
from cmedalign.eval.api_adapter import OpenAICompatibleClient, load_endpoint_config
from cmedalign.rewards.judge_scorer import score_episode_with_judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="JSONL: each line {profile: {...}, history: [{role,content},...]}")
    ap.add_argument("--judge-a", default="DEEPSEEK_V4_PRO")
    ap.add_argument("--judge-b", default="MINIMAX_M3")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client_a = OpenAICompatibleClient(load_endpoint_config(args.judge_a), cache_dir=Path("artifacts/logs/judge_calib_a"))
    client_b = OpenAICompatibleClient(load_endpoint_config(args.judge_b), cache_dir=Path("artifacts/logs/judge_calib_b"))

    scores_a: dict[str, list[float]] = {"clinical": [], "safety": [], "process": [], "communication": []}
    scores_b: dict[str, list[float]] = {"clinical": [], "safety": [], "process": [], "communication": []}
    n_unparseable_a = n_unparseable_b = 0
    n = 0

    with open(args.episodes, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            profile = PatientProfile(**rec["profile"])
            history = [Message(**m) for m in rec["history"]]

            sa = score_episode_with_judge(client_a, profile, history)
            sb = score_episode_with_judge(client_b, profile, history)
            n_unparseable_a += sa.parser_status == "unparseable"
            n_unparseable_b += sb.parser_status == "unparseable"

            scores_a["clinical"].append(sa.clinical_correctness)
            scores_a["safety"].append(sa.safety_score)
            scores_a["process"].append(sa.process_score)
            scores_a["communication"].append(sa.communication_score)
            scores_b["clinical"].append(sb.clinical_correctness)
            scores_b["safety"].append(sb.safety_score)
            scores_b["process"].append(sb.process_score)
            scores_b["communication"].append(sb.communication_score)
            n += 1

    report = {
        "judge_a": args.judge_a,
        "judge_b": args.judge_b,
        "n_episodes": n,
        "n_unparseable_judge_a": n_unparseable_a,
        "n_unparseable_judge_b": n_unparseable_b,
        "spearman_by_component": {},
    }
    for component in scores_a:
        if n >= 3:
            rho, p = spearmanr(scores_a[component], scores_b[component])
            report["spearman_by_component"][component] = {"rho": rho, "p_value": p}
        else:
            report["spearman_by_component"][component] = {"rho": None, "p_value": None, "note": "n<3, not enough samples"}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
