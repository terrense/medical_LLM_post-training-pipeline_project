"""OpenRLHF `--train.agent_func_path` bridge for the GRPO multi-turn patient-consultation
environment (paper's M2->M3 stage, per cmedalign_paper/main.tex §Stage 3).

This is a SCAFFOLD, not yet a complete, tested implementation — it needs a real GPU
(vLLM-served policy + a separately-served frozen Qwen2.5-7B-Instruct patient model) to
run at all, so it can't be exercised end-to-end on this Windows prep machine. What IS
done and reused here without modification: the hidden/visible profile boundary
(`cmedalign.agents.patient`), the deterministic parts of the reward
(`cmedalign.rewards.components`), the generic OpenAI-compatible client
(`cmedalign.eval.api_adapter`) for the frozen patient model, and — as of 2026-07-19 —
LLM-judge scoring of the open-ended reward components via
`cmedalign.rewards.judge_scorer` (unit-tested with mocked API responses; the live judge
model defaults to the already-verified-working DEEPSEEK_V4_PRO alias, swappable via
env var, see `_reward_judge_client()` below). What's still a TODO: this hasn't been
exercised against a *real* rollout yet (needs the GPU/vLLM side), and the judge prompt
hasn't been calibrated against real transcripts or a second judge — see
`scripts/calibrate_reward_judge.py` for the sampling-based cross-judge calibration
check, which should be run once real episodes exist, before trusting this judge's
scores for actual training.

Reference implementation this was written against:
vendor/OpenRLHF/examples/python/agent_func_openai_server_executor.py (see
artifacts/audits/openrlhf_commit_notes.md for the exact pinned-commit interface:
`AgentExecutor.run_agent(self, prompt, label, session_id)` must return
`{"reward": float, "scores": {...}, "extra_logs": {...}}`; `self.client` is the *policy
being trained*, served locally by OpenRLHF/vLLM — never call it for the patient side,
only for doctor turns, so token traces for the correct model get collected).

Usage (once wired to a real training run):
    --train.agent_func_path src/cmedalign/agents/openrlhf_agent_func.py
    (plus env vars: CMEDALIGN_PROFILES_PATH, PATIENT_SIM_BASE_URL, PATIENT_SIM_API_KEY,
    PATIENT_SIM_MODEL — the frozen patient simulator is served as its own
    OpenAI-compatible endpoint, e.g. a separate vLLM instance for Qwen2.5-7B-Instruct;
    optionally CMEDALIGN_REWARD_JUDGE_ALIAS to override the default DEEPSEEK_V4_PRO judge)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from cmedalign.agents.patient import build_patient_visible_profile
from cmedalign.data.schemas import Message, PatientProfile
from cmedalign.eval.api_adapter import EndpointConfig, OpenAICompatibleClient, load_endpoint_config
from cmedalign.rewards.components import EpisodeFeatures, total_reward
from cmedalign.rewards.judge_scorer import score_episode_with_judge

FINAL_MARKER = "FINAL:"

try:
    from openrlhf.utils.agent import AgentExecutorBase
except ImportError:  # pragma: no cover - only importable inside the real training env
    class AgentExecutorBase:  # type: ignore[no-redef]
        """Stand-in so this module can at least be imported/linted without OpenRLHF
        installed (e.g. on this Windows prep machine). Never used for real training."""


_PROFILES_CACHE: Optional[dict[str, PatientProfile]] = None


def _load_profiles() -> dict[str, PatientProfile]:
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        path = Path(os.environ["CMEDALIGN_PROFILES_PATH"])
        profiles = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    profiles[rec["profile_id"]] = PatientProfile(**rec)
        _PROFILES_CACHE = profiles
    return _PROFILES_CACHE


def _patient_client() -> OpenAICompatibleClient:
    cfg = EndpointConfig(
        alias="PATIENT_SIM",
        base_url=os.environ["PATIENT_SIM_BASE_URL"],
        api_key=os.environ.get("PATIENT_SIM_API_KEY", "EMPTY"),
        model=os.environ["PATIENT_SIM_MODEL"],
    )
    return OpenAICompatibleClient(cfg, cache_dir=Path("artifacts/logs/patient_sim_cache"))


def _reward_judge_client() -> OpenAICompatibleClient:
    # Default judge is DEEPSEEK_V4_PRO because it's the one alias already confirmed
    # working end-to-end (see STATUS.md 2026-07-19). Override via
    # CMEDALIGN_REWARD_JUDGE_ALIAS once real Claude/GPT *API* credentials (not a
    # ChatGPT/Claude.ai subscription -- see module docstring) are configured.
    alias = os.environ.get("CMEDALIGN_REWARD_JUDGE_ALIAS", "DEEPSEEK_V4_PRO")
    cfg = load_endpoint_config(alias)
    return OpenAICompatibleClient(cfg, cache_dir=Path("artifacts/logs/reward_judge_cache"))


def _patient_system_prompt(visible_profile: dict) -> str:
    # Deliberately does NOT include target_assessment/acceptable_actions/forbidden_claims/
    # required_info weights -- see cmedalign.agents.patient.PATIENT_VISIBLE_FIELDS.
    return (
        "你在扮演一名患者，只能依据下面给出的自身信息真实回答医生的问题。"
        "如果被问到这里没有的信息，就说不知道/不确定，不要编造。"
        "用口语化的中文回答，不要主动透露没被问到的信息。\n\n"
        f"你的情况：{json.dumps(visible_profile, ensure_ascii=False)}"
    )


def _score_episode(profile: PatientProfile, history: list[Message], revealed_info: set[str], num_turns: int, turn_budget: int) -> EpisodeFeatures:
    """clinical_correctness/safety_score/process_score/communication_score now come from
    a real LLM judge call (cmedalign.rewards.judge_scorer), not a placeholder — but this
    judge has NOT yet been exercised against a real rollout (needs GPU/vLLM) or
    calibrated against a second judge (see scripts/calibrate_reward_judge.py). Treat
    scores from this as provisional until that calibration has actually run.
    """
    weighted_total = sum(item.weight for item in profile.required_info)
    revealed_weight = sum(item.weight for item in profile.required_info if item.item in revealed_info)
    info_coverage = (revealed_weight / weighted_total) if weighted_total else 0.0

    judge_client = _reward_judge_client()
    judge_scores = score_episode_with_judge(judge_client, profile, history)

    # Heuristic: a red flag counts as "handled" only if it also appears verbatim among
    # the revealed required_info items (this assumes red-flag descriptions are also
    # listed as required_info items in profile construction, which is the convention
    # used in this project's test fixtures -- re-verify against real GRPO profiles once
    # they're built, this is not yet validated against real data).
    red_flag_missed = any(rf not in revealed_info for rf in profile.red_flags)

    return EpisodeFeatures(
        clinical_correctness=judge_scores.clinical_correctness,
        info_coverage=info_coverage,  # real, computed from revealed required_info
        safety_score=judge_scores.safety_score,
        process_score=judge_scores.process_score,
        communication_score=judge_scores.communication_score,
        num_turns=num_turns,
        turn_budget=turn_budget,
        response_length_tokens=sum(len(m.content) for m in history if m.role == "assistant"),
        length_budget=int(os.environ.get("CMEDALIGN_LENGTH_BUDGET", "2000")),  # TODO: freeze from training-only dev tuning, see main.tex Table hyper
        red_flag_missed=red_flag_missed,  # heuristic: red-flag string not among revealed_info items; refine once real profiles exist
        role_leakage=False,  # TODO: wire in cmedalign.agents.patient's leak-detection check against the doctor's own turns
    )


class AgentExecutor(AgentExecutorBase):
    """Overrides run_agent() for the doctor<->patient multi-turn GRPO episode.

    `label` is expected to be a `profile_id` string (one GRPO "prompt" == one hidden
    patient profile). `prompt` is the doctor's system/opening instruction.
    """

    async def run_agent(self, prompt: str, label: str, session_id: str) -> dict:
        profile = _load_profiles()[label]
        turn_budget = int(os.environ.get("CMEDALIGN_TURN_BUDGET_DEFAULT", "8"))

        patient_client = _patient_client()
        visible_profile = build_patient_visible_profile(profile)
        patient_system_prompt = _patient_system_prompt(visible_profile)

        history: list[Message] = []
        revealed: set[str] = set()
        num_turns = 0
        terminated_reason = "turn_budget"

        for turn in range(1, turn_budget + 1):
            num_turns = turn
            doctor_messages = [{"role": "system", "content": prompt}] + [
                {"role": m.role, "content": m.content} for m in history
            ]
            doctor_resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=doctor_messages,
                max_tokens=self.sampling_params.max_tokens,
                temperature=self.sampling_params.temperature,
                top_p=self.sampling_params.top_p,
                logprobs=self.sampling_params.logprobs is not None,
                top_logprobs=self.sampling_params.logprobs or 1,
                extra_body={"session_id": session_id},
            )
            doctor_text = doctor_resp.choices[0].message.content
            history.append(Message(role="assistant", content=doctor_text))

            if doctor_text.strip().startswith(FINAL_MARKER):
                terminated_reason = "final_marker"
                break

            patient_result = patient_client.chat(
                messages=[
                    {"role": "system", "content": patient_system_prompt},
                    *[{"role": "user" if m.role == "assistant" else "assistant", "content": m.content} for m in history],
                ],
                decoding={"temperature": 0.7, "max_tokens": 300},
            )
            patient_text = patient_result["raw_text"]
            history.append(Message(role="user", content=patient_text))

            for item in profile.required_info:
                if item.item not in revealed and item.item.lower() in patient_text.lower():
                    revealed.add(item.item)

        features = _score_episode(profile, history, revealed, num_turns, turn_budget)
        reward = total_reward(features)

        return {
            "reward": reward,
            "scores": {
                "clinical": features.clinical_correctness,
                "info": features.info_coverage,
                "safety": features.safety_score,
                "process": features.process_score,
                "communication": features.communication_score,
            },
            "extra_logs": {
                "num_turns": num_turns,
                "terminated_reason": terminated_reason,
                "profile_id": label,
                "revealed_info_count": len(revealed),
            },
        }
