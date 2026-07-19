"""One-off manual smoke test for API baselines. Loads real credentials from .env (never
committed), makes one minimal chat call per configured alias, and reports whether the
reply looks like it contains raw <think> reasoning content that our pipeline must strip
before scoring/human-eval packaging. Not part of the automated test suite -- this talks
to real paid endpoints and costs real (tiny) money each run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cmedalign.eval.api_adapter import ConfigError, OpenAICompatibleClient, load_endpoint_config
from cmedalign.eval.human_pack import strip_thinking_trace


def load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)


def main() -> None:
    load_dotenv(Path(__file__).parent.parent / ".env")
    cache_dir = Path(__file__).parent.parent / "artifacts" / "logs" / "api_smoke_cache"

    for alias in ["DEEPSEEK_V4_PRO", "DEEPSEEK_V4_FLASH", "MINIMAX_M3"]:
        print(f"--- {alias} ---")
        try:
            cfg = load_endpoint_config(alias)
        except ConfigError as e:
            print(f"  SKIP (not configured): {e}")
            continue
        if cfg.base_url == "TODO_FILL_IN" or cfg.model == "TODO_FILL_IN":
            print("  SKIP: base_url or model still a TODO placeholder in .env")
            continue

        client = OpenAICompatibleClient(cfg, cache_dir=cache_dir, max_retries=2)
        try:
            result = client.chat(
                messages=[{"role": "user", "content": "用一句话回答：感冒和流感的主要区别是什么？"}],
                decoding={"max_tokens": 200, "temperature": 0},
            )
        except Exception as e:  # noqa: BLE001 - this is a manual diagnostic script
            print(f"  CALL FAILED: {type(e).__name__}: {e}")
            continue

        raw = result["raw_text"]
        stripped = strip_thinking_trace(raw)
        print(f"  server_model: {result['server_model']}")
        print(f"  finish_reason: {result['finish_reason']}")
        print(f"  had_think_tags: {raw != stripped}")
        print(f"  raw_text (first 300 chars): {raw[:300]!r}")
        if raw != stripped:
            print(f"  stripped_text (first 300 chars): {stripped[:300]!r}")


if __name__ == "__main__":
    main()
