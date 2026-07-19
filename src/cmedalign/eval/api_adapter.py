"""Generic OpenAI-compatible chat completion client for the API baselines
(DeepSeek-V4-Pro/Flash, MiniMax-M3, ...).

Design constraints from the execution spec:
  - base_url / api_key / model alias are read ONLY from environment variables, never
    hardcoded, never printed (not even in debug logs).
  - retries use exponential backoff, but only for transient errors (timeouts,
    connection errors, 5xx). A content-filter/safety refusal is a real model result,
    not a network failure — it must be recorded, not retried forever.
  - every response is cached on disk keyed by a hash of the exact request, so re-running
    a retried/crashed job never double-calls (and never double-bills) the same request.
  - the server's returned `model` field is always saved alongside our configured alias,
    since the paper-facing name and the API's own model id can differ.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger("cmedalign.api_adapter")


class ConfigError(RuntimeError):
    pass


class TransientAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class EndpointConfig:
    alias: str  # e.g. "DEEPSEEK_V4_PRO"
    base_url: str
    api_key: str
    model: str


def load_endpoint_config(alias: str, env: Optional[dict] = None) -> EndpointConfig:
    """Reads `{alias}_BASE_URL`, `{alias}_API_KEY`, `{alias}_MODEL` from the environment
    (or an injected mapping, for tests). Raises ConfigError with NO secret material in
    the message if anything is missing."""
    env = env if env is not None else os.environ
    base_url = env.get(f"{alias}_BASE_URL")
    api_key = env.get(f"{alias}_API_KEY")
    model = env.get(f"{alias}_MODEL")
    missing = [
        name
        for name, val in (("BASE_URL", base_url), ("API_KEY", api_key), ("MODEL", model))
        if not val
    ]
    if missing:
        raise ConfigError(f"{alias}: missing env vars {[f'{alias}_{m}' for m in missing]}")
    return EndpointConfig(alias=alias, base_url=base_url, api_key=api_key, model=model)


def redact(text: str, secrets: list[str]) -> str:
    out = text
    for s in secrets:
        if s:
            out = out.replace(s, "***REDACTED***")
    return out


def _request_hash(model: str, messages: list[dict], decoding: dict, seed: Optional[int]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "decoding": decoding, "seed": seed},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAICompatibleClient:
    def __init__(
        self,
        config: EndpointConfig,
        cache_dir: Path,
        max_retries: int = 5,
        timeout_s: float = 60.0,
        backoff_base_s: float = 1.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        post_fn: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.backoff_base_s = backoff_base_s
        self.sleep_fn = sleep_fn
        # injectable for tests; defaults to a real POST against config.base_url
        self._post_fn = post_fn or self._default_post

    def _default_post(self, url: str, headers: dict, json_body: dict) -> requests.Response:
        return requests.post(url, headers=headers, json=json_body, timeout=self.timeout_s)

    def _cache_path(self, req_hash: str) -> Path:
        return self.cache_dir / f"{req_hash}.json"

    def log_safe(self, msg: str) -> None:
        logger.info(redact(msg, [self.config.api_key]))

    def chat(
        self,
        messages: list[dict],
        decoding: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> dict:
        """Returns a dict shaped like a trimmed RawGeneration: model (configured alias),
        server_model (what the API actually reported), raw_text, finish_reason,
        token_usage, accessed_at, from_cache."""
        decoding = decoding or {}
        req_hash = _request_hash(self.config.model, messages, decoding, seed)
        cache_path = self._cache_path(req_hash)
        if cache_path.exists():
            self.log_safe(f"[{self.config.alias}] cache hit for request {req_hash[:12]}")
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["from_cache"] = True
            return cached

        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        body = {"model": self.config.model, "messages": messages, **decoding}
        if seed is not None:
            body["seed"] = seed

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                self.log_safe(
                    f"[{self.config.alias}] POST {self.config.base_url} attempt={attempt + 1}"
                )
                resp = self._post_fn(url, headers=headers, json_body=body)
                if resp.status_code >= 500:
                    raise TransientAPIError(f"server error {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                break
            except (TransientAPIError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt == self.max_retries - 1:
                    raise TransientAPIError(
                        f"[{self.config.alias}] exhausted {self.max_retries} retries"
                    ) from e
                self.sleep_fn(self.backoff_base_s * (2**attempt))
        else:  # pragma: no cover - loop always breaks or raises
            raise TransientAPIError(f"[{self.config.alias}] unexpected retry exit") from last_err

        choice = data["choices"][0]
        result = {
            "configured_model_alias": self.config.model,
            "server_model": data.get("model"),
            "raw_text": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason", "unknown"),
            "token_usage": data.get("usage", {}),
            "accessed_at": datetime.now(timezone.utc).isoformat(),
            "from_cache": False,
        }
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
