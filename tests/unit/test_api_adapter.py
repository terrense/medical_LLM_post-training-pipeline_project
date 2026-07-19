import logging

import pytest
import requests

from cmedalign.eval.api_adapter import (
    ConfigError,
    OpenAICompatibleClient,
    TransientAPIError,
    load_endpoint_config,
    redact,
)


class FakeResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _fake_env():
    return {
        "DEEPSEEK_V4_PRO_BASE_URL": "https://api.example.com/v1",
        "DEEPSEEK_V4_PRO_API_KEY": "sk-supersecretvalue123",
        "DEEPSEEK_V4_PRO_MODEL": "deepseek-v4-pro-alias",
    }


def test_load_endpoint_config_success():
    cfg = load_endpoint_config("DEEPSEEK_V4_PRO", env=_fake_env())
    assert cfg.base_url == "https://api.example.com/v1"
    assert cfg.model == "deepseek-v4-pro-alias"


def test_load_endpoint_config_missing_raises_without_leaking():
    env = _fake_env()
    del env["DEEPSEEK_V4_PRO_API_KEY"]
    with pytest.raises(ConfigError) as exc_info:
        load_endpoint_config("DEEPSEEK_V4_PRO", env=env)
    assert "API_KEY" in str(exc_info.value)
    assert "sk-" not in str(exc_info.value)


def test_redact_masks_secret():
    out = redact("Authorization: Bearer sk-supersecretvalue123", ["sk-supersecretvalue123"])
    assert "sk-supersecretvalue123" not in out
    assert "REDACTED" in out


def test_api_retry_then_success_then_idempotent_cache(tmp_path):
    cfg = load_endpoint_config("DEEPSEEK_V4_PRO", env=_fake_env())
    calls = []

    def fake_post(url, headers, json_body):
        calls.append(1)
        if len(calls) == 1:
            return FakeResponse(500)
        return FakeResponse(
            200,
            {
                "model": "deepseek-v4-pro-2026-06-01",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
        )

    client = OpenAICompatibleClient(
        cfg, cache_dir=tmp_path, sleep_fn=lambda s: None, post_fn=fake_post
    )
    messages = [{"role": "user", "content": "hi"}]

    result1 = client.chat(messages, decoding={"temperature": 0})
    assert result1["raw_text"] == "hello"
    assert result1["server_model"] == "deepseek-v4-pro-2026-06-01"
    assert result1["from_cache"] is False
    assert len(calls) == 2  # one 500 retry + one success

    # Identical request again -- must hit the cache, not the network.
    result2 = client.chat(messages, decoding={"temperature": 0})
    assert result2["from_cache"] is True
    assert result2["raw_text"] == "hello"
    assert len(calls) == 2  # unchanged: no new HTTP call was made


def test_non_retryable_client_error_not_retried(tmp_path):
    cfg = load_endpoint_config("DEEPSEEK_V4_PRO", env=_fake_env())
    calls = []

    def fake_post(url, headers, json_body):
        calls.append(1)
        return FakeResponse(400, {"error": "content filter triggered"})

    client = OpenAICompatibleClient(
        cfg, cache_dir=tmp_path, sleep_fn=lambda s: None, post_fn=fake_post
    )
    with pytest.raises(requests.HTTPError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1  # no retry loop for a 4xx


def test_exhausts_retries_on_persistent_5xx(tmp_path):
    cfg = load_endpoint_config("DEEPSEEK_V4_PRO", env=_fake_env())
    calls = []

    def fake_post(url, headers, json_body):
        calls.append(1)
        return FakeResponse(503)

    client = OpenAICompatibleClient(
        cfg, cache_dir=tmp_path, max_retries=3, sleep_fn=lambda s: None, post_fn=fake_post
    )
    with pytest.raises(TransientAPIError):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 3


def test_api_logs_redact_secrets(tmp_path, caplog):
    cfg = load_endpoint_config("DEEPSEEK_V4_PRO", env=_fake_env())

    def fake_post(url, headers, json_body):
        return FakeResponse(
            200,
            {
                "model": "deepseek-v4-pro-2026-06-01",
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    client = OpenAICompatibleClient(
        cfg, cache_dir=tmp_path, sleep_fn=lambda s: None, post_fn=fake_post
    )
    with caplog.at_level(logging.INFO, logger="cmedalign.api_adapter"):
        client.chat([{"role": "user", "content": "hi"}])

    assert cfg.api_key not in caplog.text
