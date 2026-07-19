"""Qwen3 chat-template rendering and assistant-only loss masking.

Three verified, load-bearing findings from hand-testing the real Qwen3-8B tokenizer
(2026-07-19, see artifacts/audits/openrlhf_commit_notes.md and STATUS.md log) drive the
design of this module:

1. `tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)` silently
   returns an all-zero mask for Qwen3, because its shipped template has no
   `{% generation %}` block. transformers only prints a warning, it does not raise. Any
   code that trusted that flag would silently train on nothing. We compute the mask
   ourselves instead.

2. With `enable_thinking=False`, the template inserts a literal empty
   `<think>\n\n</think>\n\n` stub after `<|im_start|>assistant\n` -- but **only for the
   conversation's final message** (the one being generated / about to be generated).
   Earlier, already-completed assistant turns embedded as history get NO think wrapper
   at all (their thinking is stripped when re-serialized as history, matching how the
   model would see its own past turns at inference time). A masking implementation that
   assumes a uniform per-turn stub (e.g. by rendering `messages[:i]` with
   `add_generation_prompt=True` for every turn `i` in isolation, as if turn `i` is always
   "the final one") gets this wrong for every non-final assistant turn — it was tried
   here first and produced token-for-token mismatches against the real full-conversation
   render. The fix below scans the one true full-conversation tokenization directly and
   detects the stub per-occurrence, instead of assuming it.

3. `apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` (dict) by default
   in the installed transformers version, not a plain `list[int]` — silently breaks
   `len()`-based slicing if not caught. Always pass `return_dict=False` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = -100

_THINK_STUB_TEXT = "<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class MaskedExample:
    input_ids: list[int]
    labels: list[int]  # IGNORE_INDEX outside assistant spans


def render(tokenizer, messages: list[dict], enable_thinking: bool, add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def _tokenize_ids(tokenizer, messages: list[dict], enable_thinking: bool, add_generation_prompt: bool) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        return_dict=False,
    )


def build_masked_example(tokenizer, messages: list[dict], enable_thinking: bool) -> MaskedExample:
    """messages must end with an assistant turn (a complete training example).

    Scans the single true full-conversation tokenization for each
    `<|im_start|>assistant\\n` header, detects (rather than assumes) a following
    empty-think stub, and unmasks from there through that turn's closing `<|im_end|>`
    (inclusive, so the model learns to stop) -- but not the separator newline after it.
    """
    if messages[-1]["role"] != "assistant":
        raise ValueError("build_masked_example: last message must be from the assistant")

    full_ids = _tokenize_ids(tokenizer, messages, enable_thinking, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(full_ids)

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_stub_ids = tokenizer.encode(_THINK_STUB_TEXT, add_special_tokens=False)

    n = len(full_ids)
    j = 0
    found_assistant_turns = 0
    while j < n - 1:
        if full_ids[j] == im_start_id and tokenizer.decode([full_ids[j + 1]]).strip() == "assistant":
            content_start = j + 3  # <|im_start|>, "assistant", "\n"
            stub_len = len(think_stub_ids)
            if full_ids[content_start : content_start + stub_len] == think_stub_ids:
                content_start += stub_len

            content_end = content_start
            while content_end < n and full_ids[content_end] != im_end_id:
                content_end += 1
            if content_end >= n:
                raise ValueError("build_masked_example: assistant turn missing closing <|im_end|>")
            content_end += 1  # include <|im_end|> itself as a trainable label

            labels[content_start:content_end] = full_ids[content_start:content_end]
            found_assistant_turns += 1
            j = content_end
        else:
            j += 1

    expected_assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    if found_assistant_turns != expected_assistant_turns:
        raise ValueError(
            f"build_masked_example: found {found_assistant_turns} assistant turns in the "
            f"tokenized sequence but expected {expected_assistant_turns} -- masking is "
            "unreliable, refusing to proceed silently"
        )

    return MaskedExample(input_ids=full_ids, labels=labels)
