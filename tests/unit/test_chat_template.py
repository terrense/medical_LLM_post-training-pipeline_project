from cmedalign.data.chat_template import IGNORE_INDEX, build_masked_example, render

SYSTEM = "You are a careful medical assistant. You are not a substitute for a real doctor."


def test_chat_template_exact_qwen3(qwen3_tokenizer):
    msgs = [{"role": "user", "content": "I have a headache."}]
    rendered = render(qwen3_tokenizer, msgs, enable_thinking=False, add_generation_prompt=True)
    assert rendered == (
        "<|im_start|>user\nI have a headache.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_non_thinking_flag_is_recorded_in_rendered_text(qwen3_tokenizer):
    msgs = [{"role": "user", "content": "I have a headache."}]
    non_thinking = render(qwen3_tokenizer, msgs, enable_thinking=False, add_generation_prompt=True)
    thinking = render(qwen3_tokenizer, msgs, enable_thinking=True, add_generation_prompt=True)

    # enable_thinking=False forces an explicit empty <think></think> stub right at the
    # start of the assistant turn -- so the flag's effect is externally verifiable from
    # the rendered string alone, which is what we save in RawGeneration.enable_thinking.
    assert "<think>\n\n</think>\n\n" in non_thinking
    assert non_thinking.endswith("<think>\n\n</think>\n\n")
    # enable_thinking=True leaves the assistant turn fully open (no forced empty stub).
    assert thinking.endswith("<|im_start|>assistant\n")
    assert "<think>" not in thinking


def test_assistant_only_mask_single_turn(qwen3_tokenizer):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "I have a headache."},
        {"role": "assistant", "content": "How long has it lasted, and is it worse with light?"},
    ]
    example = build_masked_example(qwen3_tokenizer, msgs, enable_thinking=False)
    assert len(example.input_ids) == len(example.labels)

    # everything up to and including the empty <think></think> stub must be masked
    trainable_ids = [t for t, l in zip(example.input_ids, example.labels) if l != IGNORE_INDEX]
    trainable_text = qwen3_tokenizer.decode(trainable_ids)
    assert "<think>" not in trainable_text
    assert "system" not in trainable_text.lower()
    assert "headache" not in trainable_text  # that's in the user turn, must stay masked
    assert "How long has it lasted" in trainable_text


def test_assistant_only_mask_multi_turn(qwen3_tokenizer):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "I have a headache."},
        {"role": "assistant", "content": "How long has it lasted?"},
        {"role": "user", "content": "About two days."},
        {"role": "assistant", "content": "Any nausea or visual changes?"},
    ]
    example = build_masked_example(qwen3_tokenizer, msgs, enable_thinking=False)
    trainable_ids = [t for t, l in zip(example.input_ids, example.labels) if l != IGNORE_INDEX]
    trainable_text = qwen3_tokenizer.decode(trainable_ids)

    assert "How long has it lasted?" in trainable_text
    assert "Any nausea or visual changes?" in trainable_text
    assert "About two days" not in trainable_text  # user turn, must stay masked
    assert trainable_text.count("<think>") == 0


def test_eot_token_included_in_assistant_labels(qwen3_tokenizer):
    msgs = [
        {"role": "user", "content": "I have a headache."},
        {"role": "assistant", "content": "How long has it lasted?"},
    ]
    example = build_masked_example(qwen3_tokenizer, msgs, enable_thinking=False)
    im_end_id = qwen3_tokenizer.convert_tokens_to_ids("<|im_end|>")

    # the assistant turn's closing <|im_end|> must be a trainable label, not masked --
    # otherwise the model never learns to stop generating.
    assistant_label_ids = [l for l in example.labels if l != IGNORE_INDEX]
    assert im_end_id in assistant_label_ids


def test_build_masked_example_requires_assistant_last(qwen3_tokenizer):
    import pytest

    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError):
        build_masked_example(qwen3_tokenizer, msgs, enable_thinking=False)
