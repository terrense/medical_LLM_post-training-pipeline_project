import pytest


@pytest.fixture(scope="session")
def qwen3_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    except Exception as e:  # noqa: BLE001 - broad on purpose: any offline-cache miss should skip, not fail
        pytest.skip(f"Qwen3-8B tokenizer not available in local HF cache: {e}")
    return tok
