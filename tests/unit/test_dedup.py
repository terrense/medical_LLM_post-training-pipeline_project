from cmedalign.data.dedup import (
    cross_split_contamination,
    exact_hash,
    find_embedding_near_duplicates,
    find_exact_duplicates,
    find_minhash_near_duplicates,
)


def test_exact_hash_is_whitespace_insensitive():
    assert exact_hash("hello   world") == exact_hash("hello world")


def test_exact_hash_is_case_insensitive():
    assert exact_hash("Hello World") == exact_hash("hello world")


def test_find_exact_duplicates_groups_matches():
    records = {
        "a": "the patient reports a headache for three days",
        "b": "The patient reports a headache for three days",
        "c": "completely unrelated text about something else",
    }
    dupes = find_exact_duplicates(records)
    assert len(dupes) == 1
    group = next(iter(dupes.values()))
    assert set(group) == {"a", "b"}


def test_minhash_near_duplicates_catches_light_paraphrase():
    records = {
        "a": "患者主诉发热三天，伴有咳嗽和乏力，无胸痛" * 3,
        "b": "患者主诉发热三天，伴有咳嗽和乏力，无胸痛，无恶心" * 3,
        "c": "完全不同的另一段文本内容，讨论财务报表和季度收入" * 3,
    }
    matches = find_minhash_near_duplicates(records, threshold=0.5)
    matched_pairs = {frozenset((m.query_id, m.match_id)) for m in matches}
    assert frozenset(("a", "b")) in matched_pairs
    assert frozenset(("a", "c")) not in matched_pairs


def test_cross_split_contamination_detects_test_leak():
    train = {"t1": "患者主诉头痛三天，伴恶心呕吐，无发热"}
    test_clean = {"e1": "完全不同的病例，关于骨折和外伤"}
    test_leak = {"e2": "患者主诉头痛三天，伴恶心呕吐，无发热"}

    assert cross_split_contamination(train, test_clean, threshold=0.5) == []
    leaked = cross_split_contamination(train, test_leak, threshold=0.5)
    assert len(leaked) == 1
    assert {leaked[0].query_id, leaked[0].match_id} == {"t1", "e2"}


def test_embedding_near_duplicates_with_fake_embedder():
    records = {"a": "x", "b": "y", "c": "z"}

    def fake_embed(texts: list[str]) -> list[list[float]]:
        # "a" and "b" map to (nearly) identical vectors, "c" is orthogonal.
        mapping = {"x": [1.0, 0.0], "y": [0.99, 0.01], "z": [0.0, 1.0]}
        return [mapping[t] for t in texts]

    matches = find_embedding_near_duplicates(records, fake_embed, threshold=0.9)
    pairs = {frozenset((m.query_id, m.match_id)) for m in matches}
    assert frozenset(("a", "b")) in pairs
    assert frozenset(("a", "c")) not in pairs
