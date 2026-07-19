import json

import pytest

from cmedalign.eval.human_pack import (
    build_blind_assignment,
    build_blinded_cases,
    decrypt_blind_map,
    encrypt_blind_map,
    generate_encryption_key,
    position_counts,
    strip_thinking_trace,
    write_blinded_cases_jsonl,
    write_encrypted_blind_map,
)

SYSTEMS = ["M0", "M1", "M2", "M3", "baseline-72B"]


def _case_ids(n):
    return [f"case_{i:03d}" for i in range(n)]


def test_blind_positions_balanced_exactly_over_80_cases():
    case_ids = _case_ids(80)
    assignment = build_blind_assignment(case_ids, SYSTEMS, seed=42)
    counts = position_counts(assignment, SYSTEMS)
    for sys_name in SYSTEMS:
        for label in ["A", "B", "C", "D", "E"]:
            assert counts[sys_name][label] == 16, (sys_name, label, counts[sys_name])


def test_blind_assignment_rejects_non_multiple_case_count():
    with pytest.raises(ValueError):
        build_blind_assignment(_case_ids(83), SYSTEMS, seed=1)


def test_blind_assignment_is_deterministic_given_seed():
    a1 = build_blind_assignment(_case_ids(20), SYSTEMS, seed=7)
    a2 = build_blind_assignment(_case_ids(20), SYSTEMS, seed=7)
    assert a1 == a2


def test_each_case_gets_a_full_permutation_of_systems():
    assignment = build_blind_assignment(_case_ids(10), SYSTEMS, seed=3)
    for case_id, label_map in assignment.items():
        assert set(label_map.values()) == set(SYSTEMS)
        assert set(label_map.keys()) == {"A", "B", "C", "D", "E"}


def test_strip_thinking_trace_removes_think_blocks():
    text = "<think>secret reasoning that must not reach raters</think>Final answer: take ibuprofen."
    cleaned = strip_thinking_trace(text)
    assert "secret reasoning" not in cleaned
    assert "Final answer" in cleaned


def test_blind_package_contains_no_model_identifiers(tmp_path):
    case_ids = _case_ids(5)
    assignment = build_blind_assignment(case_ids, SYSTEMS, seed=1)
    cases = {cid: [{"role": "user", "content": f"prompt for {cid}"}] for cid in case_ids}
    system_outputs = {
        cid: {sys_name: f"response from {sys_name} for {cid}" for sys_name in SYSTEMS}
        for cid in case_ids
    }
    forbidden = SYSTEMS + ["Qwen3-8B", "checkpoints/m2_dpo/step_4000"]

    # The synthetic outputs deliberately embed the system name, so this MUST raise --
    # proving the leak check actually works, not just that it exists.
    with pytest.raises(ValueError):
        build_blinded_cases(cases, system_outputs, assignment, forbidden)


def test_blind_package_passes_when_outputs_are_clean(tmp_path):
    case_ids = _case_ids(5)
    assignment = build_blind_assignment(case_ids, SYSTEMS, seed=1)
    cases = {cid: [{"role": "user", "content": f"prompt for {cid}"}] for cid in case_ids}
    system_outputs = {
        cid: {sys_name: f"a clinically reasonable response for {cid}" for sys_name in SYSTEMS}
        for cid in case_ids
    }
    forbidden = SYSTEMS + ["Qwen3-8B", "checkpoints/m2_dpo/step_4000"]

    blinded = build_blinded_cases(cases, system_outputs, assignment, forbidden)
    assert len(blinded) == 5

    out_path = tmp_path / "cases_blinded.jsonl"
    write_blinded_cases_jsonl(blinded, out_path)
    lines = out_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        rec = json.loads(line)
        blob = json.dumps(rec, ensure_ascii=False).lower()
        for ident in forbidden:
            assert ident.lower() not in blob


def test_blind_map_roundtrips_through_encryption(tmp_path):
    case_ids = _case_ids(5)
    assignment = build_blind_assignment(case_ids, SYSTEMS, seed=9)
    key = generate_encryption_key()

    path = tmp_path / "blind_map.enc"
    write_encrypted_blind_map(assignment, key, path)

    raw_bytes = path.read_bytes()
    # ciphertext must not contain plaintext system names
    assert b"M0" not in raw_bytes and b"baseline-72B" not in raw_bytes

    recovered = decrypt_blind_map(raw_bytes, key)
    assert recovered == assignment


def test_blind_map_wrong_key_fails_to_decrypt(tmp_path):
    from cryptography.fernet import InvalidToken

    assignment = build_blind_assignment(_case_ids(5), SYSTEMS, seed=2)
    key = generate_encryption_key()
    wrong_key = generate_encryption_key()
    token = encrypt_blind_map(assignment, key)

    with pytest.raises(InvalidToken):
        decrypt_blind_map(token, wrong_key)
