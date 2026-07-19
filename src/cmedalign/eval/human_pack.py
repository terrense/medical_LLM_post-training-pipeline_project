"""Blind human-evaluation packaging.

Builds a position-balanced, randomized A-E assignment across (M0, M1, M2, M3, strongest
external baseline) for the 80-case human eval, and the corresponding blinded case file.
The unblinding key (`blind_map`) is the single most sensitive artifact this project
produces before scores are frozen — it must be encrypted at rest and never enter the
analysis directory unencrypted before rating is complete.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet

LABELS_POOL = ["A", "B", "C", "D", "E"]

_THINKING_TRACE_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking_trace(text: str) -> str:
    return _THINKING_TRACE_RE.sub("", text).strip()


def build_blind_assignment(
    case_ids: list[str], systems: list[str], seed: int
) -> dict[str, dict[str, str]]:
    """Returns {case_id: {label: system_name}}, exactly position-balanced across systems
    when len(case_ids) is a multiple of len(systems) (built from Latin-square blocks, so
    balance holds by construction, not by chance)."""
    import random

    if len(case_ids) % len(systems) != 0:
        raise ValueError(
            f"build_blind_assignment: {len(case_ids)} cases is not a multiple of "
            f"{len(systems)} systems — exact position balance is not achievable"
        )

    rng = random.Random(seed)
    labels = LABELS_POOL[: len(systems)]
    n = len(systems)
    assignment: dict[str, dict[str, str]] = {}

    for block_start in range(0, len(case_ids), n):
        block_cases = case_ids[block_start : block_start + n]
        base_perm = systems[:]
        rng.shuffle(base_perm)
        rows = [base_perm[i:] + base_perm[:i] for i in range(n)]  # cyclic shifts = latin square
        rng.shuffle(rows)
        for case_id, row in zip(block_cases, rows):
            assignment[case_id] = dict(zip(labels, row))

    return assignment


def position_counts(assignment: dict[str, dict[str, str]], systems: list[str]) -> dict[str, dict[str, int]]:
    counts = {sys_name: {label: 0 for label in LABELS_POOL} for sys_name in systems}
    for case_map in assignment.values():
        for label, sys_name in case_map.items():
            counts[sys_name][label] += 1
    return counts


@dataclass
class BlindedCase:
    case_id: str
    prompt: list[dict]
    outputs: dict[str, str]  # label -> text, NO system identity


def build_blinded_cases(
    cases: dict[str, list[dict]],  # case_id -> prompt messages
    system_outputs: dict[str, dict[str, str]],  # case_id -> {system_name: output_text}
    assignment: dict[str, dict[str, str]],
    forbidden_identifiers: list[str],
) -> list[BlindedCase]:
    """Raises ValueError if any forbidden identifier (model name, checkpoint path, ...)
    is found verbatim (case-insensitive) in the prompt or any output text."""
    blinded: list[BlindedCase] = []
    for case_id, label_to_system in assignment.items():
        outputs = {}
        for label, sys_name in label_to_system.items():
            raw_text = system_outputs[case_id][sys_name]
            cleaned = strip_thinking_trace(raw_text)
            outputs[label] = cleaned

        haystack = json.dumps(cases[case_id], ensure_ascii=False) + " " + " ".join(outputs.values())
        haystack_lower = haystack.lower()
        for ident in forbidden_identifiers:
            if ident and ident.lower() in haystack_lower:
                raise ValueError(
                    f"case {case_id}: forbidden identifier '{ident}' found in blinded package content"
                )

        blinded.append(BlindedCase(case_id=case_id, prompt=cases[case_id], outputs=outputs))
    return blinded


def write_blinded_cases_jsonl(blinded: list[BlindedCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for case in blinded:
            f.write(
                json.dumps(
                    {"case_id": case.case_id, "prompt": case.prompt, "outputs": case.outputs},
                    ensure_ascii=False,
                )
                + "\n"
            )


def generate_encryption_key() -> bytes:
    return Fernet.generate_key()


def encrypt_blind_map(assignment: dict[str, dict[str, str]], key: bytes) -> bytes:
    payload = json.dumps(assignment, ensure_ascii=False).encode("utf-8")
    return Fernet(key).encrypt(payload)


def decrypt_blind_map(token: bytes, key: bytes) -> dict[str, dict[str, str]]:
    payload = Fernet(key).decrypt(token)
    return json.loads(payload.decode("utf-8"))


def write_encrypted_blind_map(assignment: dict[str, dict[str, str]], key: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encrypt_blind_map(assignment, key))
