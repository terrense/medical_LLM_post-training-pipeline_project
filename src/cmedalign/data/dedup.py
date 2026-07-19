"""Exact + near-duplicate detection for conversation records.

Three tiers, matching G1's "精确+MinHash+embedding 近重复审计":
  1. exact_hash        — sha256 over normalized text, catches verbatim dupes.
  2. minhash_near_dup   — MinHash/LSH Jaccard similarity over shingles, catches paraphrase-
                          light near-dupes cheaply (no model needed).
  3. embedding_near_dup — pluggable: needs an embedding model, deferred to the GPU server.
                          The interface exists here so the pipeline shape is fixed now.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from datasketch import MinHash, MinHashLSH


def normalize_text(text: str) -> str:
    """Whitespace/punctuation-insensitive normalization used before hashing, so that
    trivial formatting differences don't hide an actual duplicate."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 5) -> set[str]:
    norm = normalize_text(text)
    if len(norm) < k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def _minhash_for(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


@dataclass(frozen=True)
class NearDupMatch:
    query_id: str
    match_id: str
    jaccard_estimate: float


def find_exact_duplicates(records: dict[str, str]) -> dict[str, list[str]]:
    """records: {id: text}. Returns {hash: [ids]} for any hash with >1 id."""
    by_hash: dict[str, list[str]] = {}
    for rec_id, text in records.items():
        h = exact_hash(text)
        by_hash.setdefault(h, []).append(rec_id)
    return {h: ids for h, ids in by_hash.items() if len(ids) > 1}


def find_minhash_near_duplicates(
    records: dict[str, str], threshold: float = 0.85, num_perm: int = 128
) -> list[NearDupMatch]:
    """records: {id: text}. Returns pairwise near-dup matches above `threshold`.

    threshold=0.85 is a starting point, not a frozen constant — the spec requires this
    to run against real test-set items (test_eval_ids_absent_from_all_training_manifests
    depends on it), so tune conservatively (higher threshold = fewer false positives)
    once real data is loaded, and log whatever threshold was actually used per audit run.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    for rec_id, text in records.items():
        mh = _minhash_for(text, num_perm=num_perm)
        minhashes[rec_id] = mh
        lsh.insert(rec_id, mh)

    matches: list[NearDupMatch] = []
    seen_pairs: set[tuple[str, str]] = set()
    for rec_id, mh in minhashes.items():
        for other_id in lsh.query(mh):
            if other_id == rec_id:
                continue
            pair = tuple(sorted((rec_id, other_id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            jaccard = minhashes[pair[0]].jaccard(minhashes[pair[1]])
            matches.append(NearDupMatch(pair[0], pair[1], jaccard))
    return matches


class Embedder(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...


def find_embedding_near_duplicates(
    records: dict[str, str],
    embed_fn: Embedder,
    threshold: float = 0.92,
) -> list[NearDupMatch]:
    """Cosine-similarity near-dup pass. Requires a real `embed_fn` (deferred to the GPU
    server / an API embedding model) — this function is fully implemented and testable
    with a fake/mock embedder locally, only the *model* is deferred."""
    import numpy as np

    ids = list(records.keys())
    vecs = np.array(embed_fn([records[i] for i in ids]), dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vecs / norms
    sims = unit @ unit.T

    matches: list[NearDupMatch] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                matches.append(NearDupMatch(ids[i], ids[j], float(sims[i, j])))
    return matches


def cross_split_contamination(
    train_ids_texts: dict[str, str],
    eval_ids_texts: dict[str, str],
    threshold: float = 0.85,
) -> list[NearDupMatch]:
    """Checks eval-set items against the training pool. This is what
    test_eval_ids_absent_from_all_training_manifests should call in anger — any non-empty
    result here is a G1 blocker, not a warning."""
    combined = {**train_ids_texts, **{f"__eval__{k}": v for k, v in eval_ids_texts.items()}}
    raw_matches = find_minhash_near_duplicates(combined, threshold=threshold)
    cross: list[NearDupMatch] = []
    for m in raw_matches:
        q_is_eval = m.query_id.startswith("__eval__")
        m_is_eval = m.match_id.startswith("__eval__")
        if q_is_eval != m_is_eval:
            cross.append(
                NearDupMatch(
                    m.query_id.replace("__eval__", ""),
                    m.match_id.replace("__eval__", ""),
                    m.jaccard_estimate,
                )
            )
    return cross
