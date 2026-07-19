"""License/provenance ledger for every data source touched.

G1 requires: "每个来源 license/revision/hash 完整" and unclear-license data must be
quarantined, never trained on. This module is the single place that enforces that.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

LicenseStatus = Literal["cleared", "unclear", "rejected"]


@dataclass
class LedgerEntry:
    source: str
    revision: str
    license_id: str
    license_status: LicenseStatus
    url: Optional[str]
    retrieved_at: str  # ISO8601
    raw_file_sha256: str
    notes: str = ""


class LicenseLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, LedgerEntry] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        for key, val in data.items():
            self.entries[key] = LedgerEntry(**val)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.entries.items()}, f, ensure_ascii=False, indent=2)

    def record(self, entry: LedgerEntry) -> None:
        key = f"{entry.source}@{entry.revision}"
        self.entries[key] = entry

    def is_trainable(self, source: str, revision: str) -> bool:
        key = f"{source}@{revision}"
        entry = self.entries.get(key)
        return entry is not None and entry.license_status == "cleared"

    def quarantined_sources(self) -> list[str]:
        return [k for k, v in self.entries.items() if v.license_status != "cleared"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
