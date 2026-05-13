from __future__ import annotations

import hashlib


def assign_split(
    template_id: str,
    account_number: str,
    amount: str,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> str:
    key = f"{template_id}|{account_number}|{amount}"
    bucket = _stable_bucket(key)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def _stable_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
