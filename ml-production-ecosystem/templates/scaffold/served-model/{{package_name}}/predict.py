from __future__ import annotations


def predict(features: dict[str, float], threshold: float = 0.0) -> bool:
    """Dummy inference seam.

    Replace this function with real model loading/inference. The template keeps
    the contract stable while returning a deterministic boolean example.
    """
    return sum(features.values()) >= threshold
