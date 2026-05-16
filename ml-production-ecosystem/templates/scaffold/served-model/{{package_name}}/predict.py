def predict(features: dict[str, float]) -> float:
    return sum(features.values())
