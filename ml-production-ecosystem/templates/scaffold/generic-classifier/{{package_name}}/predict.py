def predict(features: dict[str, float]) -> str:
    score = sum(features.values())
    return "positive" if score >= 0 else "negative"
