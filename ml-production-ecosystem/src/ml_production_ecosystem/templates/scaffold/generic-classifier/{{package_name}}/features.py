def build_features(record: dict[str, object]) -> dict[str, float]:
    return {key: float(value) for key, value in record.items() if isinstance(value, int | float)}
