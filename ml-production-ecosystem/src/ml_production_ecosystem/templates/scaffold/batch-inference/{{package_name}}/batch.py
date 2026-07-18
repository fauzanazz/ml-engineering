def predict_record(record: dict[str, object]) -> dict[str, object]:
    return {**record, "prediction": "pending"}

def predict_batch(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [predict_record(record) for record in records]
