from __future__ import annotations

from collections import Counter


def compute_wer(reference: str, hypothesis: str) -> dict:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    word_errors = _levenshtein_distance(reference_words, hypothesis_words)
    reference_word_count = len(reference_words)
    wer = word_errors / reference_word_count if reference_word_count else float(word_errors > 0)
    return {
        "word_errors": word_errors,
        "reference_words": reference_word_count,
        "wer": wer,
    }


def compute_entity_error_rate(row: dict, hypothesis: str) -> dict:
    entities = row.get("entities", [])
    errors_by_type = Counter()
    normalized_hypothesis = hypothesis.casefold()

    for entity in entities:
        entity_text = entity["text"].casefold()
        if entity_text not in normalized_hypothesis:
            errors_by_type[entity["type"]] += 1

    entity_errors = sum(errors_by_type.values())
    entity_count = len(entities)
    return {
        "entity_errors": entity_errors,
        "entities": entity_count,
        "entity_error_rate": entity_errors / entity_count if entity_count else 0.0,
        "errors_by_type": dict(errors_by_type),
    }


def evaluate_predictions(manifest_rows: list[dict], prediction_rows: list[dict]) -> dict:
    predictions_by_id = {row["utterance_id"]: row["hypothesis"] for row in prediction_rows}
    word_errors = 0
    reference_words = 0
    entity_errors = 0
    entity_count = 0
    errors_by_type = Counter()

    for row in manifest_rows:
        hypothesis = predictions_by_id[row["utterance_id"]]
        wer_result = compute_wer(row["text"], hypothesis)
        entity_result = compute_entity_error_rate(row, hypothesis)
        word_errors += wer_result["word_errors"]
        reference_words += wer_result["reference_words"]
        entity_errors += entity_result["entity_errors"]
        entity_count += entity_result["entities"]
        errors_by_type.update(entity_result["errors_by_type"])

    return {
        "rows": len(manifest_rows),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "wer": word_errors / reference_words if reference_words else 0.0,
        "entity_errors": entity_errors,
        "entities": entity_count,
        "entity_error_rate": entity_errors / entity_count if entity_count else 0.0,
        "errors_by_type": dict(errors_by_type),
    }



def _levenshtein_distance(reference_words: list[str], hypothesis_words: list[str]) -> int:
    previous_row = list(range(len(hypothesis_words) + 1))
    for reference_index, reference_word in enumerate(reference_words, start=1):
        current_row = [reference_index]
        for hypothesis_index, hypothesis_word in enumerate(hypothesis_words, start=1):
            substitution_cost = 0 if reference_word == hypothesis_word else 1
            current_row.append(
                min(
                    previous_row[hypothesis_index] + 1,
                    current_row[hypothesis_index - 1] + 1,
                    previous_row[hypothesis_index - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]
