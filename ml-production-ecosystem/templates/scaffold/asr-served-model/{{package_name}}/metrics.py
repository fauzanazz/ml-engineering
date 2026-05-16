def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    mismatches = sum(left != right for left, right in zip(reference_words, hypothesis_words))
    edits = mismatches + abs(len(reference_words) - len(hypothesis_words))
    return edits / len(reference_words)
