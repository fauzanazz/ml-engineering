def recommend(user_id: str, candidates: list[str], limit: int = 10) -> list[str]:
    del user_id
    return candidates[:limit]
