def build_examples(seed_prompts: list[str]) -> list[dict[str, str]]:
    return [
        {
            "prompt": prompt,
            "answer": f"synthetic response for: {prompt}",
        }
        for prompt in seed_prompts
    ]
