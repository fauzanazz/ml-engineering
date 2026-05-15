PIPELINE_STAGES = (
    "ingestion",
    "validation",
    "training",
    "approval",
    "batch_inference",
    "monitoring",
    "rollback",
)


def planned_stages() -> tuple[str, ...]:
    return PIPELINE_STAGES


def main() -> None:
    print(" -> ".join(planned_stages()))


if __name__ == "__main__":
    main()
