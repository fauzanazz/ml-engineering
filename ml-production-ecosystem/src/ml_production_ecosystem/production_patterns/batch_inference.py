"""Production-pattern batch inference entrypoint.

This wrapper keeps the foundation implementation reusable while exposing the
workflow from 02-production-patterns as the production-pattern layer.
"""

from ml_production_ecosystem.recommendation.batch import main as foundation_batch_main


def main() -> None:
    foundation_batch_main()


if __name__ == "__main__":
    main()
