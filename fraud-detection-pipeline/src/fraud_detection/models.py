from typing import Protocol

from lightgbm import LGBMClassifier


class ModelFactory(Protocol):
    def create(self, scale_pos_weight: float | None = None) -> LGBMClassifier:
        ...


class LightGbmFactory:
    def __init__(self, scale_pos_weight: float | None = None) -> None:
        self._scale_pos_weight = scale_pos_weight

    def create(self, scale_pos_weight: float | None = None) -> LGBMClassifier:
        resolved = scale_pos_weight if scale_pos_weight is not None else self._scale_pos_weight
        extra = {} if resolved is None else {"scale_pos_weight": resolved}
        return LGBMClassifier(
            n_estimators=20,
            num_leaves=4,
            min_child_samples=1,
            random_state=42,
            verbose=-1,
            **extra,
        )
