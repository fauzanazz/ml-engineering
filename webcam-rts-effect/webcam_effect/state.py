from dataclasses import dataclass


KICAU_LABEL = "kicau"
NONE_LABEL = "none"


@dataclass(frozen=True)
class PosePrediction:
    label: str
    confidence: float


@dataclass
class PoseStateMachine:
    activate_threshold: float = 0.7
    deactivate_threshold: float = 0.4
    active: bool = False

    def update(self, predictions: list[PosePrediction]) -> bool:
        score = smoothed_kicau_score(predictions)

        if self.active:
            self.active = score >= self.deactivate_threshold
        else:
            self.active = score >= self.activate_threshold

        return self.active


def smoothed_kicau_score(predictions: list[PosePrediction]) -> float:
    if not predictions:
        return 0.0

    kicau_predictions = [prediction for prediction in predictions if prediction.label == KICAU_LABEL]
    if len(kicau_predictions) <= len(predictions) // 2:
        return 0.0

    scores = [prediction.confidence for prediction in kicau_predictions]
    return sum(scores) / len(scores)
