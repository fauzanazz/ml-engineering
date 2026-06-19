from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def best_person_box(boxes: list[BoundingBox]) -> BoundingBox | None:
    if not boxes:
        return None
    return max(boxes, key=lambda box: (box.confidence, box.area))


def crop_box(frame, box: BoundingBox):
    height, width = frame.shape[:2]
    x1 = max(0, min(width, box.x1))
    y1 = max(0, min(height, box.y1))
    x2 = max(0, min(width, box.x2))
    y2 = max(0, min(height, box.y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]
