from dataclasses import dataclass

from webcam_effect.state import PosePrediction
from webcam_effect.tracking import BoundingBox, crop_box


PERSON_CLASS_ID = 0


@dataclass
class YoloPersonDetector:
    model_path: str
    device: str = "mps"
    data: str = "coco8.yaml"
    confidence: float = 0.35

    def __post_init__(self):
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def detect(self, frame) -> list[BoundingBox]:
        results = self.model.predict(frame, device=self.device, conf=self.confidence, verbose=False)
        if not results:
            return []

        result = results[0]
        if result.boxes is None:
            return []

        detected_boxes = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id != PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detected_boxes.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf[0].item()),
                )
            )
        return detected_boxes


@dataclass
class YoloPersonSegmenter:
    model_path: str
    device: str = "mps"
    confidence: float = 0.35

    def __post_init__(self):
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def crop(self, frame, segmentation_input: str = "masked-crop"):
        result = self._best_person_result(frame)
        if result is None:
            return None

        box, mask = result
        frame_crop = crop_box(frame, box)
        if frame_crop is None or segmentation_input == "crop":
            return frame_crop
        if segmentation_input != "masked-crop":
            raise ValueError(f"unknown segmentation input: {segmentation_input}")

        return apply_mask_to_crop(frame, frame_crop, box, mask)

    def _best_person_result(self, frame):
        results = self.model.predict(frame, device=self.device, conf=self.confidence, verbose=False)
        if not results:
            return None

        result = results[0]
        if result.boxes is None or result.masks is None:
            return None

        best_index = None
        best_confidence = -1.0
        for index, box in enumerate(result.boxes):
            if int(box.cls[0].item()) != PERSON_CLASS_ID:
                continue
            confidence = float(box.conf[0].item())
            if confidence > best_confidence:
                best_index = index
                best_confidence = confidence
        if best_index is None:
            return None

        x1, y1, x2, y2 = [int(value) for value in result.boxes[best_index].xyxy[0].tolist()]
        box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=best_confidence)
        mask = result.masks.data[best_index].detach().cpu().numpy()
        return box, mask


def apply_mask_to_crop(frame, frame_crop, box: BoundingBox, mask):
    import cv2

    frame_height, frame_width = frame.shape[:2]
    resized_mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, min(frame_width, box.x1))
    y1 = max(0, min(frame_height, box.y1))
    x2 = max(0, min(frame_width, box.x2))
    y2 = max(0, min(frame_height, box.y2))
    if x2 <= x1 or y2 <= y1:
        return frame_crop

    alpha = (resized_mask[y1:y2, x1:x2] >= 0.5).astype(frame_crop.dtype)[:, :, None]
    return frame_crop * alpha


@dataclass
class YoloFrameClassifier:
    model_path: str
    device: str = "mps"
    data: str = "coco8.yaml"

    def __post_init__(self):
        from ultralytics import YOLO

        self.model = YOLO(self.model_path)

    def predict(self, frame) -> PosePrediction:
        results = self.model.predict(frame, device=self.device, verbose=False)
        if not results or results[0].probs is None:
            return PosePrediction(label="none", confidence=0.0)

        result = results[0]
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf.item())
        label = result.names.get(class_id, str(class_id))
        return PosePrediction(label=label, confidence=confidence)
