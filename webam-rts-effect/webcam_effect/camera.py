from dataclasses import dataclass


def parse_camera_source(source: str):
    if source.isdigit():
        return int(source)
    return source


@dataclass
class CameraSource:
    source: str = "0"
    width: int = 1280
    height: int = 720

    def open(self):
        import cv2

        capture = cv2.VideoCapture(parse_camera_source(self.source))
        if not capture.isOpened():
            raise RuntimeError(f"could not open camera source: {self.source}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return capture
