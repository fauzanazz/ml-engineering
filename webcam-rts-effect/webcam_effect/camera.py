from dataclasses import dataclass

SUPPORTED_RESOLUTIONS = {
    "640x480": (640, 480),
    "1280x720": (1280, 720),
}


def parse_camera_source(source: str):
    if source.isdigit():
        return int(source)
    return source

def parse_resolution(value: str) -> tuple[int, int]:
    if value not in SUPPORTED_RESOLUTIONS:
        supported = ", ".join(SUPPORTED_RESOLUTIONS)
        raise ValueError(f"unsupported resolution: {value}. Use one of: {supported}")
    return SUPPORTED_RESOLUTIONS[value]


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
