import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeConfig:
    camera_url: str = os.environ.get("CAMERA_URL", "0")
    core_url: str = os.environ.get("CORE_URL", "https://core.local:8443")
    tls_ca: str = os.environ.get("TLS_CA", "network_security/certs/ca.crt")
    tls_cert: str = os.environ.get("TLS_CERT", "network_security/certs/jetson-01.crt")
    tls_key: str = os.environ.get("TLS_KEY", "network_security/certs/jetson-01.key")
    frame_width: int = 960
    det_size: tuple = (640, 640)
    model_name: str = os.environ.get("FACE_MODEL", "buffalo_l")
    providers: tuple = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    min_det_score: float = 0.6
    min_face_px: int = 80
    request_interval_s: float = 0.3
    grant_cooldown_s: float = 3.0
