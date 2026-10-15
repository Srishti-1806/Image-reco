import logging
import signal
import time

from . import dispatcher, feature_extractor, liveness
from .client import CoreClient
from .config import EdgeConfig
from .face_detector import FaceDetector
from .video_stream import VideoStream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("edge")


def main():
    cfg = EdgeConfig()
    live = liveness.build()
    detector = FaceDetector(cfg.model_name, cfg.providers, cfg.det_size)
    client = CoreClient(cfg.core_url, cfg.tls_ca, cfg.tls_cert, cfg.tls_key)
    stream = VideoStream(cfg.camera_url, cfg.frame_width).start()

    running = True
    def _stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_sent = 0.0
    while running:
        frame = stream.read()
        if frame is None or time.monotonic() - last_sent < cfg.request_interval_s:
            continue

        t0 = time.perf_counter()
        face = FaceDetector.best_face(detector.detect(frame), cfg.min_det_score, cfg.min_face_px)
        if face is None or not live.is_live(frame, face):
            continue

        emb = feature_extractor.extract(face)
        infer_ms = (time.perf_counter() - t0) * 1000
        resp = client.verify(emb)
        last_sent = time.monotonic()
        log.info("edge=%.0fms total=%.0fms -> %s", infer_ms, (time.perf_counter() - t0) * 1000,
                 resp and resp["status"])
        dispatcher.handle(resp)
        if resp and resp["action"] == "ACCESS_GRANTED":
            time.sleep(cfg.grant_cooldown_s)

    stream.stop()


if __name__ == "__main__":
    main()
