"""Block 4: face detection (SCRFD via InsightFace) with TensorRT -> CUDA -> CPU fallback.
InsightFace's FaceAnalysis runs detection + alignment + recognition in one call,
so `detect()` returns faces that already carry an embedding for feature_extractor."""
import logging

log = logging.getLogger("detector")


class FaceDetector:
    def __init__(self, model_name: str, providers: tuple, det_size: tuple):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=model_name,
            providers=list(providers),
            allowed_modules=["detection", "recognition"],
            addons=["liveness"],
            liveness_mode="normal",
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, frame):
        return self.app.get(frame)

    @staticmethod
    def best_face(faces, min_score: float, min_px: int):
        """Largest face that passes basic quality gates, else None."""
        ok = []
        for f in faces:
            liveness_result = getattr(f, "liveness", None)
            if (
                not isinstance(liveness_result, dict)
                or liveness_result.get("status") != "ok"
                or liveness_result.get("is_live") is not True
            ):
                continue
            x1, y1, x2, y2 = f.bbox
            if f.det_score >= min_score and min(x2 - x1, y2 - y1) >= min_px:
                ok.append((float((x2 - x1) * (y2 - y1)), f))
        return max(ok, key=lambda t: t[0])[1] if ok else None
