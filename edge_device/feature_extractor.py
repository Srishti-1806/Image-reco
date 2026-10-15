"""Block 5: turn an aligned face into a 512-D L2-normalised embedding (ArcFace)."""
import numpy as np


def extract(face) -> list[float]:
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    if emb.shape != (512,) or not np.isfinite(emb).all():
        raise ValueError("bad embedding from model")
    return emb.tolist()
