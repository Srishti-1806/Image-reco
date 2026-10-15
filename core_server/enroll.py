"""Enrollment CLI (run on the core server / admin machine).

  python -m core_server.enroll add --id emp001 --name "Asha Rao" --image a.jpg b.jpg
  python -m core_server.enroll delete --id emp001

`add` needs insightface + opencv installed (see requirements-edge.txt).
Multiple images are averaged into one template for robustness.
"""
import argparse

import numpy as np

from .config import Settings
from .crypto import PayloadCipher
from .vector_db import VectorStore, normalize


def embed_images(paths: list[str]):
    import cv2
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    vecs = []
    for p in paths:
        img = cv2.imread(p)
        faces = app.get(img) if img is not None else []
        if len(faces) != 1:
            raise SystemExit(f"{p}: expected exactly 1 face, found {len(faces)}")
        vecs.append(faces[0].normed_embedding)
    return normalize(np.mean(vecs, axis=0))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--id", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--image", nargs="+", required=True)
    d = sub.add_parser("delete")
    d.add_argument("--id", required=True)
    args = ap.parse_args()

    s = Settings.from_env()
    store = VectorStore(s.qdrant_url, s.qdrant_api_key, s.embedding_dim)
    store.ensure_collection()
    if args.cmd == "add":
        enc = PayloadCipher(s.payload_key).encrypt(args.name, aad=args.id)
        store.upsert(args.id, embed_images(args.image), enc)
        print(f"enrolled {args.id}")
    else:
        store.delete(args.id)
        print(f"deleted {args.id}")


if __name__ == "__main__":
    main()
