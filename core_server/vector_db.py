"""Block 6: Qdrant wrapper. Embeddings are L2-normalised so COSINE == dot product."""
import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION = "secure_identities"


@dataclass
class Match:
    person_id: str
    score: float
    encrypted_name: str


def normalize(vec) -> list[float]:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError("invalid embedding")
    return (v / n).tolist()


def point_id(person_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"face-recog:{person_id}"))


class VectorStore:
    def __init__(self, url: str, api_key: str | None, dim: int = 512):
        self.client = QdrantClient(":memory:") if url == ":memory:" else QdrantClient(url=url, api_key=api_key)
        self.dim = dim

    def ensure_collection(self):
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                COLLECTION, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )

    def upsert(self, person_id: str, embedding, encrypted_name: str):
        self.client.upsert(
            COLLECTION,
            [PointStruct(id=point_id(person_id), vector=normalize(embedding),
                         payload={"person_id": person_id, "encrypted_name": encrypted_name})],
        )

    def delete(self, person_id: str):
        self.client.delete(COLLECTION, points_selector=[point_id(person_id)])

    def best_match(self, embedding) -> Match | None:
        res = self.client.query_points(COLLECTION, query=normalize(embedding), limit=1).points
        if not res:
            return None
        p = res[0]
        return Match(p.payload["person_id"], float(p.score), p.payload["encrypted_name"])
