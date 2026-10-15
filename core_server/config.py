import base64
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    payload_key: bytes = b""
    match_threshold: float = 0.45
    allowed_devices: dict[str, str] = field(default_factory=dict)  # CN -> zone
    alert_webhook_url: str | None = None
    audit_db_path: str = "audit.sqlite3"
    rate_limit_per_sec: int = 10
    embedding_dim: int = 512

    @classmethod
    def from_env(cls) -> "Settings":
        key_b64 = os.environ.get("PAYLOAD_KEY_B64", "")
        key = base64.b64decode(key_b64) if key_b64 else b""
        if len(key) != 32:
            raise RuntimeError("PAYLOAD_KEY_B64 must decode to exactly 32 bytes (AES-256).")
        devices = {}
        for item in os.environ.get("ALLOWED_DEVICES", "").split(","):
            if item.strip():
                cn, _, zone = item.strip().partition(":")
                devices[cn] = zone or "public"
        return cls(
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            payload_key=key,
            match_threshold=float(os.environ.get("MATCH_THRESHOLD", "0.45")),
            allowed_devices=devices,
            alert_webhook_url=os.environ.get("ALERT_WEBHOOK_URL") or None,
            audit_db_path=os.environ.get("AUDIT_DB_PATH", "audit.sqlite3"),
            rate_limit_per_sec=int(os.environ.get("RATE_LIMIT_PER_SEC", "10")),
        )
