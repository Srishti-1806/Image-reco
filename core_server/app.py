import logging
import math
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .crypto import PayloadCipher
from .db_manager import AuditLog
from .matching_engine import decide
from .vector_db import VectorStore

log = logging.getLogger("core")


class VerifyRequest(BaseModel):
    embedding: list[float] = Field(min_length=512, max_length=512)

    @field_validator("embedding")
    @classmethod
    def finite(cls, v):
        if not all(math.isfinite(x) for x in v):
            raise ValueError("embedding contains non-finite values")
        return v


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.store = VectorStore(settings.qdrant_url, settings.qdrant_api_key, settings.embedding_dim)
        app.state.store.ensure_collection()
        app.state.cipher = PayloadCipher(settings.payload_key)
        app.state.audit = AuditLog(settings.audit_db_path)
        app.state.hits = defaultdict(deque)
        yield

    app = FastAPI(title="Secure Central Matching Engine", lifespan=lifespan)

    def authenticate(
        request: Request,
        x_client_cn: str | None = Header(None),
        x_client_verify: str | None = Header(None),
    ) -> tuple[str, str]:
        # Headers are set by the nginx gateway after a successful mTLS handshake.
        if x_client_verify != "SUCCESS" or not x_client_cn:
            raise HTTPException(401, "client certificate required")
        zone = settings.allowed_devices.get(x_client_cn)
        if zone is None:
            raise HTTPException(403, "device not enrolled")
        now = time.monotonic()
        q = request.app.state.hits[x_client_cn]
        while q and now - q[0] > 1.0:
            q.popleft()
        if len(q) >= settings.rate_limit_per_sec:
            raise HTTPException(429, "rate limit exceeded")
        q.append(now)
        return x_client_cn, zone

    async def send_alert(device_id: str, zone: str, reason: str, score):
        if not settings.alert_webhook_url:
            return
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                await c.post(settings.alert_webhook_url, json={
                    "device_id": device_id, "zone": zone, "reason": reason, "score": score, "ts": time.time()})
        except Exception:  # alerting must never break the decision path
            log.exception("alert webhook failed")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/api/v1/verify-face")
    def verify_face(payload: VerifyRequest, request: Request, bg: BackgroundTasks,
                    device: tuple[str, str] = Depends(authenticate)):
        device_id, zone = device
        st = request.app.state
        try:
            match = st.store.best_match(payload.embedding)
        except ValueError:
            raise HTTPException(400, "invalid embedding")

        score = match.score if match else None
        d = decide(score, settings.match_threshold, zone)
        person_id = match.person_id if (match and d.status == "ALLOWED") else None
        st.audit.record(device_id, zone, d.status, d.reason, person_id, score)

        if d.action == "ALERT":
            bg.add_task(send_alert, device_id, zone, d.reason, score)

        resp = {"status": d.status, "action": d.action, "reason": d.reason}
        if person_id:
            resp["person_id"] = person_id
            resp["name"] = st.cipher.decrypt(match.encrypted_name, aad=person_id)
            resp["confidence"] = round(score, 4)
        return resp

    return app
