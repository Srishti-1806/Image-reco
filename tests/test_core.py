import base64
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core_server.app import create_app
from core_server.config import Settings
from core_server.crypto import PayloadCipher
from core_server.matching_engine import decide

KEY = os.urandom(32)
AUTH = {"X-Client-CN": "jetson-01", "X-Client-Verify": "SUCCESS"}


def rand_vec(seed):
    return np.random.default_rng(seed).normal(size=512).astype(np.float32)


@pytest.fixture
def client(tmp_path):
    s = Settings(qdrant_url=":memory:", payload_key=KEY, match_threshold=0.45,
                 allowed_devices={"jetson-01": "secure", "jetson-02": "public"},
                 audit_db_path=str(tmp_path / "a.db"), rate_limit_per_sec=1000)
    with TestClient(create_app(s)) as c:
        enc = c.app.state.cipher.encrypt("Asha Rao", aad="emp001")
        c.app.state.store.upsert("emp001", rand_vec(1), enc)
        yield c


def test_crypto_roundtrip_and_aad():
    c = PayloadCipher(KEY)
    tok = c.encrypt("Asha", aad="x")
    assert c.decrypt(tok, aad="x") == "Asha"
    with pytest.raises(Exception):
        c.decrypt(tok, aad="other")


def test_decide():
    assert decide(0.9, 0.45, "secure").action == "ACCESS_GRANTED"
    assert decide(0.1, 0.45, "secure").action == "ALERT"
    assert decide(0.1, 0.45, "public").action == "NONE"
    assert decide(None, 0.45, "secure").status == "DENIED"


def test_match_allowed(client):
    noisy = rand_vec(1) + 0.05 * rand_vec(99)
    r = client.post("/api/v1/verify-face", json={"embedding": noisy.tolist()}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ALLOWED" and body["name"] == "Asha Rao"


def test_unknown_denied_and_alert(client):
    r = client.post("/api/v1/verify-face", json={"embedding": rand_vec(7).tolist()}, headers=AUTH)
    assert r.json()["status"] == "DENIED" and r.json()["action"] == "ALERT"


def test_auth_required(client):
    body = {"embedding": rand_vec(1).tolist()}
    assert client.post("/api/v1/verify-face", json=body).status_code == 401
    bad = {"X-Client-CN": "evil", "X-Client-Verify": "SUCCESS"}
    assert client.post("/api/v1/verify-face", json=body, headers=bad).status_code == 403


def test_bad_embedding(client):
    r = client.post("/api/v1/verify-face", json={"embedding": [0.1] * 10}, headers=AUTH)
    assert r.status_code == 422
