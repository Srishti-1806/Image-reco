"""AES-256-GCM for PII stored next to vectors. Key comes from env/KMS, never from code."""
import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class PayloadCipher:
    def __init__(self, key: bytes):
        if len(key) != 32:
            raise ValueError("AES-256 key must be 32 bytes")
        self._aead = AESGCM(key)

    def encrypt(self, plaintext: str, aad: str = "") -> str:
        nonce = os.urandom(12)
        ct = self._aead.encrypt(nonce, plaintext.encode(), aad.encode())
        return base64.b64encode(nonce + ct).decode()

    def decrypt(self, token: str, aad: str = "") -> str:
        raw = base64.b64decode(token)
        return self._aead.decrypt(raw[:12], raw[12:], aad.encode()).decode()
