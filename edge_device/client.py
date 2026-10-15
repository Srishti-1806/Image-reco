import logging

import requests

log = logging.getLogger("client")


class CoreClient:
    """mTLS client. The server identifies this device by its certificate CN."""

    def __init__(self, base_url: str, ca: str, cert: str, key: str, timeout: float = 2.0):
        self.url = base_url.rstrip("/") + "/api/v1/verify-face"
        self.s = requests.Session()
        self.s.verify = ca
        self.s.cert = (cert, key)
        self.timeout = timeout

    def verify(self, embedding: list[float]) -> dict | None:
        try:
            r = self.s.post(self.url, json={"embedding": embedding}, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.error("core unreachable / rejected: %s", e)
            return None      # fail closed: no response means no access
