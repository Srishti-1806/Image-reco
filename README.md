# secure-face-recog

Edge/core face recognition for access control.

- **edge_device/** (Jetson): RTSP ingest -> CLAHE/denoise -> SCRFD detect -> ArcFace 512-D embedding -> mTLS POST.
- **core_server/** (GPU server): FastAPI + Qdrant cosine match, AES-256-GCM encrypted identity payloads, SQLite audit log, alert webhook.
- **deploy/**: nginx gateway (TLS 1.3 only, mandatory client certs) + Docker for the core.

## Quick start

```bash
cp .env.example .env            # fill PAYLOAD_KEY_B64, QDRANT_API_KEY, ...
./network_security/gen_certs.sh init core.local
./network_security/gen_certs.sh device jetson-01
docker compose up -d --build

# enroll (multiple photos per person are averaged)
python -m core_server.enroll add --id emp001 --name "Asha Rao" --image a1.jpg a2.jpg

# on the Jetson
pip install -r requirements-edge.txt
python -m edge_device.main
```

## Tests
```bash
pip install -r requirements-dev.txt && pytest
```

## Local Streamlit lab

The dashboard has two pages: **Add Person** asks for a name, scans a short camera
frame burst after a 3-2-1 countdown, and generates the person's ID; **Verify Access**
scans a burst after the same countdown and checks it through the core API. InsightFace's liveness
PAD addon screens the face before its embedding is produced: failed, missing, or
inconclusive results are denied. Allowed attempts play a pyttsx3 "ting"; denied
attempts play a beep. It runs locally with an in-memory Qdrant store and temporary
encryption key. PAD is probabilistic and this lab setup is not suitable to control
real access.

```bash
pip install -r requirements-ui.txt
streamlit run streamlit_app.py
```

Use the machine's camera attached to the Streamlit server. `CAMERA_URL` selects
its device index or stream and `FACE_MODEL` selects InsightFace's model; inference
runs on CPU. On first launch, InsightFace downloads its separately distributed
liveness model addon and verifies its SHA-256 digest. Production still requires
validated PAD performance on representative attack samples, plus mTLS deployment.

## Things you MUST do before production
1. **Threshold**: `MATCH_THRESHOLD` default 0.45 is a starting point for buffalo_l cosine similarity
   (0.85 would reject almost every genuine user). Tune on your own data for a target FAR/FRR.
2. **Liveness**: validate PAD false-accept/false-reject rates against representative printed-photo, screen-replay, mask, lighting, and camera attack samples; use IR/depth hardware where the threat model requires it.
3. **Keys**: move `PAYLOAD_KEY_B64` into a KMS/Vault; plan rotation. Encrypt disks (embeddings are biometric data).
4. **Legal**: consent, retention, deletion (`enroll delete`), DPDP Act / GDPR / local rules.
5. Fail-closed is built in: if the core is unreachable, the edge grants nothing.
