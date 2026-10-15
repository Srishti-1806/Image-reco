"""Two-page local lab UI for enrolling and verifying faces."""
import atexit
import os
import time
import uuid
from collections.abc import Sequence

import numpy as np
import streamlit as st
from fastapi.testclient import TestClient

from core_server.app import create_app
from core_server.config import Settings
from edge_device import feature_extractor, liveness
from edge_device.config import EdgeConfig
from edge_device.face_detector import FaceDetector

DEVICE_ID = "streamlit-lab"
AUTH_HEADERS = {"X-Client-CN": DEVICE_ID, "X-Client-Verify": "SUCCESS"}


@st.cache_resource
def get_demo_client() -> TestClient:
    settings = Settings(
        qdrant_url=":memory:",
        payload_key=os.urandom(32),
        allowed_devices={DEVICE_ID: "secure"},
        audit_db_path=":memory:",
        rate_limit_per_sec=1000,
    )
    client = TestClient(create_app(settings))
    client.__enter__()
    atexit.register(client.__exit__, None, None, None)
    return client


@st.cache_resource
def get_face_detector(model_name: str) -> FaceDetector:
    return FaceDetector(model_name, ("CPUExecutionProvider",), (640, 640))


def countdown_capture(camera_url: str, display) -> list[bytes]:
    import cv2

    source = int(camera_url) if camera_url.isdecimal() else camera_url
    capture = (
        cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if os.name == "nt" and isinstance(source, int)
        else cv2.VideoCapture(source)
    )
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source {camera_url!r}.")
        countdown_started = time.monotonic()
        while True:
            elapsed = time.monotonic() - countdown_started
            seconds_left = 3 - int(elapsed)
            if seconds_left <= 0:
                break
            ok, preview = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not read a frame from camera source {camera_url!r}."
                )
            cv2.putText(
                preview,
                str(seconds_left),
                (max(24, preview.shape[1] // 2 - 40), max(64, preview.shape[0] // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 255, 255),
                6,
                cv2.LINE_AA,
            )
            display.image(
                preview,
                channels="BGR",
                caption=f"Hold still — capturing in {seconds_left}",
            )
            time.sleep(0.05)

        frames = []
        for _ in range(8):
            ok, frame = capture.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                frames.append(encoded.tobytes())
            time.sleep(0.08)
        if not frames:
            raise RuntimeError(f"Could not read a frame from camera source {camera_url!r}.")
        display.markdown("### Checking captured frames…")
        return frames
    finally:
        capture.release()


def extract_camera_embedding(
    image_bytes: bytes | Sequence[bytes], model_name: str
) -> list[float]:
    import cv2

    checker = liveness.build()
    detector = get_face_detector(model_name)
    image_frames = [image_bytes] if isinstance(image_bytes, bytes) else list(image_bytes)
    if not image_frames:
        raise ValueError("Camera se koi frame capture nahi hua. Dobara try karein.")

    decoded_any = False
    detected_any = False
    usable_any = False
    pad_rejections = []
    live_faces = []

    for image in image_frames:
        frame = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        decoded_any = True

        detected_faces = detector.detect(frame)
        if not detected_faces:
            continue
        detected_any = True

        for face in detected_faces:
            bbox = face.bbox
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            if float(face.det_score) < 0.45 or min(width, height) < 48:
                continue
            usable_any = True

            result = getattr(face, "liveness", None)
            if not checker.is_live(frame, face):
                if isinstance(result, dict):
                    score = result.get("live_score")
                    reason = result.get("reason")
                    if isinstance(score, (int, float)):
                        pad_rejections.append(f"live score {float(score):.2f}")
                    elif reason:
                        pad_rejections.append(str(reason))
                    else:
                        pad_rejections.append(str(result.get("status", "unknown")))
                else:
                    pad_rejections.append("PAD verdict missing")
                continue

            if getattr(face, "normed_embedding", None) is not None:
                live_faces.append((width * height, frame, face))

    if live_faces:
        _, _, selected_face = max(live_faces, key=lambda item: item[0])
        return feature_extractor.extract(selected_face)

    if not decoded_any:
        raise ValueError("Could not decode the camera image.")
    if not detected_any:
        raise ValueError(
            "Captured frames mein face detect nahi hua. Camera ke center mein "
            "seedhe dekhein, achhi light rakhein aur thoda paas aayen."
        )
    if not usable_any:
        raise ValueError(
            "Face bahut chhota ya unclear tha. Camera ke paas/center mein aayen "
            "aur dobara capture karein."
        )

    detail = max(pad_rejections, key=len) if pad_rejections else "PAD verdict missing"
    raise ValueError(
        f"Captured frames mein koi face PAD pass nahi hua ({detail}). "
        "Safety ke liye enrollment/access rok diya. Live person seedhe camera "
        "dekhein, achhi roshni mein rahein aur phir try karein."
    )


def enroll_person(client: TestClient, name: str, embedding: list[float]) -> str:
    person_id = f"person-{uuid.uuid4().hex[:12]}"
    encrypted_name = client.app.state.cipher.encrypt(name.strip(), aad=person_id)
    client.app.state.store.upsert(person_id, embedding, encrypted_name)
    return person_id


def verify_embedding(client: TestClient, embedding: list[float]) -> tuple[int, dict]:
    response = client.post(
        "/api/v1/verify-face",
        json={"embedding": embedding},
        headers=AUTH_HEADERS,
    )
    return response.status_code, response.json()


def play_feedback(allowed: bool) -> None:
    if allowed:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say("ting")
        engine.runAndWait()
        return

    import winsound

    winsound.Beep(700, 700)


def main() -> None:
    st.set_page_config(page_title="Secure Face Recognition", page_icon="🔐")
    st.title("Secure Face Recognition")
    st.caption("Local camera access demo")
    st.warning(
        "Presentation-attack detection (PAD) is enabled. It screens captured faces "
        "for likely photo/screen spoofs; uncertain or failed checks are denied. "
        "PAD is probabilistic, not a guarantee, so this demo must not control a real door."
    )

    client = get_demo_client()
    health = client.get("/healthz")
    if not health.is_success or not health.json().get("ok"):
        st.error("Core API health check failed.")
        return

    config = EdgeConfig()
    page = st.sidebar.radio("Page", ("Add Person", "Verify Access"))
    st.sidebar.caption("Camera: " + config.camera_url)

    if page == "Add Person":
        st.subheader("Add a new person")
        st.write(
            "Enter name, then camera shows a 3-second countdown and scans a short "
            "burst of frames for a face that passes PAD."
        )
        name = st.text_input("Person's name", key="new_person_name")
        if st.button(
            "Start face enrollment",
            type="primary",
            disabled=not name.strip(),
            use_container_width=True,
        ):
            countdown = st.empty()
            try:
                image = countdown_capture(config.camera_url, countdown)
                embedding = extract_camera_embedding(
                    image, config.model_name
                )
                person_id = enroll_person(client, name, embedding)
            except (ImportError, RuntimeError, ValueError) as exc:
                st.error(f"Enrollment failed: {exc}")
            else:
                st.success(f"{name.strip()} enrolled successfully.")
                st.code(f"Generated ID: {person_id}")

    else:
        st.subheader("Verify access")
        st.write(
            "Camera shows a 3-second countdown, then scans a short frame burst. "
            "A face must pass PAD before matching."
        )
        if st.button(
            "Start verification",
            type="primary",
            use_container_width=True,
        ):
            countdown = st.empty()
            try:
                image = countdown_capture(config.camera_url, countdown)
                embedding = extract_camera_embedding(
                    image, config.model_name
                )
                status_code, result = verify_embedding(client, embedding)
            except (ImportError, RuntimeError, ValueError) as exc:
                st.error(f"Verification failed: {exc}")
            else:
                st.image(image[-1], caption="Captured camera frame")
                if status_code == 200 and result.get("status") == "ALLOWED":
                    st.success(f"ACCESS ALLOWED — Welcome, {result.get('name', 'user')}")
                    st.metric("Match confidence", result.get("confidence", 0))
                    try:
                        play_feedback(allowed=True)
                    except Exception as exc:
                        st.warning(f"Access allowed, but ting audio could not play: {exc}")
                else:
                    st.error("ACCESS DENIED")
                    try:
                        play_feedback(allowed=False)
                    except Exception as exc:
                        st.warning(f"Access denied, but beep audio could not play: {exc}")

    st.sidebar.metric("Verification attempts", client.app.state.audit.count())


if __name__ == "__main__":
    main()
