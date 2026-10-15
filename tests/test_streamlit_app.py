from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    countdown_capture,
    enroll_person,
    extract_camera_embedding,
    play_feedback,
)
from edge_device.face_detector import FaceDetector
from edge_device.liveness import build


@pytest.fixture
def detector_stubs(monkeypatch):
    def install(faces):
        monkeypatch.setattr(
            "streamlit_app.get_face_detector",
            lambda model_name: SimpleNamespace(
                detect=lambda frame: faces(frame) if callable(faces) else faces
            ),
        )

    monkeypatch.setattr(
        "streamlit_app.liveness.build",
        lambda: SimpleNamespace(
            is_live=lambda frame, face: (
                isinstance(getattr(face, "liveness", None), dict)
                and face.liveness.get("status") == "ok"
                and face.liveness.get("is_live") is True
            )
        ),
    )
    monkeypatch.setattr(
        "streamlit_app.feature_extractor.extract",
        lambda face: [0.5] * 512,
    )
    return install


def test_dashboard_has_two_pages_and_only_asks_for_name_on_enrollment():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert [option for option in app.radio[0].options] == ["Add Person", "Verify Access"]
    assert [widget.label for widget in app.text_input] == ["Person's name"]
    app.text_input[0].set_value("Test Person").run(timeout=30)
    assert any(button.label == "Start face enrollment" and not button.disabled for button in app.button)

    app.radio[0].set_value("Verify Access").run(timeout=30)
    assert not app.exception
    assert any(button.label == "Start verification" and not button.disabled for button in app.button)


def test_countdown_captures_frame_burst_and_releases_camera(monkeypatch):
    class FakeCamera:
        released = False

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class FakeDisplay:
        def __init__(self):
            self.frames = []
            self.captions = []
            self.messages = []

        def image(self, frame, channels, caption):
            self.frames.append(frame.copy())
            self.captions.append(caption)

        def markdown(self, message):
            self.messages.append(message)

    camera = FakeCamera()
    display = FakeDisplay()
    clock = iter(index * 0.25 for index in range(20))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: camera)
    monkeypatch.setattr("streamlit_app.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("streamlit_app.time.sleep", lambda _: None)

    images = countdown_capture("0", display)

    assert len(images) == 8
    assert all(image.startswith(b"\xff\xd8") for image in images)
    assert "Hold still — capturing in 3" in display.captions
    assert "Hold still — capturing in 2" in display.captions
    assert "Hold still — capturing in 1" in display.captions
    assert display.messages == ["### Checking captured frames…"]
    assert len(display.frames) > 3
    assert camera.released


def test_enrollment_generates_an_id_and_stores_the_encrypted_name():
    records = []
    client = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                cipher=SimpleNamespace(
                    encrypt=lambda name, aad: f"encrypted:{name}:{aad}"
                ),
                store=SimpleNamespace(
                    upsert=lambda person_id, embedding, encrypted_name: records.append(
                        (person_id, embedding, encrypted_name)
                    )
                ),
            )
        )
    )
    embedding = [0.25] * 512

    person_id = enroll_person(client, "  Asha Rao ", embedding)

    assert person_id.startswith("person-")
    assert len(person_id) == len("person-") + 12
    assert records == [(person_id, embedding, f"encrypted:Asha Rao:{person_id}")]


def test_liveness_is_fail_closed_for_missing_and_spoof_verdicts():
    checker = build()
    assert checker.is_live(None, SimpleNamespace(liveness={"status": "ok", "is_live": True}))
    assert not checker.is_live(None, SimpleNamespace(liveness={"status": "ok", "is_live": False}))
    assert not checker.is_live(None, SimpleNamespace(liveness={"status": "input_rejected"}))
    assert not checker.is_live(None, SimpleNamespace())


def test_face_detector_rejects_face_without_positive_liveness_verdict():
    fake_face = SimpleNamespace(
        bbox=np.array([0, 0, 100, 100]),
        det_score=0.99,
        liveness={"status": "ok", "is_live": False},
    )
    live_face = SimpleNamespace(
        bbox=np.array([0, 0, 100, 100]),
        det_score=0.99,
        liveness={"status": "ok", "is_live": True},
    )

    assert FaceDetector.best_face([fake_face], 0.6, 80) is None
    assert FaceDetector.best_face([live_face], 0.6, 80) is live_face


def test_camera_path_returns_clear_error_for_invalid_image():
    success, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert success

    with pytest.raises(ValueError, match="Could not decode the camera image"):
        extract_camera_embedding(b"not an image", "buffalo_l")


def test_camera_path_explains_when_no_face_is_detected(detector_stubs):
    detector_stubs([])
    success, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert success

    with pytest.raises(ValueError, match="face detect nahi hua"):
        extract_camera_embedding(encoded.tobytes(), "buffalo_l")


def test_camera_path_finds_live_face_across_burst_frames(detector_stubs):
    rejected_face = SimpleNamespace(
        bbox=np.array([0, 0, 70, 64]),
        det_score=0.66,
        liveness={"status": "input_rejected", "reason": "crop inconclusive"},
        normed_embedding=None,
    )
    live_face = SimpleNamespace(
        bbox=np.array([0, 0, 70, 64]),
        det_score=0.58,
        liveness={"status": "ok", "is_live": True, "live_score": 0.92},
        normed_embedding=np.ones(512, dtype=np.float32),
    )
    success, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert success

    detected_batches = iter([[rejected_face], [live_face]])
    detector_stubs(lambda frame: next(detected_batches))

    # A sequence of captured frames lets a later positive PAD verdict win.
    embedding = extract_camera_embedding([encoded.tobytes(), encoded.tobytes()], "buffalo_l")

    assert len(embedding) == 512


def test_camera_path_reports_pad_rejection_and_never_returns_embedding(detector_stubs):
    face = SimpleNamespace(
        bbox=np.array([0, 0, 70, 64]),
        det_score=0.58,
        liveness={"status": "ok", "is_live": False, "live_score": 0.001},
        normed_embedding=None,
    )
    detector_stubs([face])
    success, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert success

    with pytest.raises(ValueError, match=r"koi face PAD pass nahi hua \(live score 0.00\)"):
        extract_camera_embedding([encoded.tobytes()], "buffalo_l")


def test_allowed_plays_ting_with_pyttsx3(monkeypatch):
    calls = []

    class FakeEngine:
        def say(self, phrase):
            calls.append(("say", phrase))

        def runAndWait(self):
            calls.append(("run",))

    monkeypatch.setitem(
        __import__("sys").modules,
        "pyttsx3",
        SimpleNamespace(init=lambda: FakeEngine()),
    )

    play_feedback(allowed=True)

    assert calls == [("say", "ting"), ("run",)]


def test_denied_plays_beep(monkeypatch):
    calls = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "winsound",
        SimpleNamespace(Beep=lambda frequency, duration: calls.append((frequency, duration))),
    )

    play_feedback(allowed=False)

    assert calls == [(700, 700)]
