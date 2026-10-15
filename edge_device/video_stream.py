"""Blocks 1 & 2: threaded RTSP ingest + resize / CLAHE / light denoise.
Preprocessing runs in the reader thread so the inference loop never waits on it."""
import logging
import queue
import threading
import time

import cv2

log = logging.getLogger("video")


class VideoStream:
    def __init__(self, url: str, width: int = 960, clahe: bool = True, denoise: bool = True, queue_size: int = 2):
        self.url = int(url) if url.isdigit() else url
        self.width, self.use_clahe, self.denoise = width, clahe, denoise
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)

    def read(self, timeout: float = 1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        if self.denoise:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
        if self.use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            frame = cv2.cvtColor(cv2.merge((self._clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        return frame

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                log.warning("camera open failed, retry in %.0fs", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            backoff = 1.0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log.warning("frame read failed, reconnecting")
                    break
                frame = self._preprocess(frame)
                if self._q.full():            # drop oldest: always serve the freshest frame
                    try:
                        self._q.get_nowait()
                    except queue.Empty:
                        pass
                self._q.put(frame)
            cap.release()
