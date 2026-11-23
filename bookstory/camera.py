"""웹캠 읽기. 별도 스레드에서 계속 읽어 가장 최근 프레임만 들고 있는다.

화면 재생(메인 스레드)이 카메라를 기다리느라 멈추지 않게 하기 위해서다.
그림자 인식기(watcher.py)도 latest()로 같은 프레임을 가져다 쓴다.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index, self.width, self.height = index, width, height
        self._frame: np.ndarray | None = None   # 최근 프레임 (RGB)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self.status = "꺼짐"                      # 꺼짐 / 연결 중 / 켜짐 / 카메라 없음

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.status = "연결 중"
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._frame = None
        self.status = "꺼짐"

    def latest(self) -> np.ndarray | None:
        """가장 최근 프레임 (RGB, HxWx3). 아직 없으면 None."""
        with self._lock:
            return self._frame

    def _loop(self) -> None:
        # Windows에서는 DirectShow가 가장 빨리 열린다
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            self.status = "카메라 없음"
            self._running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        failures = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    self.status = "카메라 없음"
                    break
                time.sleep(0.02)
                continue
            failures = 0
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frame = rgb
            self.status = "켜짐"
        cap.release()
        self._running = False
