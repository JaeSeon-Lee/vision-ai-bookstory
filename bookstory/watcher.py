"""웹캠을 지켜보며 그림자 모양을 인식한다 (별도 스레드).

카메라(Camera.latest())에서 최근 프레임을 가져와 학습된 인식기(silhouette.py)로 맞힌다.
- 시작하면 먼저 빈 스크린(배경)을 찍는다. 이때 스크린 앞에서 손을 치워 두어야 한다.
  나중에 B 키(recapture)로 다시 찍을 수 있다.
- 한 프레임의 결과만 믿지 않는다. 최근 STABLE_N 프레임 중 STABLE_K번 이상 같은 동물이어야
  "인식됨(stable)"으로 본다. 그림자가 들어오고 나가는 순간 잘못 맞히는 것을 걸러 낸다.

무대(Stage)는 매 화면마다 stable을 읽어, 인식된 동물이 있으면 on_shadow를 부른다.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

from .silhouette import MODEL_PATH, SilhouetteRecognizer

BG_FRAMES = 12      # 배경으로 쓸 프레임 수
BG_DELAY = 1.0      # 카메라가 켜진 뒤(노출이 자리 잡을 때까지) 이만큼 기다렸다가 배경을 찍는다
STABLE_N, STABLE_K = 8, 6
INTERVAL = 1 / 15   # 초당 15번 정도 인식한다


class ShadowWatcher:
    def __init__(self, camera) -> None:
        self.camera = camera
        self.status = "꺼짐"     # 꺼짐 / 준비 중 / 모델 없음 / 배경 찍는 중 / 인식 중 / 오류
        self.raw: tuple[str | None, float] = (None, 0.0)   # 이번 프레임의 결과
        self._stable: tuple[str, np.ndarray | None] | None = None
        self._history: deque[str | None] = deque(maxlen=STABLE_N)
        self._masks: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._want_bg = True
        self._rec: SilhouetteRecognizer | None = None
        self.diag = False                   # 진단 켬(D 키): 화면에 이유를 보인다 (저장하지 않음)
        self.info: list[str] = []           # 진단 글자 (한 칸이 한 줄, 첫 줄은 판정)
        self.outline: np.ndarray | None = None      # 찾은 스크린 (480폭 마스크)
        self.shadow: np.ndarray | None = None       # 찾은 그림자 (480폭 마스크)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stable(self) -> tuple[str, np.ndarray | None] | None:
        """(동물 이름, 그림자 마스크) — 안정적으로 인식된 그림자가 없으면 None."""
        with self._lock:
            return self._stable

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._want_bg = True
        self._thread = threading.Thread(target=self._loop, name="shadow-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.diag = False
        self._reset()
        self.status = "꺼짐"

    def recapture(self) -> None:
        """빈 스크린(배경)을 다시 찍는다. 조명이 바뀌었거나 카메라가 움직였을 때."""
        self._want_bg = True

    def toggle_diag(self) -> None:
        self.diag = not self.diag

    def _inspect(self, ms: float) -> None:
        """진단: 이번 프레임에서 인식기가 본 것을 화면용으로 남긴다. 파일로는 저장하지 않는다."""
        rec, s = self._rec, self._rec.last
        self.outline = rec.screen
        self.shadow = s.mask if s.valid else None
        verdict = s.reason or ("다섯 모양이 아님" if rec.last_name == "none" else rec.last_name)
        self.info = [verdict, f"진하기 {s.level:.2f} · 크기 {s.area:.2f}", f"선명도 {s.sharp:.2f} · {ms:.0f}ms"]

    def _reset(self) -> None:
        self._history.clear()
        self.raw = (None, 0.0)
        with self._lock:
            self._stable = None

    def _loop(self) -> None:
        if self._rec is None:
            self.status = "준비 중"          # 모델(sklearn) 불러오기에 몇 초 걸린다
            if not MODEL_PATH.exists():
                self.status = "모델 없음"
                self._running = False
                return
            try:
                self._rec = SilhouetteRecognizer.load()
            except Exception as e:   # 모델 파일이 깨졌거나 sklearn이 없을 때
                print(f"[그림자 인식] 모델을 불러오지 못했다: {e}")
                self.status = "오류"
                self._running = False
                return
        last = None
        while self._running:
            if self._want_bg:
                self._capture_background()
                continue
            frame = self.camera.latest()
            if frame is None or frame is last:
                time.sleep(0.01)
                continue
            last = frame
            t0 = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            try:
                name, prob, mask = self._rec.predict(gray)
            except Exception as e:
                print(f"[그림자 인식] {e}")
                name, prob, mask = None, 0.0, None
            self.raw = (name, prob)
            if self.diag:
                try:
                    self._inspect(1000 * (time.perf_counter() - t0))
                except Exception as e:      # 진단이 실패해도 인식은 계속한다
                    print(f"[그림자 진단] {e}")
            self._history.append(name)
            if name and mask is not None:
                self._masks[name] = mask
            top, count = Counter(self._history).most_common(1)[0]
            with self._lock:
                self._stable = (top, self._masks.get(top)) if top and count >= STABLE_K else None
            time.sleep(max(0.0, INTERVAL - (time.perf_counter() - t0)))

    def _capture_background(self) -> None:
        self.status = "배경 찍는 중"
        self._reset()
        self._want_bg = False
        start = time.perf_counter()
        frames, last = [], None
        while self._running and len(frames) < BG_FRAMES:
            if self._want_bg:            # 찍는 중에 다시 찍기를 눌렀으면 처음부터
                return
            frame = self.camera.latest()
            if frame is None or frame is last or time.perf_counter() - start < BG_DELAY:
                time.sleep(0.03)
                continue
            last = frame
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        if frames:
            self._rec.set_background(frames)
            self.status = "인식 중"
