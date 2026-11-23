"""자막 읽어 주기: 이야기 자막을 대화형 AI와 같은 목소리(Gemini TTS, Leda)로 읽는다.

진행 방식 (Stage가 사용):
    narrator.toggle()                 S 키. 켜고 끈다 (켜면 아직 없는 목소리를 뒤에서 만든다)
    narrator.update(cut, t, paused)   매 프레임 호출. 새 자막 페이지가 나타나면 그 페이지를 읽는다
    narrator.stop()                   다른 장면으로 건너뛸 때. 읽던 것을 멈춘다

목소리는 페이지마다 한 번만 만들어 assets/audio/narration/에 저장해 두고 다시 쓴다.
(만드는 데 한 페이지에 약 4초라 공연 중에 만들면 늦는다. tools/make_narration.py로 미리 만들 수 있다)
자막이 바뀌었을 때 앞 페이지를 아직 읽고 있으면 끊지 않고 다 읽은 뒤에 이어 읽는다.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path

from .talk import TTS_MODEL, TTS_VOICE, Player, synthesize
from .timeline import Cut, narration_page_at, narration_pages

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "assets" / "audio" / "narration"
RETRIES = 3          # 목소리 만들기가 실패하면 몇 번 더 해 볼지
LATE_LIMIT = 1.5     # 목소리가 늦게 준비됐을 때, 페이지가 나타난 지 이만큼(초)이 지났으면 읽지 않는다


def speech_text(page: str) -> str:
    """자막 한 페이지를 읽을 문장으로 (줄바꿈은 띄어쓰기로)."""
    return " ".join(line.strip() for line in page.split("\n") if line.strip())


def cache_path(text: str) -> Path:
    key = hashlib.sha1(f"{TTS_MODEL}|{TTS_VOICE}|{text}".encode("utf-8")).hexdigest()[:16]
    return CACHE / f"{key}.wav"


def all_texts(cuts: list[Cut]) -> list[str]:
    """이야기 순서대로 모든 자막 페이지의 읽을 문장 (같은 문장은 한 번만)."""
    out: list[str] = []
    for cut in cuts:
        for page in narration_pages(cut.narration):
            text = speech_text(page)
            if text and text not in out:
                out.append(text)
    return out


def _synthesize_patiently(text: str) -> bytes:
    """TTS 모델은 분당 요청 수 제한(429)이 있다. 걸리면 서버가 알려 준 시간만큼 기다렸다 다시 한다."""
    for _ in range(10):
        try:
            return synthesize(text)
        except Exception as e:  # noqa: BLE001
            if "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else 30.0
            print(f"[narrator] 요청 제한 — {wait:.0f}초 기다려요")
            time.sleep(wait)
    raise RuntimeError("요청 제한이 계속 풀리지 않아요")


def make_voice(text: str) -> Path:
    """그 문장의 목소리 파일을 (없으면 만들어) 돌려준다."""
    path = cache_path(text)
    if not path.exists():
        wav = _synthesize_patiently(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(wav)
        tmp.replace(path)          # 다 쓴 뒤에 이름을 바꿔서 반쯤 쓴 파일을 읽지 않게
    return path


class Narrator:
    def __init__(self, cuts: list[Cut]) -> None:
        self.cuts = cuts
        self.enabled = False
        self.player = Player()
        self._key: tuple | None = None        # 지금 보이는 자막 페이지 (컷 번호, 페이지 번호)
        self._shown_at = 0.0                  # 그 페이지가 나타난 실제 시각
        self._queued: str | None = None       # 앞 페이지를 다 읽으면 읽을 문장
        self._worker: threading.Thread | None = None
        self.missing = 0                      # 아직 만들지 못한 목소리 수 (화면 안내용)

    # ---- 조작 ----
    def toggle(self) -> None:
        self.enabled = not self.enabled
        if self.enabled:
            self._key = None                  # 켜자마자 지금 보이는 페이지부터 읽는다
            self._prepare()
        else:
            self.stop()

    def stop(self) -> None:
        self.player.stop()
        self._queued = None
        self._key = None

    def update(self, cut: Cut, t: float, paused: bool) -> None:
        if not self.enabled:
            return
        if paused:
            self.player.pause()
            return
        self.player.resume()
        at = narration_page_at(cut, t)
        key = (cut.number, at[0]) if at else None
        if key != self._key:
            self._key, self._shown_at = key, time.perf_counter()
            if at is not None:
                self._queued = speech_text(narration_pages(cut.narration)[at[0]])
        if self._queued and self.player.finished:
            text, self._queued = self._queued, None
            path = cache_path(text)
            if path.exists():
                self.player.play(path.read_bytes())
            elif time.perf_counter() - self._shown_at < LATE_LIMIT:
                self._queued = text           # 아직 만드는 중이면 조금 기다린다

    # ---- 목소리 미리 만들기 ----
    def _prepare(self) -> None:
        texts = [x for x in all_texts(self.cuts) if not cache_path(x).exists()]
        self.missing = len(texts)
        if texts and (self._worker is None or not self._worker.is_alive()):
            self._worker = threading.Thread(target=self._make_all, args=(texts,),
                                            name="narrator", daemon=True)
            self._worker.start()

    def _make_all(self, texts: list[str]) -> None:
        """(스레드) 없는 목소리를 이야기 순서대로 만든다."""
        for text in texts:
            for attempt in range(RETRIES + 1):
                try:
                    make_voice(text)
                    break
                except Exception as e:  # noqa: BLE001 — 네트워크·키 문제면 그 페이지만 건너뛴다
                    print(f"[narrator] 목소리 만들기 실패 ({attempt + 1}회): {e!r}")
                    time.sleep(2 * (attempt + 1))
            self.missing = max(0, self.missing - 1)
