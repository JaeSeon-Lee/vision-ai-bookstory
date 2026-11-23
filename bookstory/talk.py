"""이야기가 끝난 뒤의 대화형 AI: 녹음 → 질문 이해·답변(Gemini) → 음성 합성(Gemini TTS) → 재생.

진행 방식 (Stage가 사용):
    talk.toggle_record()   M 키. 한 번 누르면 녹음, 다시 누르면 녹음을 끝내고 질문을 보낸다
    talk.poll()            매 프레임 호출. 끝난 요청의 결과를 받아 재생을 시작한다
    talk.cancel()          화면을 떠날 때. 녹음·요청·재생을 모두 멈추고 대화 기록을 지운다
    talk.state             "idle" / "recording" / "thinking" / "speaking" / "error"
    talk.subtitle()        지금 보여 줄 답변 자막 (없으면 "")

API 호출은 별도 스레드에서 해서 화면이 멈추지 않는다. 결과는 큐로 받아 메인 스레드(poll)에서 처리한다.
"""
from __future__ import annotations

import io
import json
import os
import queue
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QAudioSource, QMediaDevices, QtAudio

from .timeline import page_seconds

ROOT = Path(__file__).resolve().parent.parent

CHAT_MODEL = "gemini-3.8-flash"           # 질문 듣기 + 답 만들기
# 생각 단계. 기본값(높음)은 14초나 걸리고, "low"면 약 3초로 flash-lite만큼 빠르면서 답이 훨씬 낫다
# (2026-11-17 비교: lite는 답이 막연하고 "~단다" 반말이 섞였다). "minimal"은 이 모델이 받지 않는다
CHAT_THINKING = "low"
TTS_MODEL = "gemini-3.8-flash-lite-tts"   # 답을 목소리로 (약 4초)
TTS_VOICE = "Leda"                        # 부드러운 여성 목소리
# 말투 지시문("다정하게 말해 주세요: ...")을 앞에 붙이면 모델이 가끔 그 지시문까지 소리 내어 읽는다.
# 이 모델은 system_instruction도 받지 않으므로 대답 문장만 보낸다 (말투는 목소리 선택으로).

MIN_RECORD = 0.6      # 이보다 짧은 녹음(실수로 두 번 누름)은 보내지 않는다 (초)
MAX_RECORD = 30.0     # 녹음을 끄지 않아도 이 시간이 지나면 자동으로 끝내고 보낸다 (초)
HISTORY_TURNS = 6     # 앞선 질문·답을 몇 개까지 기억할지 (이어지는 질문용)
LINE_CHARS = 24       # 자막 한 줄 글자 수 (대략)

SYSTEM_PROMPT = """너는 그림자 구연 동화 「기다림의 달」 공연이 끝난 뒤, 동화를 들은 아이들의 질문에 답해 주는 다정한 이야기 선생님이야.
아이가 마이크에 대고 한 말이 음성으로 들어온다. 질문을 알아듣고 답해 줘.

[답하는 범위]
이 공연과 관련된 것이면 일반적인 질문도 답해 준다.
- 동화의 줄거리, 등장인물(새, 토끼, 여우, 사슴, 달, 구름), 장면, 그리고 "기다림"이라는 교훈
- 동화에 나온 자연과 동물에 대한 궁금증 (예: 달은 왜 구름에 가려지는지, 밤하늘의 별, 언덕·숲·연못, 새·토끼·여우·사슴은 어떻게 사는지)
- 그림자와 손그림자 놀이 (그림자가 생기는 까닭, 손으로 동물 그림자를 만드는 방법)
- 이 공연 자체 (손그림자를 카메라가 알아보고 동물 그림으로 바꿔 준다는 것 정도로 쉽게)
이 범위와 상관없는 질문(게임, 연예인, 숙제 풀이 등)이면 혼내지 말고 부드럽게 "그건 선생님이 잘 모르겠어요" 하고, 동화나 그림자에 대한 질문을 해 보도록 다정하게 이끌어 줘.
무섭거나 위험하거나 아이에게 맞지 않는 내용은 답하지 말고 부드럽게 다른 이야기로 돌려.

[말투]
- 만 4~8세 아이가 알아들을 쉬운 말. 모든 문장을 다정한 해요체("~했어요", "~이에요")로 끝내고, "~단다", "~란다", "~야" 같은 반말 어미는 쓰지 마.
- 1~3문장, 100자 안팎으로 짧게. 목소리로 읽어 주고 자막으로도 보여 주니까 이모지·기호·목록·괄호를 쓰지 마.
- 아이의 질문을 칭찬하거나 공감하는 말로 시작해도 좋다. 모르는 사실은 지어내지 마.
- 소리가 비었거나 알아듣기 어려우면 "잘 못 들었어요. 한 번 더 말해 줄래요?"처럼 다시 물어봐.

[동화 원문]
{story}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "아이가 한 말을 받아 적은 것 (못 알아들었으면 빈 문자열)"},
        "answer": {"type": "string", "description": "아이에게 들려줄 답"},
    },
    "required": ["question", "answer"],
}


# ---------------------------------------------------------------- 오디오
def _to_mono_int16(raw: bytes, fmt: QAudioFormat) -> np.ndarray:
    """마이크가 준 형식 그대로의 소리를 모노 16비트로 바꾼다."""
    sf = fmt.sampleFormat()
    if sf == QAudioFormat.SampleFormat.Int16:
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sf == QAudioFormat.SampleFormat.Int32:
        a = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648
    elif sf == QAudioFormat.SampleFormat.Float:
        a = np.frombuffer(raw, np.float32)
    else:  # UInt8
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    ch = max(1, fmt.channelCount())
    a = a[: len(a) // ch * ch].reshape(-1, ch).mean(axis=1)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


def _wav_bytes(pcm: np.ndarray, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


class Recorder:
    """기본 마이크로 녹음한다. stop()이 WAV 바이트를 돌려준다."""

    def __init__(self) -> None:
        self._source: QAudioSource | None = None
        self._io: QIODevice | None = None
        self._buf = bytearray()
        self._fmt = QAudioFormat()
        self.started = 0.0

    @property
    def active(self) -> bool:
        return self._source is not None

    def start(self) -> bool:
        device = QMediaDevices.defaultAudioInput()
        if device.isNull():
            return False
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        if not device.isFormatSupported(fmt):
            fmt = device.preferredFormat()
        self._fmt = fmt
        self._buf = bytearray()
        self._source = QAudioSource(device, fmt)
        self._io = self._source.start()
        if self._io is None or self._source.error() != QtAudio.Error.NoError:
            self._source = self._io = None
            return False
        self._io.readyRead.connect(self._read)
        self.started = time.perf_counter()
        return True

    def _read(self) -> None:
        if self._io is not None:
            self._buf.extend(self._io.readAll().data())

    def elapsed(self) -> float:
        return time.perf_counter() - self.started if self.active else 0.0

    def stop(self) -> tuple[bytes, float]:
        """녹음을 끝낸다. (WAV 바이트, 길이 초). 녹음 중이 아니면 (b"", 0)."""
        if self._source is None:
            return b"", 0.0
        self._read()
        self._source.stop()
        self._source = self._io = None
        pcm = _to_mono_int16(bytes(self._buf), self._fmt)
        rate = self._fmt.sampleRate()
        return _wav_bytes(pcm, rate), len(pcm) / max(1, rate)


class Player:
    """WAV 바이트를 기본 스피커로 재생한다."""

    def __init__(self) -> None:
        self._sink: QAudioSink | None = None
        self._buffer: QBuffer | None = None
        self.duration = 0.0

    def play(self, wav: bytes) -> bool:
        self.stop()
        with wave.open(io.BytesIO(wav)) as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            pcm = w.readframes(w.getnframes())
        if width != 2 or not pcm:
            return False
        fmt = QAudioFormat()
        fmt.setSampleRate(rate)
        fmt.setChannelCount(ch)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            return False
        self.duration = len(pcm) / (rate * ch * 2)
        self._buffer = QBuffer()
        self._buffer.setData(QByteArray(pcm))
        self._buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self._sink = QAudioSink(device, fmt)
        self._sink.start(self._buffer)
        return True

    def position(self) -> float:
        """지금까지 재생한 시간 (초)."""
        return self._sink.processedUSecs() / 1e6 if self._sink else 0.0

    @property
    def finished(self) -> bool:
        if self._sink is None:
            return True
        return (self._sink.state() != QtAudio.State.ActiveState
                and self.position() >= self.duration - 0.05)

    def pause(self) -> None:
        if self._sink is not None and self._sink.state() == QtAudio.State.ActiveState:
            self._sink.suspend()

    def resume(self) -> None:
        if self._sink is not None and self._sink.state() == QtAudio.State.SuspendedState:
            self._sink.resume()

    def stop(self) -> None:
        if self._sink is not None:
            self._sink.stop()
        self._sink = None
        if self._buffer is not None:
            self._buffer.close()
        self._buffer = None


# ---------------------------------------------------------------- Gemini
_client = None
_client_lock = threading.Lock()


def gemini_client():
    """Gemini 클라이언트 (대화·자막 읽어 주기가 함께 쓴다)."""
    global _client
    with _client_lock:
        if _client is None:
            from dotenv import load_dotenv
            from google import genai
            load_dotenv(ROOT / ".env")
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError("GEMINI_API_KEY가 없어요 (.env)")
            # TTS 모델은 v1beta에만 있다
            _client = genai.Client(http_options={"api_version": "v1beta"})
        return _client


def synthesize(text: str) -> bytes:
    """문장을 TTS_VOICE 목소리로 읽은 WAV 바이트. 지시문 없이 문장만 보낸다."""
    from google.genai import types
    r = gemini_client().models.generate_content(
        model=TTS_MODEL, contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE))),
        ))
    audio = r.candidates[0].content.parts[0].inline_data.data
    if not audio.startswith(b"RIFF"):   # 머리말 없는 PCM(24kHz 16비트 모노)이면 WAV로 감싼다
        audio = _wav_bytes(np.frombuffer(audio, np.int16), 24000)
    return audio


# ---------------------------------------------------------------- 자막
def answer_pages(text: str) -> list[str]:
    """답을 자막 페이지로 나눈다. 한 페이지는 최대 두 줄, 문장 단위로 끊는다."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", text.strip()) if s.strip()]
    lines: list[str] = []
    for s in sentences:
        line = ""
        for word in s.split():
            if line and len(line) + 1 + len(word) > LINE_CHARS:
                lines.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            lines.append(line)
        lines.append("")                      # 문장 끝 표시
    pages, cur = [], []
    for line in lines:
        if line == "":                        # 문장이 끝났고 이미 두 줄이면 페이지를 넘긴다
            if len(cur) >= 2:
                pages.append("\n".join(cur))
                cur = []
            continue
        if len(cur) == 2:
            pages.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        pages.append("\n".join(cur))
    return pages


# ---------------------------------------------------------------- 대화
def _polite(text: str) -> str:
    """모델이 가끔 섞는 "~단다/~란다"를 동화 원문과 같은 "~답니다/~랍니다"로 바꾼다."""
    return re.sub(r"([단란])다(?=[.!?]|$)", lambda m: {"단": "답", "란": "랍"}[m.group(1)] + "니다",
                  text, flags=re.M)


# ---------------------------------------------------------------- 대화
class Talk:
    def __init__(self) -> None:
        self.recorder = Recorder()
        self.player = Player()
        self.state = "idle"
        self.message = ""                 # error 상태에서 보여 줄 말
        self.history: list[tuple[str, str]] = []
        self._answer = ""
        self._pages: list[str] = []
        self._results: queue.Queue = queue.Queue()
        self._request = 0                 # 요청 번호 (취소된 요청의 결과는 버린다)
        self._error_until = 0.0

    # ---- 조작 ----
    def toggle_record(self) -> None:
        if self.state == "recording":
            self._finish_recording()
            return
        # 답하는 중이거나 생각하는 중이어도 새 질문을 받는다 (앞의 것은 그만둔다)
        self._request += 1
        self.player.stop()
        if self.recorder.start():
            self.state = "recording"
        else:
            self._fail("마이크를 찾을 수 없어요")

    def cancel(self) -> None:
        self._request += 1
        self.recorder.stop()
        self.player.stop()
        self.history.clear()
        self.state = "idle"
        self._answer, self._pages = "", []

    def poll(self) -> None:
        if self.state == "recording" and self.recorder.elapsed() >= MAX_RECORD:
            self._finish_recording()
        try:
            while True:
                req, result = self._results.get_nowait()
                if req == self._request:
                    self._on_result(result)
        except queue.Empty:
            pass
        if self.state == "speaking" and self.player.finished:
            self.state = "idle"
        if self.state == "error" and time.perf_counter() > self._error_until:
            self.state = "idle"

    # ---- 화면용 ----
    def subtitle(self) -> str:
        if self.state != "speaking" or not self._pages:
            return ""
        weights = [page_seconds(p) for p in self._pages]
        total = sum(weights)
        pos = self.player.position()
        acc = 0.0
        for page, wgt in zip(self._pages, weights):
            acc += self.player.duration * wgt / total
            if pos < acc:
                return page
        return self._pages[-1]

    # ---- 내부 ----
    def _fail(self, message: str) -> None:
        self.state = "error"
        self.message = message
        self._error_until = time.perf_counter() + 3.5

    def _finish_recording(self) -> None:
        wav, seconds = self.recorder.stop()
        if seconds < MIN_RECORD:
            self.state = "idle"
            return
        self.state = "thinking"
        req = self._request
        history = list(self.history)
        threading.Thread(target=self._work, args=(req, wav, history), name="talk",
                         daemon=True).start()

    def _on_result(self, result: dict) -> None:
        if "error" in result:
            self._fail(result["error"])
            return
        question, answer = result["question"], result["answer"]
        if question:
            self.history = (self.history + [(question, answer)])[-HISTORY_TURNS:]
        self._answer = answer
        self._pages = answer_pages(answer)
        if self.player.play(result["audio"]):
            self.state = "speaking"
        else:
            self._fail("스피커를 찾을 수 없어요")

    def _work(self, req: int, wav: bytes, history: list[tuple[str, str]]) -> None:
        """(스레드) 질문 이해·답 만들기 → 목소리 만들기."""
        try:
            from google.genai import types
            client = gemini_client()
            story = (ROOT / "story.md").read_text(encoding="utf-8")
            contents = []
            for q, a in history:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=q)]))
                contents.append(types.Content(role="model", parts=[types.Part.from_text(
                    text=json.dumps({"question": q, "answer": a}, ensure_ascii=False))]))
            contents.append(types.Content(role="user", parts=[
                types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                types.Part.from_text(text="방금 아이가 한 말이에요. 받아 적고 답해 주세요. "
                                        "답의 모든 문장은 '~요'로 끝내 주세요."),
            ]))
            r = client.models.generate_content(
                model=CHAT_MODEL, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT.format(story=story),
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.6,
                    thinking_config=types.ThinkingConfig(thinking_level=CHAT_THINKING),
                ))
            data = json.loads(r.text)
            question = data.get("question", "").strip()
            answer = data.get("answer", "").strip() or "잘 못 들었어요. 한 번 더 말해 줄래요?"
            answer = _polite(answer)
            if req != self._request:
                return
            audio = synthesize(answer)
            self._results.put((req, {"question": question, "answer": answer, "audio": audio}))
        except Exception as e:  # noqa: BLE001 — 네트워크·키 문제 등은 화면에 알리고 계속한다
            print(f"[talk] 오류: {e!r}")
            self._results.put((req, {"error": "지금은 대답하기 어려워요. 잠시 뒤에 다시 물어봐 줄래요?"}))
