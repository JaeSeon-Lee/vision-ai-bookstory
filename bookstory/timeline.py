"""컷과 레이어 애니메이션을 정의하는 자료 구조.

모든 좌표는 1920x1080 기준 화면 좌표다. 창 크기가 달라도 앱이 비율을 맞춰 확대·축소한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


def _clamp(t: float) -> float:
    return min(max(t, 0.0), 1.0)


def ease_in_out(t: float) -> float:
    """처음과 끝이 부드러운 곡선 (smoothstep)."""
    t = _clamp(t)
    return t * t * (3 - 2 * t)


def linear(t: float) -> float:
    """일정한 빠르기. 이어지는 구간에서 멈칫하지 않는다."""
    return _clamp(t)


def ease_in(t: float) -> float:
    """천천히 출발해 점점 빨라진다 (떨어질 때, 날아가 버릴 때)."""
    t = _clamp(t)
    return t * t


def ease_out(t: float) -> float:
    """빠르게 출발해 점점 느려진다 (뛰어오를 때, 멈춰 설 때)."""
    t = _clamp(t)
    return 1 - (1 - t) * (1 - t)


def back_out(t: float) -> float:
    """목표를 살짝 지나쳤다가 돌아온다 (착지 뒤 출렁임)."""
    t = _clamp(t)
    c = 1.70158
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


EASES: dict[str, Callable[[float], float]] = {
    "inout": ease_in_out, "linear": linear, "in": ease_in, "out": ease_out, "back": back_out,
}


@dataclass
class Track:
    """시간(초)에 따라 값이 변하는 키프레임 목록. [(시간, 값), ...] 또는 [(시간, 값, 이징), ...]

    이징은 앞 키에서 이 키로 오는 구간의 곡선이다 ("inout" 기본, "linear", "in", "out", "back").
    """
    keys: list[tuple]

    def at(self, t: float) -> float:
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1]
        for i, (k0, k1) in enumerate(zip(keys, keys[1:])):
            t0, v0, t1, v1 = k0[0], k0[1], k1[0], k1[1]
            if t == t1:
                # 같은 시각에 키가 여럿이면(값이 순간적으로 바뀜) 그 시각부터 마지막 키의 값.
                # 자세 그림(Layer.poses)도 그 시각부터 바뀌므로 둘이 어긋나지 않는다
                j = i + 1
                while j + 1 < len(keys) and keys[j + 1][0] == t1:
                    j += 1
                return keys[j][1]
            if t <= t1:
                if t1 <= t0:
                    return v1
                ease = EASES[k1[2]] if len(k1) > 2 else ease_in_out
                return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
        return keys[-1][1]


def const(v: float) -> Track:
    return Track([(0.0, v)])


@dataclass
class Wiggle:
    """계속 되풀이되는 흔들림. 숨 쉬기, 날갯짓 출렁임, 걸음마다 통통 튀기 등.

    prop  : "x", "y", "rotation", "scale_x", "scale_y", "scale"(가로·세로 함께)
            scale 계열은 1에 더해지는 비율이다 (0.02 = 2%).
            "flap" : 날갯짓 (Layer.flap 프레임이 있을 때만). +1 = 날개 끝까지 내려침,
                     -1 = 끝까지 들어 올림, 0 = 원래 그림 자세
    wave  : "sin" (오르내림) 또는 "bounce" (|sin|, 걸음처럼 한쪽으로만 튐 — amp를 음수로 주면 위로)
    phase : 초. 앞 컷에서 이어지는 흔들림이면 앞 컷까지의 길이를 넣어 끊기지 않게 한다
    start/end, ramp_in/ramp_out : 흔들림이 켜지는 구간과 서서히 켜지고 꺼지는 시간
    """
    prop: str
    amp: float
    period: float
    phase: float = 0.0
    wave: str = "sin"
    start: float = -math.inf
    end: float = math.inf
    ramp_in: float = 0.0
    ramp_out: float = 0.0

    def at(self, t: float) -> float:
        if t < self.start or t > self.end:
            return 0.0
        gain = 1.0
        if self.ramp_in > 0:
            gain = min(gain, ease_in_out((t - self.start) / self.ramp_in))
        if self.ramp_out > 0:
            gain = min(gain, ease_in_out((self.end - t) / self.ramp_out))
        s = math.sin(2 * math.pi * (t + self.phase) / self.period)
        if self.wave == "bounce":
            s = abs(s)
        return self.amp * gain * s


@dataclass
class Layer:
    """배경 위에 겹치는 이미지 레이어 (구름 등)."""
    image: str                      # assets/images/layers/ 기준 파일 이름
    x: Track                        # 레이어 중심 x
    y: Track                        # 레이어 중심 y
    width: Track                    # 화면에서의 폭 (높이는 비율 유지)
    opacity: Track = field(default_factory=lambda: const(1.0))
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # RGB 곱 (밤 색조용)
    above_dark: bool = False        # True면 어둠 위에 그린다 (완성된 동물처럼 밝게 보여야 할 때)
    # ---- 역동감 (동물) ----
    rotation: Track = field(default_factory=lambda: const(0.0))  # 도. 오른쪽 보는 동물은 -가 고개 듦
    scale_x: Track = field(default_factory=lambda: const(1.0))   # 가로 배율 (음수면 좌우 뒤집기)
    scale_y: Track = field(default_factory=lambda: const(1.0))   # 세로 배율 (찌그러짐·늘어남)
    pivot: tuple[float, float] = (0.0, 0.0)  # 회전·배율 중심. 그림 크기 기준, (0, 0.5)=발밑 가운데
    wiggles: list[Wiggle] = field(default_factory=list)
    # True면 컷의 그림자 연출이 끝난(done_at) 뒤부터 완성된 동물 그림 대신 이 레이어를 그린다
    after_shadow: bool = False
    # 날갯짓 프레임 폴더 (assets/images/layers/ 기준, frames.json이 든 곳. tools/split_bird.py가 만든다).
    # 있으면 "flap" 흔들림 값에 맞는 날개 각도의 프레임을 image 대신 그린다
    flap: str = ""
    # 시점별로 바꿔 그리는 자세 그림 [(시각, 그림), ...] (시각 순). t초에는 t 이전의 마지막 그림.
    # 비었거나 첫 시각 전이면 image. image와 다른 그림은 layers/<그림>.json의 box 범위에 그린다
    # (tools/align_pose.py가 만든다. 자세가 바뀌어도 크기·땅 높이가 같도록)
    poses: list[tuple[float, str]] = field(default_factory=list)
    pose_fade: float = 0.0   # 자세 그림이 바뀔 때 앞 그림과 섞는 시간 (초). 0이면 바로 바뀐다

    def image_at(self, t: float) -> str:
        """t초에 그릴 자세 그림."""
        return self.images_at(t)[-1][0]

    def images_at(self, t: float) -> list[tuple[str, float]]:
        """t초에 그릴 자세 그림들 [(그림, 진하기)] (그리는 순서대로).
        바뀌는 중이면 섞는 시간의 앞 절반은 앞 그림 위에 새 그림이 진해지고, 뒤 절반은 새 그림
        위에 앞 그림이 옅어진다 (가운데에서 둘 다 진하기 1이라 튀지 않고, 반투명해지지도 않는다)."""
        prev, name, since = self.image, self.image, math.inf
        for when, art in self.poses:
            if when > t:
                break
            prev, name, since = name, art, t - when
        if prev == name or since >= self.pose_fade:
            return [(name, 1.0)]
        f = since / self.pose_fade
        return [(prev, 1.0), (name, 2 * f)] if f < 0.5 else [(name, 1.0), (prev, 2 * (1 - f))]

    def flap_at(self, t: float) -> float:
        """t초의 날갯짓 값 (-1 들어 올림 ~ 0 원래 자세 ~ +1 내려침)."""
        return max(-1.0, min(1.0, sum(wg.at(t) for wg in self.wiggles if wg.prop == "flap")))

    def pose(self, t: float) -> tuple[float, float, float, float, float]:
        """t초의 (x, y, 회전, 가로 배율, 세로 배율). 흔들림까지 더한 값."""
        w = {"x": 0.0, "y": 0.0, "rotation": 0.0, "scale_x": 0.0, "scale_y": 0.0, "scale": 0.0,
             "flap": 0.0}
        for wg in self.wiggles:
            w[wg.prop] += wg.at(t)
        return (self.x.at(t) + w["x"], self.y.at(t) + w["y"],
                self.rotation.at(t) + w["rotation"],
                self.scale_x.at(t) * (1 + w["scale_x"] + w["scale"]),
                self.scale_y.at(t) * (1 + w["scale_y"] + w["scale"]))


@dataclass
class ShadowSpec:
    """컷 안에서 동물 그림자를 기다렸다가 완성하는 연출.

    0 ~ wait_at초      : 동물 자리에 실루엣(목표 그림자)이 서서히 나타난다
    wait_at초          : 그림자가 인식될 때까지 시간이 멈추고 실루엣이 은은하게 숨 쉰다
    인식 후 0~0.6초    : 인식된 손그림자가 실루엣 자리에 나타난다
    인식 후 0.6~2.8초  : 손그림자가 동물 실루엣 모양으로 서서히 바뀐다
    인식 후 2.8~4.0초  : 실루엣에 색이 입혀지며 동물 그림이 완성된다
    """
    animal: str        # "bird" 등 (인식기가 보내는 이름)
    art: str           # assets/images/layers/ 기준 동물 그림 (투명 PNG).
                       # 비우면 "신호" 그림자(예: 달): 인식되면 그림자가 빛 속으로 사라지기만 한다
    x: float           # 동물 중심 x (1920x1080 기준)
    y: float           # 동물 중심 y
    width: float       # 화면에서의 폭
    wait_at: float = 1.2

    APPEAR = 0.6       # 손그림자가 나타나는 시간
    MORPH = 2.2        # 모양이 바뀌는 시간
    COLOR = 1.2        # 색이 입혀지는 시간
    FADE = 0.8         # 신호 그림자가 사라지는 시간

    @property
    def reveal_time(self) -> float:
        if not self.art:
            return self.APPEAR + self.FADE
        return self.APPEAR + self.MORPH + self.COLOR

    @property
    def done_at(self) -> float:
        """컷 시간 기준으로 동물 그림이 완성되는 시점."""
        return self.wait_at + self.reveal_time


@dataclass
class ShadowHold:
    """손그림자를 계속 유지하고 있으면 컷의 at초에서 멈춰 머무는 연출.

    at초에 도달했을 때 animal 그림자가 아직 유지되고 있으면 이야기 시간이 at에서 멈추고,
    그동안 layers를 loop초 길이로 되풀이해 그린다 (레이어 시각 0 ~ loop). layers가 비었으면
    컷의 레이어를 at + (0 ~ loop)초로 되풀이해 그린다 (흔들림만 이어지는 경우).
    그림자를 풀면 지금 되풀이의 다음 멈출 자리(exits 또는 loop 끝)까지 마저 움직인 뒤 이어서 진행한다.
    그래서 시각 0, exits, loop에서의 모습은 at초의 컷 모습과 같아야 튀지 않는다.
    at초에 그림자를 유지하고 있지 않으면 멈추지 않고 그대로 지나간다.
    """
    animal: str
    at: float
    loop: float
    layers: list[Layer] = field(default_factory=list)
    exits: tuple[float, ...] = ()

    def exit_after(self, u: float) -> float:
        """멈춘 지 u초에 그림자를 풀었을 때 이어서 진행할 시각 (멈춘 때부터 잰 초)."""
        base = math.floor(u / self.loop) * self.loop
        local = u - base
        return base + min(e for e in (*self.exits, self.loop) if e >= local)


@dataclass
class Cut:
    number: int
    title: str
    background: str                 # assets/images/scenes/ 기준 파일 이름
    duration: float                 # 초. 끝나면 마지막 장면에서 멈춘다
    layers: list[Layer] = field(default_factory=list)
    darkness: Track = field(default_factory=lambda: const(0.0))  # 0~1, 화면을 남색으로 어둡게
    narration: str = ""
    # 다음 배경으로 서서히 바뀌는 연출. next_background가 레이어까지 덮으며 나타난다
    next_background: str = ""
    next_mix: Track = field(default_factory=lambda: const(0.0))  # 0=현재 배경, 1=다음 배경
    next_under_layers: bool = False  # True면 다음 배경이 레이어(구름) 아래에서 바뀐다
    warmth: Track = field(default_factory=lambda: const(0.0))   # 0~1, 따뜻한 빛 (엔딩)
    narration_until: float = 0.0     # 0이 아니면 이 시점에 자막이 서서히 사라진다
    title_alpha: Track = field(default_factory=lambda: const(0.0))  # "기다림의 달" 제목 진하기
    hold: bool = False               # True면 자동으로 넘어가지 않고 Space를 기다린다 (메인 화면)
    # 동물 그림자 연출. 있으면 wait_at 시점에서 그림자가 인식될 때까지 기다린다
    shadow: ShadowSpec | None = None
    # 나레이션이 나타나는 시점 (초). 그림자 연출이 있는 컷은 동물이 완성된 뒤로 둔다
    narration_from: float = 0.0
    # 그림자를 계속 유지하면 머무는 연출 (풀면 다음으로)
    shadow_hold: ShadowHold | None = None
    # 질문 받기 (마지막 컷). 이 시점부터 화면 가운데에 질문 안내가 떠오르고 대화형 AI가 켜진다.
    # 자동으로 넘어가지 않고, Space를 누르면 메인 화면으로 돌아간다
    ask_from: float | None = None


def page_seconds(page: str) -> float:
    """자막 한 장을 읽어 주는 데 필요한 시간 (구연 속도 기준)."""
    letters = sum(1 for ch in page if not ch.isspace())
    return max(3.0, 1.2 + letters * 0.16)


def narration_pages(narration: str) -> list[str]:
    """빈 줄로 나눈 자막 페이지들."""
    return [p.strip("\n") for p in narration.split("\n\n") if p.strip()]


def narration_page_at(cut: "Cut", t: float) -> tuple[int, float] | None:
    """t초에 보이는 자막 페이지 (번호, 그 페이지가 시작된 시각). 자막이 없는 때면 None.
    각 페이지는 글자 수에 맞춰 narration_from ~ 끝 구간을 나눠 받는다."""
    pages = narration_pages(cut.narration)
    end = cut.narration_until or cut.duration
    if t < cut.narration_from or t >= end or not pages:
        return None
    span = max(0.1, end - cut.narration_from)
    weights = [page_seconds(pg) for pg in pages]
    total = sum(weights)
    start = cut.narration_from
    for i, wgt in enumerate(weights):
        length = span * wgt / total
        if t < start + length or i == len(pages) - 1:
            return i, start
        start += length
    return None


def narration_length(narration: str) -> float:
    return sum(page_seconds(p) for p in narration_pages(narration))


# 그림자 이름 → 화면 표시용 한글
SHADOW_NAMES = {"bird": "새", "rabbit": "토끼", "fox": "여우", "deer": "사슴", "moon": "달"}
