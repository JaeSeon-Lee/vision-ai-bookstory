"""컷별 연출 정의. 나레이션은 story.md의 나레이션 부분을 따른다.

컷은 자동으로 이어서 재생된다. shadow가 있는 컷(3 새, 6 토끼, 9 여우, 12 사슴)은 동물 자리에
따라 만들 손그림자 모양을 보여 주고, 그 그림자가 인식될 때까지 기다린 뒤 손그림자를 동물로
바꿔 완성한다. 컷 14는 달 그림자를 기다렸다가 바람이 불어 구름이 걷히게 한다.
shadow_hold가 있는 컷(3 새, 7 토끼, 10 여우)은 그 대사 뒤에도 손그림자를 계속 유지하고 있으면
그 자리에 머무르며 동물이 움직이고(여우는 두리번거림), 그림자를 풀면 다음으로 넘어간다.
나레이션은 빈 줄로 나누면 여러 장의 자막으로 차례로 넘어간다.

좌표는 모두 1920x1080 기준. 동물 위치는 각 배경의 빈 자리에 맞췄다.
"""
import json
import math
from pathlib import Path

from .timeline import (Cut, Layer, ShadowHold, ShadowSpec, Track, Wiggle, const,
                       narration_length, narration_pages, page_seconds)

# 동물별로 따라 만들 손그림자 모양 (assets/images/ 기준, 0/255 흑백 마스크 PNG).
# 그림자 영상(assets/video/dark)에서 tools/make_guides.py로 뽑았다 — 인식기가 학습한 바로 그 손 모양.
# 그림자를 기다리는 동안 이 모양을 동물 자리에 보여 줘서 무엇을 만들지 알려 준다 (글자 안내는 없음).
# 시험용 키(1~5)로 인식을 흉내 내면 이 모양이 인식된 것으로 보고 동물로 바꾼다.
SHADOW_GUIDES = {a: f"guides/{a}.png" for a in ("bird", "rabbit", "fox", "deer", "moon")}

NIGHT_DARKNESS = 0.44   # 달이 가려진 뒤의 어둡기 (컷 2 끝 ~ 컷 14)
NIGHT_TINT = (0.95, 0.95, 1.0)   # 생성된 구름이 배경 구름보다 살짝 하얘서 아주 약하게 푸르게
TRANSITION = 2.0        # 장소가 바뀔 때 다음 배경으로 서서히 넘어가는 시간


# ---------------------------------------------------------------- 도우미
FEET = (0.0, 0.45)   # 땅 위 동물의 회전·찌그러짐 중심 (발밑 가운데)
LAYERS = Path(__file__).resolve().parent.parent / "assets" / "images" / "layers"


def pose_place(x: float, y: float, width: float, pose: str) -> dict:
    """기준 그림이 화면 (x, y, width)에 있을 때 자세 그림이 놓이는 자리 dict(x, y, width).
    자리는 layers/<자세>.json의 box (tools/align_pose.py). 자세 그림을 따로 쓸 때
    (그림자 연출 등) 기준 그림 레이어 안에서 바뀌는 것과 똑같은 크기·자리가 된다."""
    meta = json.loads((LAYERS / Path(pose).with_suffix(".json").name).read_text(encoding="utf-8"))
    (bw, bh), (x0, y0, x1, y1) = meta["size"], meta["box"]
    height = width * bh / bw
    return dict(x=x + ((x0 + x1) / 2 / bw - 0.5) * width,
                y=y + ((y0 + y1) / 2 / bh - 0.5) * height,
                width=(x1 - x0) / bw * width)


def _track(v) -> Track:
    return v if isinstance(v, Track) else const(v)


def animal(art: str, x, y, width, opacity=None, **motion) -> Layer:
    """완성된 동물 레이어. 숫자를 주면 고정, Track을 주면 움직인다.
    motion: rotation, scale_x, scale_y (숫자 또는 Track), pivot, wiggles"""
    for k in ("rotation", "scale_x", "scale_y"):
        if k in motion:
            motion[k] = _track(motion[k])
    return Layer(image=art, x=_track(x), y=_track(y), width=_track(width),
                 opacity=_track(1.0 if opacity is None else opacity), above_dark=True, **motion)


def settled(spec: ShadowSpec, wiggles: list[Wiggle], pivot=(0.0, 0.0), flap: str = "") -> Layer:
    """그림자 연출로 완성된 동물이 그 자리에서 이어서 살아 움직이는 레이어.
    완성(done_at) 직후에는 흔들림이 0에서 서서히 켜져서 완성된 그림과 똑같이 이어진다."""
    return Layer(image=spec.art, x=const(spec.x), y=const(spec.y), width=const(spec.width),
                 above_dark=True, after_shadow=True, pivot=pivot, wiggles=wiggles, flap=flap)


def breathing(phase: float = 0.0, amp: float = 0.015, period: float = 3.0,
              start: float = -math.inf, end: float = math.inf,
              ramp_in: float = 0.0, ramp_out: float = 0.0) -> list[Wiggle]:
    """가만히 있을 때 숨 쉬듯 몸이 세로로 아주 조금 부풀었다 가라앉는다 (pivot=FEET와 함께)."""
    return [Wiggle("scale_y", amp, period, phase, start=start, end=end,
                   ramp_in=ramp_in, ramp_out=ramp_out)]


BIRD_FLAP = "bird_flap"   # 새 날갯짓 프레임 폴더 (tools/split_bird.py)


def flutter(phase: float = 0.0, start: float = -math.inf, ramp_in: float = 0.0) -> list[Wiggle]:
    """날갯짓에 맞춰 몸이 오르내리고, 살짝 기울고, 날개 치는 박자로 조금씩 늘었다 줄어든다.
    flap 프레임이 있는 레이어는 날개도 퍼덕인다. 날개를 내려치는 동안 몸이 받쳐 올려지도록
    맞췄다 (내려치기 중간에 몸이 가장 낮고, 다 내려쳤을 때 가장 빠르게 떠오른다)."""
    beat = 1.4
    return [Wiggle("y", 14, beat, phase, start=start, ramp_in=ramp_in),
            Wiggle("rotation", 3.0, beat, phase + beat / 4, start=start, ramp_in=ramp_in),
            Wiggle("scale_y", 0.03, beat / 2, phase, start=start, ramp_in=ramp_in),
            Wiggle("flap", 1.0, beat, phase - beat / 4, start=start, ramp_in=ramp_in)]


def walk(start: float, end: float, step: float = 0.42, bob: float = 7.0, sway: float = 1.6,
         ramp_out: float = 0.25) -> list[Wiggle]:
    """걸을 때 발걸음마다 통통 튀고(bounce) 몸이 앞뒤로 살짝 기우뚱한다."""
    return [Wiggle("y", -bob, step * 2, -start, "bounce", start, end, 0.25, ramp_out),
            Wiggle("rotation", sway, step * 2, -start, "sin", start, end, 0.25, ramp_out)]


def hops(t0: float, t1: float, start: tuple, end: tuple, count: int, height: float,
         lean: float = 10.0, art: str = "", land: str = "", landed: bool = False) -> dict:
    """폴짝폴짝 뛰는 움직임. start/end = (x, y, width). animal()에 넘길 움직임을 돌려준다.

    art/land를 주면 자세 그림도 바꾼다: 땅에 있을 때(웅크림·착지)는 land(다리를 오므린 그림),
    공중에서는 art(뛰는 그림). 그림은 납작해지고 늘어나는 한가운데에서 바뀐다.
    landed: 처음부터 land 자세로 시작한다 (앞 컷이 착지로 끝났을 때).

    한 번 뛸 때마다: 웅크림 → 몸을 늘이고 고개를 들며 도약 → 공중에서 앞으로 → 고개를 숙이며
    떨어짐 → 착지할 때 납작해짐 (= 다음 도약의 웅크림). x는 공중에서만 일정한 빠르기로 움직이고,
    y는 올라갈 때 감속·떨어질 때 가속해서 포물선을 그린다. 마지막 착지 뒤에는 출렁이며 선다.
    """
    T = (t1 - t0) / count
    point = lambda k: tuple(start[j] + (end[j] - start[j]) * k for j in range(3))
    squash, stretch = (1.12, 0.84), (0.9, 1.14)   # (가로, 세로) 배율
    if land:
        squash = (1.05, 0.93)   # 착지 그림이 이미 웅크린 모양이라 조금만 더 납작하게
    x, y, w = [(t0, start[0])], [(t0, start[1])], [(t0, start[2])]
    rot, sx, sy = [(t0, 0.0)], [(t0, 1.0)], [(t0, 1.0)]
    poses = [(-math.inf, land if landed else art)] if land else []
    for i in range(count):
        a = t0 + i * T
        p0, p1 = point(i / count), point((i + 1) / count)
        if land:
            if i == 0 and not landed:
                poses.append((a + 0.08 * T, land))          # 웅크리며 다리를 모은다
            poses.append((a + 0.23 * T, art))               # 몸을 늘이며 다리를 뻗는다
            poses.append((a + 0.97 * T, land))              # 땅에 닿으며 다리를 오므린다
        crouch, launch, land_at = a + 0.16 * T, a + 0.3 * T, a + T
        apex, before_land = (launch + land_at) / 2, land_at - 0.12 * T
        # 웅크림
        sx.append((crouch, squash[0], "out")); sy.append((crouch, squash[1], "out"))
        rot.append((crouch, 3.0))
        # 도약: 몸이 늘어나며 고개를 든다
        x.append((launch, p0[0])); y.append((launch, p0[1])); w.append((launch, p0[2]))
        sx.append((launch, stretch[0], "out")); sy.append((launch, stretch[1], "out"))
        rot.append((launch, -lean, "out"))
        # 가장 높은 곳
        y.append((apex, (p0[1] + p1[1]) / 2 - height, "out"))
        sx.append((apex, 1.0)); sy.append((apex, 1.0))
        rot.append((apex, 0.0))
        # 떨어지며 고개를 숙이고, 닿기 직전 살짝 늘어났다가
        sx.append((before_land, 0.95)); sy.append((before_land, 1.06))
        rot.append((before_land, lean * 0.7))
        # 착지: 납작해진다
        x.append((land_at, p1[0], "linear")); y.append((land_at, p1[1], "in"))
        w.append((land_at, p1[2], "linear"))
        sx.append((land_at, squash[0], "in")); sy.append((land_at, squash[1], "in"))
        rot.append((land_at, 2.0, "out"))
    settle = t1 + 0.45                              # 출렁이며 제자리
    sx.append((settle, 1.0, "back")); sy.append((settle, 1.0, "back"))
    rot.append((settle, 0.0))
    return dict(x=Track(x), y=Track(y), width=Track(w), rotation=Track(rot),
                scale_x=Track(sx), scale_y=Track(sy), pivot=FEET, poses=poses)


def fade_to_next(duration: float) -> Track:
    """컷 끝에서 다음 배경으로 넘어가는 섞기 값."""
    return Track([(0.0, 0.0), (duration - TRANSITION, 0.0), (duration, 1.0)])


# ---------------------------------------------------------------- 장면 1: 달이 사라진 밤
MOON_01 = (1389, 213)   # bg_01의 달 중심 (지름 약 230px)
CLOUD_WIDTH = 1000
CLOUD_Y = 215
CLOUD_START_X = 2520    # 화면 오른쪽 밖
CLOUD_NEAR_X = 1830     # 컷 1 끝: 달 옆까지 다가옴
CLOUD_END_X = 1410      # 컷 2 끝: 달을 덮고, bg_02의 구름 덩어리와 겹치는 자리

CUT_1 = Cut(
    number=1, title="달이 비추는 밤", background="bg_01.png", duration=7.0,
    layers=[Layer(image="cloud_01.png", x=Track([(0.0, CLOUD_START_X), (7.0, CLOUD_NEAR_X)]),
                  y=const(CLOUD_Y), width=const(CLOUD_WIDTH), tint=NIGHT_TINT)],
    narration="어느 고요한 밤이었어요.\n둥근 달님이 하늘을 비추고 있었는데,",
)

CUT_2 = Cut(
    number=2, title="달이 구름에 가려짐", background="bg_01.png", duration=7.0,
    layers=[Layer(image="cloud_01.png", x=Track([(0.0, CLOUD_NEAR_X), (5.5, CLOUD_END_X)]),
                  y=const(CLOUD_Y), width=const(CLOUD_WIDTH), tint=NIGHT_TINT)],
    darkness=Track([(0.0, 0.0), (1.5, 0.0), (6.5, NIGHT_DARKNESS)]),
    # 구름이 달을 덮은 뒤 그림 속 구름이 달을 가린 bg_02로 서서히 바뀐다
    next_background="bg_02.png", next_mix=Track([(0.0, 0.0), (5.0, 0.0), (6.8, 1.0)]),
    narration="커다란 구름이 스르르 다가오더니 달님을 감춰 버렸답니다.\n달님은 어디로 갔을까요?",
)

# ---------------------------------------------------------------- 장면 2: 새의 도전 (bg_02)
BIRD = ShadowSpec(animal="bird", art="animal_bird.png", x=1000, y=400, width=560)
BIRD_HIGH = (1180, 200, 360)

N3 = "달님을 찾고 싶었던 새는 힘차게 날갯짓을 했어요.\n\"여기보다 더 높은 곳에 있나 봐!\""
D3 = BIRD.done_at + narration_length(N3)
CUT_3 = Cut(
    number=3, title="새가 달을 찾음", background="bg_02.png",
    duration=D3, darkness=const(NIGHT_DARKNESS),
    shadow=BIRD, narration_from=BIRD.done_at, narration=N3,
    # 완성되자마자 제자리에서 날갯짓하며 떠 있는다
    layers=[settled(BIRD, flutter(start=BIRD.done_at, ramp_in=1.0), flap=BIRD_FLAP)],
    # "여기보다 더 높은 곳에 있나 봐!" 뒤에도 새 그림자를 유지하면 제자리에서 계속 날갯짓한다.
    # 날갯짓 한 번(1.4초) 단위로 되풀이하므로 풀면 그 날갯짓을 마치고 컷 4로 날아오른다
    shadow_hold=ShadowHold("bird", at=D3, loop=1.4),
)

N4 = "새는 점점 더 높이, 더 멀리 날아갔어요."
D4 = max(5.0, narration_length(N4))
T4_DIP = 0.5   # 솟아오르기 전에 살짝 가라앉으며 힘을 모으는 때
CUT_4 = Cut(
    number=4, title="더 높이 날아가는 새", background="bg_02.png", duration=D4,
    darkness=const(NIGHT_DARKNESS),
    layers=[animal("animal_bird.png",
                   Track([(0.0, BIRD.x), (T4_DIP, BIRD.x - 15, "out"), (D4, BIRD_HIGH[0])]),
                   Track([(0.0, BIRD.y), (T4_DIP, BIRD.y + 28, "out"), (D4, BIRD_HIGH[1])]),
                   Track([(0.0, BIRD.width), (D4, BIRD_HIGH[2])]),
                   rotation=Track([(0.0, 0.0), (T4_DIP, 4.0, "out"), (1.3, -14.0),
                                   (D4 - 0.6, -10.0), (D4, -8.0)]),
                   scale_x=Track([(0.0, 1.0), (T4_DIP, 1.05, "out"), (0.9, 0.96, "out"),
                                  (1.5, 1.0)]),
                   scale_y=Track([(0.0, 1.0), (T4_DIP, 0.94, "out"), (0.9, 1.06, "out"),
                                  (1.5, 1.0)]),
                   wiggles=flutter(phase=D3), flap=BIRD_FLAP)],
    narration=N4,
)

N5 = "하지만 달님은 보이지 않았답니다."
D5 = narration_length(N5) + TRANSITION + 1.0
T5_LOOK = 1.2   # 높은 곳에서 잠시 두리번거리다가 멀리 날아가 버린다
T5_GONE = D5 - TRANSITION
CUT_5 = Cut(
    number=5, title="달을 찾지 못한 새", background="bg_02.png", duration=D5,
    darkness=const(NIGHT_DARKNESS),
    layers=[animal("animal_bird.png",
                   Track([(0.0, BIRD_HIGH[0]), (T5_LOOK, BIRD_HIGH[0] - 10), (T5_GONE, 1600, "in")]),
                   Track([(0.0, BIRD_HIGH[1]), (T5_LOOK, BIRD_HIGH[1] + 15), (T5_GONE, -40, "in")]),
                   Track([(0.0, BIRD_HIGH[2]), (T5_LOOK, BIRD_HIGH[2]), (T5_GONE, 150, "in")]),
                   Track([(0.0, 1.0), (T5_GONE - 1.0, 1.0), (T5_GONE, 0.0)]),
                   rotation=Track([(0.0, -8.0), (T5_LOOK, -2.0), (T5_GONE, -20.0, "in")]),
                   wiggles=flutter(phase=D3 + D4), flap=BIRD_FLAP)],
    next_background="bg_03_hidden.png", next_mix=fade_to_next(D5),
    narration=N5,
)

# ---------------------------------------------------------------- 장면 3: 토끼의 도전 (bg_03_hidden)
RABBIT_AT = (650, 690, 380)        # 뛰는 그림(animal_rabbit.png) 기준 자리 (x, y, 폭)
RABBIT_LAND = "animal_rabbit_land.png"
# 그림자는 다리를 오므린 착지 자세로 완성된다. 자리는 컷 7에서 이어받는 착지 자세와 같다
RABBIT = ShadowSpec(animal="rabbit", art=RABBIT_LAND, **pose_place(*RABBIT_AT, RABBIT_LAND))
RABBIT_TOP = (1330, 450, 300)      # 언덕 꼭대기
RABBIT_BEYOND = (1700, 520, 240)   # 언덕 너머

N6 = "이번에는 토끼가 나섰어요.\n토끼는 폴짝폴짝 뛰며 언덕 위로 올라갔어요."
D6 = RABBIT.done_at + narration_length(N6)
RABBIT_BREATH = dict(amp=0.025, period=1.6)   # 토끼는 숨이 빠르다
# 뛰는 그림과 다리를 오므린 착지 그림 (착지 그림 자리는 tools/align_pose.py)
RABBIT_POSES = dict(art="animal_rabbit.png", land=RABBIT_LAND)
CUT_6 = Cut(
    number=6, title="토끼 등장", background="bg_03_hidden.png",
    duration=D6, darkness=const(NIGHT_DARKNESS),
    shadow=RABBIT, narration_from=RABBIT.done_at, narration=N6,
    layers=[settled(RABBIT, breathing(start=RABBIT.done_at, ramp_in=0.8, **RABBIT_BREATH),
                    pivot=FEET)],
)

N7 = "\"조금만 더 올라가면 보이겠지?\""
D7 = max(5.0, narration_length(N7) + 1.0)
# 컷 7에서 토끼 그림자를 유지하는 동안: 꼭대기에서 제자리 깡충 뛰어 멀리 내다보고,
# 고개를 들어 하늘을 살피며 숨을 고른다 (되풀이 처음과 끝은 컷 7 끝의 착지 자세와 같다)
RABBIT_PEEK_LOOP = 3.2
RABBIT_PEEK = animal(
    "animal_rabbit.png",
    **hops(0.3, 1.0, RABBIT_TOP, RABBIT_TOP, 1, 38, landed=True, **RABBIT_POSES),
    wiggles=([Wiggle("rotation", -3.0, 3.2, -1.6, start=1.6, end=3.2)]      # 고개 들어 살피기
             + breathing(start=1.6, end=3.2, **RABBIT_BREATH)))
CUT_7 = Cut(
    number=7, title="언덕 위 토끼", background="bg_03_hidden.png", duration=D7,
    darkness=const(NIGHT_DARKNESS),
    layers=[animal("animal_rabbit.png",
                   **hops(0.3, 4.3, RABBIT_AT, RABBIT_TOP, 4, 70, landed=True, **RABBIT_POSES),
                   # 컷 6의 숨 쉬기가 첫 도약 전에 잦아든다
                   wiggles=breathing(phase=D6, end=0.3, ramp_out=0.3, **RABBIT_BREATH))],
    narration=N7,
    # "조금만 더 올라가면 보이겠지?" 뒤에도 토끼 그림자를 유지하면 꼭대기에서 머문다
    shadow_hold=ShadowHold("rabbit", at=D7, loop=RABBIT_PEEK_LOOP, layers=[RABBIT_PEEK]),
)

N8 = "토끼는 더 높은 곳을 찾아 언덕 너머로 사라졌어요.\n\n하지만 달님은 여전히 보이지 않았답니다."
D8 = narration_length(N8) + TRANSITION
CUT_8 = Cut(
    number=8, title="사라지는 토끼", background="bg_03_hidden.png", duration=D8,
    darkness=const(NIGHT_DARKNESS),
    layers=[animal("animal_rabbit.png", **hops(0.3, 3.3, RABBIT_TOP, RABBIT_BEYOND, 2, 60,
                                               landed=True, **RABBIT_POSES),
                   opacity=Track([(0.0, 1.0), (2.6, 1.0), (3.4, 0.0)]))],
    next_background="bg_04.png", next_mix=fade_to_next(D8),
    narration=N8,
)

# ---------------------------------------------------------------- 장면 4: 여우의 도전 (bg_04 → bg_05)
FOX = ShadowSpec(animal="fox", art="animal_fox.png", x=960, y=720, width=420)
FOX_PIVOT = (-0.148, 0.45)   # 옆모습 여우의 발 가운데. 좌우 뒤집기·회전 축 (서 있는 자리 그대로)

N9 = "그다음은 똑똑한 여우였어요.\n여우는 숲속을 살펴보며 말했지요."
D9 = FOX.done_at + narration_length(N9)


def fox_idle(phase: float = 0.0, **when) -> list[Wiggle]:
    """여우가 서서 숨 쉬며 몸을 천천히 기울여 주위를 살핀다."""
    return (breathing(phase, amp=0.015, period=2.4, **when)
            + [Wiggle("rotation", 1.2, 3.6, phase, **when)])


CUT_9 = Cut(
    number=9, title="여우 등장", background="bg_04.png",
    duration=D9, darkness=const(NIGHT_DARKNESS),
    shadow=FOX, narration_from=FOX.done_at, narration=N9,
    layers=[settled(FOX, fox_idle(start=FOX.done_at, ramp_in=1.0), pivot=FOX_PIVOT)],
)

N10 = "\"나무 사이에 숨어 있는 건 아닐까?\""
# 여우 자세 그림 (tools/make_variants.py, 자리는 tools/align_pose.py가 발 가운데로 맞춘다)
# animal_fox.png : 몸은 왼쪽, 고개는 오른쪽을 돌아봄 (그림자로 완성되는 기본 모습)
# FOX_RIGHT      : 오른쪽으로 걷는 옆모습 (고개도 앞). 좌우를 뒤집으면 왼쪽으로 걷는 모습
# FOX_MID        : 돌아서는 중간. 몸은 관객 쪽으로 비스듬히, 고개는 왼쪽.
#                  좌우를 뒤집으면 고개가 오른쪽 -> 오른쪽으로 돌아설 때의 중간 모습
FOX_SIDE = "animal_fox.png"
FOX_RIGHT = "animal_fox_right.png"
FOX_MID = "animal_fox_mid.png"
FOX_BACK = "animal_fox_back.png"   # 숲 깊은 곳으로 걸어 들어가는 뒷모습
TURN_TO_RIGHT = [(FOX_MID, -1.0), (FOX_RIGHT, 1.0)]   # (그림, 좌우 방향) 차례
TURN_TO_LEFT = [(FOX_MID, 1.0), (FOX_RIGHT, -1.0)]
TURN_LEN = 0.5      # 웅크렸다 폴짝 뛰며 돌아서는 시간
TURN_HOP = 22       # 돌아설 때 뛰어오르는 높이 (px)


def turn(t: float, frm: float, seq: list, length: float = TURN_LEN) -> dict:
    """t초부터 웅크렸다 폴짝 뛰어오르며 돌아선다. frm = 지금 좌우 방향(scale_x 부호),
    seq = [(중간 그림, 방향), (마지막 그림, 방향)]. 뛰어오를 때 중간 그림, 땅에 닿기 직전 마지막
    그림으로 바뀐다 (좌우 뒤집기도 그때 함께). 그림이 바뀌는 순간은 몸이 늘어나고 줄어드는
    동작에 묻힌다. 돌려주는 값: scale_x·scale_y 키, 자세 그림, 뛰어오름(y 흔들림)."""
    (mid, s1), (last, s2) = seq
    crouch, launch, swap, land, settle = (t + length * k for k in (0.2, 0.28, 0.6, 0.72, 1.0))
    sx = [(t, frm), (crouch, frm * 1.06, "out"), (launch, frm * 0.96), (launch, s1 * 0.96),
          (swap, s1 * 0.98), (swap, s2 * 0.98), (land, s2 * 1.07, "in"), (settle, s2, "back")]
    sy = [(t, 1.0), (crouch, 0.92, "out"), (launch, 1.07, "out"), (swap, 1.02),
          (land, 0.92, "in"), (settle, 1.0, "back")]
    air = land - launch
    hop = Wiggle("y", -TURN_HOP, air * 2, -launch, "bounce", launch, land)
    return dict(sx=sx, sy=sy, poses=[(launch, mid), (swap, last)], wiggles=[hop])


def turns(*specs: tuple) -> dict:
    """turn을 여러 번: specs = (t, frm, seq) 들. animal()에 넘길 scale_x, scale_y, poses와
    뛰어오름 흔들림(wiggles에 더해 쓴다: 돌려준 dict의 "hop")."""
    sx, sy, poses, hop = [], [], [], []
    for spec in specs:
        r = turn(*spec)
        sx += r["sx"]
        sy += r["sy"]
        poses += r["poses"]
        hop += r["wiggles"]
    return dict(scale_x=Track(sx), scale_y=Track(sy), poses=poses, hop=hop)


# 그림자로 완성된 여우는 몸이 왼쪽, 고개는 오른쪽을 돌아본다. 그쪽으로 폴짝 돌아서서 걸어가
# → 멈춰서 고개 숙여 냄새를 맡고 → 다시 왼쪽으로 폴짝 돌아서 → 조금 되돌아온다
T10_TURN1, T10_GO, T10_STOP, T10_TURN, T10_BACK, T10_END = 0.2, 0.75, 2.5, 3.45, 4.0, 5.3
T10_HOLD = T10_END + 0.4   # 되돌아와 멈춰 선 때. 여우 그림자를 유지하면 여기서 두리번거린다
# 여우는 T10_HOLD 뒤에 사라지기 시작한다 (D10 - TRANSITION - 0.8)
D10 = max(6.0, narration_length(N10) + 1.5, T10_HOLD + 0.8) + TRANSITION
FOX_TURNS_10 = turns((T10_TURN1, 1.0, TURN_TO_RIGHT), (T10_TURN, 1.0, TURN_TO_LEFT))
# 컷 10에서 여우 그림자를 유지하는 동안의 두리번거림. 컷 10 끝 모습(왼쪽을 보고 선 옆모습,
# FOX_RIGHT 좌우 뒤집기)에서: 관객 쪽을 봤다가 → 어깨 너머 오른쪽(뒤)을 돌아보고 → 다시 관객 쪽 →
# 앞(왼쪽)을 보며 고개를 들어 나무 위를 살핀다. 고개를 돌릴 때마다 몸을 살짝 움츠려 그림이 바뀌는
# 순간을 감춘다. FOX_LOOK_EXITS(앞을 보고 선 때)에서 그림자를 풀면 바로 이어서 숲으로 들어간다.
FOX_FRONT = "animal_fox_turn1.png"   # 몸은 왼쪽, 고개는 관객 쪽


def glance(t: float, frm: float, art: str, to: float) -> dict:
    """t초에 고개를 돌린다: 살짝 움츠렸다 펴며 자세 그림을 art(좌우 방향 to)로 바꾼다."""
    a, b = t - 0.12, t + 0.2
    return dict(sx=[(a, frm), (t, frm * 1.02, "out"), (t, to * 1.02), (b, to, "back")],
                sy=[(a, 1.0), (t, 0.95, "out"), (b, 1.0, "back")], poses=[(t, art)])


FOX_LOOK_LOOP = 4.2
FOX_LOOK_EXITS = (2.4,)
_glances = [glance(0.4, -1.0, FOX_FRONT, 1.0), glance(0.75, 1.0, FOX_SIDE, 1.0),
            glance(1.75, 1.0, FOX_FRONT, 1.0), glance(2.1, 1.0, FOX_RIGHT, -1.0)]
FOX_LOOK = animal(
    FOX_SIDE, 1060, 715, 400,
    scale_x=Track([(0.0, -1.0)] + [k for g in _glances for k in g["sx"]]),
    scale_y=Track([(0.0, 1.0)] + [k for g in _glances for k in g["sy"]]),
    poses=[(-math.inf, FOX_RIGHT)] + [k for g in _glances for k in g["poses"]],
    # 뒤를 돌아볼 때(오른쪽을 봄) -가 고개 듦, 앞을 볼 때(왼쪽을 봄) +가 고개 듦
    rotation=Track([(0.0, 0.0), (0.9, 0.0), (1.2, -2.5), (1.55, -2.5), (1.7, 0.0),
                    (2.5, 0.0), (3.0, 4.5), (3.5, 4.5), (4.1, 0.0)]),
    pivot=FOX_PIVOT)

CUT_10 = Cut(
    number=10, title="숲을 찾는 여우", background="bg_04.png", duration=D10,
    darkness=const(NIGHT_DARKNESS),
    layers=[animal("animal_fox.png",
                   Track([(0.0, FOX.x), (T10_GO, FOX.x), (T10_STOP, 1180), (T10_BACK, 1180),
                          (T10_END, 1060)]),
                   Track([(0.0, FOX.y), (T10_GO, FOX.y), (T10_STOP, 700), (T10_BACK, 700),
                          (T10_END, 715)]),
                   Track([(0.0, FOX.width), (T10_GO, FOX.width), (T10_STOP, 390), (T10_BACK, 390),
                          (T10_END, 400)]),
                   Track([(0.0, 1.0), (D10 - TRANSITION - 0.8, 1.0), (D10 - TRANSITION + 0.4, 0.0)]),
                   # 오른쪽을 볼 때 +는 고개를 숙이는 쪽
                   rotation=Track([(0.0, 0.0), (T10_STOP, 0.0), (T10_STOP + 0.45, 5.0, "out"),
                                   (T10_TURN - 0.4, 5.0), (T10_TURN, 0.0)]),
                   scale_x=FOX_TURNS_10["scale_x"], scale_y=FOX_TURNS_10["scale_y"],
                   poses=FOX_TURNS_10["poses"],
                   pivot=FOX_PIVOT,
                   wiggles=(fox_idle(D9, end=T10_TURN1, ramp_out=T10_TURN1) + FOX_TURNS_10["hop"]
                            + walk(T10_GO, T10_STOP) + walk(T10_BACK, T10_END)))],
    next_background="bg_05.png", next_mix=fade_to_next(D10),
    narration=N10,
    shadow_hold=ShadowHold("fox", at=T10_HOLD, loop=FOX_LOOK_LOOP, layers=[FOX_LOOK],
                           exits=FOX_LOOK_EXITS),
)

N11 = "여우는 이리저리 찾아보며\n숲 깊은 곳으로 들어갔어요.\n\n하지만 달님은 찾을 수 없었답니다."
D11 = narration_length(N11) + TRANSITION
# 컷 11: 뒷모습 여우가 꼬리를 흔들며 bg_05의 숲길을 따라 깊은 곳으로 걸어 들어간다.
# 길은 화면 아래 가운데에서 올라가며 왼쪽으로 살짝 휘었다가 다시 오른쪽으로 돌아 숲 안쪽(965, 806)에서
# 사라진다. 아래 점들은 그 길 위의 발 위치 (시각, x, 발 y). 크기는 지평선(y 790)에서 떨어진 만큼.
FOX_BACK_TAIL = "fox_back_tail"   # 뒷모습 꼬리 흔들기 프레임 (tools/tail_frames.py)
FOREST_HORIZON_Y = 790
FOX_PATH_11 = [(0.0, 1000, 1075), (1.7, 990, 960), (3.0, 945, 880), (4.2, 978, 832),
               (5.5, 965, 806)]
_BACK_ASPECT = 1313 / 1011        # 뒷모습 그림 세로/가로


def _path_tracks(path: list, scale: float = 1.05) -> tuple[Track, Track, Track]:
    """길 위 발 위치들 -> 레이어 (x, y, 폭) Track. 그림 아래 끝이 발에 오도록 놓는다."""
    xs, ys, ws = [], [], []
    for t, x, fy in path:
        w = scale * (fy - FOREST_HORIZON_Y)
        xs.append((t, x, "linear"))
        ys.append((t, fy - w * _BACK_ASPECT / 2, "linear"))
        ws.append((t, w, "linear"))
    return Track(xs), Track(ys), Track(ws)


_x11, _y11, _w11 = _path_tracks(FOX_PATH_11)
CUT_11 = Cut(
    number=11, title="깊은 숲속 여우", background="bg_05.png", duration=D11,
    darkness=const(NIGHT_DARKNESS),
    layers=[animal(FOX_BACK, _x11, _y11, _w11,
                   Track([(0.0, 0.0), (0.6, 1.0), (4.5, 1.0), (5.5, 0.0)]),
                   flap=FOX_BACK_TAIL,
                   pivot=(0.0, 0.5),
                   # 멀어질수록 발걸음 출렁임도 작아지고, 걸음마다 꼬리가 좌우로 흔들린다
                   wiggles=(walk(0.0, 5.5, bob=6.0, sway=1.2, ramp_out=4.0)
                            + [Wiggle("flap", 0.85, 0.84, 0.0, end=5.5)]))],
    next_background="bg_06_cloudy.png", next_mix=fade_to_next(D11),
    narration=N11,
)

# ---------------------------------------------------------------- 장면 5~8: 사슴의 기다림 (연못)
DEER = ShadowSpec(animal="deer", art="animal_deer.png", x=470, y=830, width=480)
DEER_PIVOT = (-0.25, 0.45)   # 앉은 사슴은 엉덩이 아래를 축으로 몸을 세운다


def deer_breath(phase: float = 0.0, **when) -> list[Wiggle]:
    """앉아서 깊고 느리게 숨 쉬는 사슴. 들이쉬면 가슴이 부풀고(세로로 늘고 가로로 살짝 좁아짐),
    조금 늦게 고개와 상체가 따라 들린다. 숨과 다른 느린 박자로 몸이 아주 약하게 흔들려서
    똑같은 반복으로 보이지 않는다. (pivot=DEER_PIVOT과 함께)"""
    beat = 3.4
    return [Wiggle("scale_y", 0.022, beat, phase, **when),
            Wiggle("scale_x", -0.006, beat, phase, **when),
            Wiggle("rotation", -0.9, beat, phase - 0.35, **when),   # -가 고개 듦
            Wiggle("rotation", 0.5, 8.3, phase, **when)]


DEER_SMILE = "animal_deer_laugh.png"   # 달이 뜬 뒤 입을 벌리고 눈을 감고 웃는 얼굴 (tools/make_variants.py)
DEER_SMILE_FADE = 0.12   # 웃는 얼굴로 바뀌는 시간. 길면 다문 입과 벌린 입이 겹쳐 보인다


def deer_still(phase: float, opacity=None, smile_at: float | None = None, **motion) -> Layer:
    """연못가에 앉아 숨 쉬는 사슴. phase = 컷 12부터 이 컷 전까지의 시간 (숨이 끊기지 않게).
    smile_at: 이 시각부터 웃는 얼굴로 서서히 바뀐다 (-math.inf면 처음부터 웃는 얼굴)."""
    poses = [] if smile_at is None else [(smile_at, DEER_SMILE)]
    return animal("animal_deer.png", DEER.x, DEER.y, DEER.width, opacity, pivot=DEER_PIVOT,
                  wiggles=deer_breath(phase), poses=poses, pose_fade=DEER_SMILE_FADE, **motion)


MOON_06 = (1325, 188)   # bg_06의 달 중심

N12 = "사슴도 달님을 보고 싶었어요."
D12 = DEER.done_at + narration_length(N12)
CUT_12 = Cut(
    number=12, title="연못가 사슴", background="bg_06_cloudy.png",
    duration=D12, darkness=const(NIGHT_DARKNESS),
    shadow=DEER, narration_from=DEER.done_at, narration=N12,
    layers=[settled(DEER, deer_breath(start=DEER.done_at, ramp_in=1.0), pivot=DEER_PIVOT)],
)

N13 = "그래서 조용히 연못가에 앉았어요.\n\"여기서 조금 기다려야겠다.\"\n\n사슴은 가만히 하늘을 바라보며 기다렸어요."
D13 = narration_length(N13) + 1.0
P13 = D12                         # 사슴 숨 쉬기가 이어지도록 컷마다 앞 컷까지의 시간을 쌓는다
CUT_13 = Cut(
    number=13, title="기다리는 사슴", background="bg_06_cloudy.png",
    duration=D13, darkness=const(NIGHT_DARKNESS),
    layers=[deer_still(P13)], narration=N13,
)

# 컷 14: 구름 뒤 달 자리에 달 모양 그림자 안내가 나타나 기다린다 (인식기 전에는 5번 키).
#   인식되면 → 바람 나레이션 시작. 구름 레이어가 달을 가린 채로 배경만 달 뜬 연못(bg_06)으로
#   바뀌고, "구름은 천천히 움직이더니" 부분에서 구름이 밀려나며 지나가는 순간 달이 드러난다.
MOON = ShadowSpec(animal="moon", art="", x=MOON_06[0], y=MOON_06[1], width=300)
CLOUD_06_Y, CLOUD_06_W = MOON_06[1] + 40, 1000
CLOUD_06_OFF_X = 2650
N14 = "그때였어요.\n살랑살랑 바람이 불어오기 시작했어요.\n\n구름은 천천히 움직이더니\n조금씩, 조금씩,\n밤하늘을 열어 주었답니다."
W14 = MOON.wait_at
T14_TALK = W14 + MOON.APPEAR + 0.2                      # 그림자 인식 직후 나레이션 시작
_p14 = [page_seconds(p) for p in narration_pages(N14)]
D14 = T14_TALK + sum(_p14)
T14_WIND = T14_TALK + _p14[0]                           # "구름은 천천히 움직이더니"가 시작되는 때
CUT_14 = Cut(
    number=14, title="바람이 불어오고", background="bg_06_cloudy.png", duration=D14,
    layers=[Layer(image="cloud_01.png",
                  x=Track([(0.0, MOON_06[0]), (T14_WIND, MOON_06[0]), (D14, CLOUD_06_OFF_X)]),
                  y=const(CLOUD_06_Y), width=const(CLOUD_06_W), tint=NIGHT_TINT,
                  opacity=Track([(0.0, 0.0), (1.0, 1.0)])),
            deer_still(P13 + D13)],
    shadow=MOON,
    # 구름이 움직이기 시작하면 배경이 달 뜬 연못으로 바뀐다. 구름이 달을 지나가기 전에
    # 바뀌기를 마쳐서, 구름이 걷히는 순간 달이 바로 보이고 수면의 달빛도 함께 살아난다
    next_background="bg_06.png", next_under_layers=True,
    next_mix=Track([(0.0, 0.0), (T14_WIND + 0.3, 0.0),
                    (T14_WIND + 0.42 * (D14 - T14_WIND), 1.0)]),
    darkness=const(NIGHT_DARKNESS),
    narration_from=T14_TALK, narration=N14,
)

# 컷 15: 드러난 달이 서서히 밝아진다
N15 = "와!\n\n구름 뒤에 숨어 있던 달님이\n환하게 모습을 드러냈어요.\n\n연못 위에도 동그란 달님이 반짝였지요."
D15 = 0.8 + narration_length(N15)
P15 = P13 + D13 + D14
CUT_15 = Cut(
    number=15, title="달의 등장", background="bg_06.png", duration=D15,
    # "와!" 하는 순간 사슴이 깜짝 놀라 몸을 세우며 달 쪽으로 고개를 든다
    # 달을 올려다보며 놀란 뒤 웃는 얼굴이 된다
    layers=[deer_still(P15, smile_at=1.6,
                       rotation=Track([(0.0, 0.0), (0.8, 0.0), (1.15, -2.5, "out"),
                                       (2.0, -1.5), (3.5, 0.0)]),
                       scale_y=Track([(0.0, 1.0), (0.8, 1.0), (1.15, 1.05, "out"),
                                      (1.9, 1.0, "back")]))],
    darkness=Track([(0.0, NIGHT_DARKNESS), (6.0, 0.0)]),
    narration_from=0.8, narration=N15,
)

N16 = ("달님을 가장 먼저 본 건\n가장 높이 날아간 새도 아니고,\n\n"
       "가장 멀리 간 토끼도 아니고,\n가장 열심히 찾은 여우도 아니었어요.\n\n"
       "조용히 기다릴 줄 알았던 사슴이었답니다.\n\n"
       "때로는 서두르기보다 잠시 기다리는 것이\n답을 찾는 가장 좋은 방법이 될 수 있어요.")
WARM = 0.35
D16 = 1.5 + narration_length(N16)
P16 = P15 + D15
CUT_16 = Cut(
    number=16, title="이야기의 마무리", background="bg_06.png",
    duration=D16,
    layers=[deer_still(P16, smile_at=-math.inf)],
    warmth=Track([(0.0, 0.0), (3.0, WARM)]),
    narration_from=1.5, narration=N16,
)

# 컷 17: "끝." → 달과 사슴만 남은 고요한 여운 → 사슴이 살며시 사라지고, 화면 가운데에
#        "동화를 들으면서 궁금한게 있었나요?"가 떠오른다. 여기서만 대화형 AI가 켜진다 (M 키로 질문).
#        자동으로 넘어가지 않고, Space를 누르면 메인 화면(컷 0)으로 돌아가 다음 공연을 기다린다.
T17_END_TEXT = 3.5             # "끝." 자막이 사라지는 때
T17_DEER_OUT = (4.5, 6.0)      # 사슴이 사라지는 구간
T17_ASK = T17_DEER_OUT[1] + 0.3   # 질문 안내가 떠오르기 시작하는 때 (1.2초 동안 진해진다)
CUT_17 = Cut(
    number=17, title="끝", background="bg_06.png", duration=T17_ASK + 1.2,
    layers=[deer_still(P16 + D16,
                       Track([(0.0, 1.0), (T17_DEER_OUT[0], 1.0), (T17_DEER_OUT[1], 0.0)]),
                       smile_at=-math.inf)],
    warmth=const(WARM),
    narration_from=0.5, narration_until=T17_END_TEXT, narration="끝.",
    ask_from=T17_ASK,
)

# ---------------------------------------------------------------- 메인 화면
CUT_0 = Cut(
    number=0, title="메인 화면", background="bg_title.png", duration=1.0,
    title_alpha=const(1.0), hold=True,
)

CUTS: list[Cut] = [CUT_0, CUT_1, CUT_2, CUT_3, CUT_4, CUT_5, CUT_6, CUT_7, CUT_8,
                   CUT_9, CUT_10, CUT_11, CUT_12, CUT_13, CUT_14, CUT_15, CUT_16, CUT_17]
