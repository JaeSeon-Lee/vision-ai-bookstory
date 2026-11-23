"""웹캠 프레임에서 그림자 모양(마스크)을 뽑고, 정말 그림자인지 판정한다.

예전 방식은 "가장 진한 곳 = 1"로 맞춰서, 빈 스크린의 작은 잡음도 진한 그림자처럼 커졌다.
조명이 조금 기울거나 카메라가 몇 픽셀만 움직여도 빈 화면이 "달"로 나온 까닭이다. 여기서는
1. 조명 기울기(평면)를 빼고 배경과 비교한다.
2. 진하기의 중간값을 기준으로 윤곽을 자른다 (일부만 유난히 진해도 모양이 무너지지 않는다).
3. 진하기·면적·윤곽 선명도를 재서 그림자가 아닌 것은 분류기에 넘기지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .recognizer import hog, to_gray

MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "models" / "shadow_silhouette.joblib"
NONE = "none"       # 다섯 가지 그림자가 아닌 것
ROUND_YES = 0.82    # 이보다 둥글면 달
ROUND_NO = 0.70     # 달이라고 나와도 이보다 덜 둥글면 달이 아니다 (다른 그림자는 최대 0.64)
BG_RATE = 0.03      # 그림자로 인정하지 않은 화면(없음·아주 작음·윤곽 흐림)이면 배경을 이 비율로 맞춘다
MIN_LEVEL = 0.035   # 그림자가 배경보다 이만큼은 어두워야 한다
MIN_AREA = 0.04     # 스크린의 이 비율보다 작은 것은 잡음
MAX_AREA = 0.75     # 이보다 크면 조명이 꺼졌거나 카메라가 가려진 것
MIN_SHARP = 0.06    # 윤곽 선명도. 조명 변화는 윤곽이 흐리다
FAINT = 0.042       # 스크린에서 가장 어두운 1%도 이보다 옅으면 빈 화면


@dataclass
class Shadow:
    mask: np.ndarray | None = None   # 그림자 영역 0/255
    level: float = 0.0               # 진하기
    area: float = 0.0                # 스크린에서 차지하는 비율
    sharp: float = 0.0               # 윤곽 선명도

    @property
    def valid(self) -> bool:
        return (self.mask is not None and self.level >= MIN_LEVEL
                and MIN_AREA <= self.area <= MAX_AREA and self.sharp >= MIN_SHARP)

    @property
    def reason(self) -> str:
        """그림자로 인정하지 않은 까닭 (진단 화면용). 인정했으면 빈 글자."""
        if self.mask is None:
            return "그림자가 안 보임" if self.level < MIN_LEVEL else "또렷한 덩어리가 없음"
        if self.level < MIN_LEVEL:
            return "그림자가 너무 옅음"
        if self.area < MIN_AREA:
            return "그림자가 너무 작음"
        if self.area > MAX_AREA:
            return "화면 대부분이 어두움 (B로 배경 다시)"
        return "윤곽이 흐림" if self.sharp < MIN_SHARP else ""


def make_background(grays: list[np.ndarray]) -> np.ndarray:
    return cv2.GaussianBlur(np.median(np.stack(grays), axis=0).astype(np.float32), (0, 0), 2)


def screen_mask(bg: np.ndarray) -> np.ndarray:
    """가장 큰 밝은 덩어리(스크린)를 0/1로. 카메라가 조금 움직여도 테두리가 그림자로
    잡히지 않게 안쪽으로 깎는다 (화면 끝에 닿은 쪽은 테두리가 없으므로 깎지 않는다)."""
    _, s = cv2.threshold(bg.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(s)
    if n > 1:
        s = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    s = cv2.copyMakeBorder(s, 9, 9, 9, 9, cv2.BORDER_REPLICATE)
    return cv2.erode(s, np.ones((17, 17), np.uint8))[9:-9, 9:-9]


def darkness_flat(gray: np.ndarray, bg: np.ndarray, screen: np.ndarray) -> np.ndarray:
    """조명 기울기와 노출을 뺀, 배경 대비 어두운 정도(0~1)."""
    on = screen > 0
    r = np.log(np.maximum(cv2.GaussianBlur(gray, (0, 0), 1.5), 1) / np.maximum(bg, 1))
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = (xx / w - 0.5).astype(np.float32), (yy / h - 0.5).astype(np.float32)
    sub = on[::4, ::4]
    v = r[::4, ::4][sub]
    a = np.stack([np.ones_like(v), x[::4, ::4][sub], y[::4, ::4][sub]], axis=1)
    # 밝은 쪽에서 시작해, 빛을 받은 부분(평면 근처의 점)만으로 평면을 되풀이해 맞춘다
    c = np.array([np.percentile(v, 80), 0.0, 0.0])
    for _ in range(6):
        near = np.abs(v - a @ c) < 0.05
        if near.sum() < 100:
            break
        ridge = np.diag([1e-6, 0.02, 0.02]) * near.sum() * 0.1
        c = np.linalg.solve(a[near].T @ a[near] + ridge, a[near].T @ v[near])
    fit = c[0] + c[1] * x + c[2] * y
    return (np.clip(1 - np.exp(r - fit), 0, 1) * on).astype(np.float32)


def find_shadow(gray: np.ndarray, bg: np.ndarray, screen: np.ndarray) -> Shadow:
    on = screen > 0
    if not on.any():
        return Shadow()
    d = darkness_flat(gray, bg, screen)
    peak = float(np.percentile(d[on], 99))
    if peak < FAINT:
        return Shadow(level=peak)
    level = float(np.median(d[d > 0.4 * peak]))
    m = (d > max(0.45 * level, 0.02)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return Shadow(level=level)
    areas = stats[1:, cv2.CC_STAT_AREA]
    sel = np.isin(labels, 1 + np.nonzero(areas >= 0.15 * areas.max())[0])
    s = Shadow(level=float(np.median(d[sel])), area=float(sel.sum() / on.sum()))
    u = sel.astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(u, k) != cv2.erode(u, k)) & (cv2.erode(screen, np.ones((7, 7), np.uint8)) > 0)
    if edge.sum() > 30:
        g = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8
        s.sharp = float(np.median(g[edge])) / max(s.level, 1e-3)
    s.mask = clean_mask(u * 255)
    return s


def clean_mask(m: np.ndarray) -> np.ndarray:
    """밝은 곳 그림자의 거친 점들을 다듬는다."""
    return cv2.morphologyEx(cv2.medianBlur(m, 5), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def crop_to_screen(mask: np.ndarray, screen: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(screen)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def features(mask: np.ndarray) -> np.ndarray | None:
    """그림자 마스크 → 윤곽 방향 특징(HOG). 위치와 크기에 상관없게 정사각형에 맞춘다."""
    ys, xs = np.nonzero(mask > 127)
    if len(ys) < 60:
        return None
    p = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(p.shape) * 1.08) + 2
    c = np.zeros((side, side), np.uint8)
    oy, ox = (side - p.shape[0]) // 2, (side - p.shape[1]) // 2
    c[oy:oy + p.shape[0], ox:ox + p.shape[1]] = p
    return hog(cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA)).astype(np.float32)


def roundness(mask: np.ndarray) -> float:
    """가장 큰 덩어리가 얼마나 원에 가까운지(0~1). 달을 받치는 가는 막대는 지우고 잰다."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    if n <= 1:
        return 0.0
    u = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    u = cv2.copyMakeBorder(u, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    k = max(3, int(cv2.distanceTransform(u, cv2.DIST_L2, 5).max() * 0.5) | 1)
    o = cv2.morphologyEx(u, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    _, r = cv2.minEnclosingCircle(c)
    return float(cv2.contourArea(c) / (np.pi * r * r + 1e-6) * min(1.0, o.sum() / max(u.sum(), 1) / 0.85))


def classify(model, mask: np.ndarray) -> tuple[str, float]:
    """(이름 또는 "none", 확신). mask는 스크린 영역만 잘라 낸 것. 확신은 1등과 2등의 점수 차."""
    f = features(mask)
    if f is None:
        return NONE, 0.0
    rnd = roundness(mask)
    if rnd >= ROUND_YES:
        return "moon", rnd
    s = model.decision_function(f[None])[0]
    order = np.argsort(-s)
    name = str(model.classes_[order[0]])
    if name == "moon" and rnd < ROUND_NO:
        name = NONE
    return name, float(s[order[0]] - s[order[1]])


class SilhouetteRecognizer:
    """앱이 쓰는 인식기. 그림자가 없거나 아주 작거나 윤곽이 흐린 화면이면 배경을 조금씩 맞춰
    조명 변화에 적응한다 (흐린 그림자가 오래 머물면 배경에 섞일 수 있다)."""

    def __init__(self, model) -> None:
        self.model = model
        self.bg: np.ndarray | None = None
        self.screen: np.ndarray | None = None
        self.last = Shadow()      # 마지막 프레임에서 찾은 그림자 (진단용)
        self.last_name = ""       # 마지막 분류 결과, "none" 포함 (진단용)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SilhouetteRecognizer":
        import joblib
        return cls(joblib.load(path))

    def set_background(self, frames: list[np.ndarray]) -> None:
        """손을 치운 빈 스크린 프레임들로 배경을 정한다."""
        self.bg = make_background([to_gray(f) for f in frames])
        self.screen = screen_mask(self.bg)

    def predict(self, frame: np.ndarray) -> tuple[str | None, float, np.ndarray | None]:
        """(이름, 확신, 마스크). 그림자가 없거나 다섯 가지가 아니면 이름은 None."""
        if self.bg is None:
            raise RuntimeError("set_background()를 먼저 불러야 한다")
        gray = to_gray(frame)
        if gray.shape != self.bg.shape or not self.screen.any():
            return None, 0.0, None
        s = self.last = find_shadow(gray, self.bg, self.screen)
        self.last_name = ""
        if not s.valid:
            if s.mask is None or s.area <= 0.01 or (s.sharp < MIN_SHARP and s.area <= MAX_AREA):
                self.bg += BG_RATE * (cv2.GaussianBlur(gray, (0, 0), 2) - self.bg)
            return None, 0.0, None
        name, conf = classify(self.model, crop_to_screen(s.mask, self.screen))
        self.last_name = name
        return (None if name == NONE else name), conf, s.mask
