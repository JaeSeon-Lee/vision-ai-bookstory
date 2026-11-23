"""그림자 모양 인식(머신러닝) 1차 방식.

앱은 새 인식기(silhouette.py)를 쓴다. 여기서는 hog와 to_gray만 앱에서 함께 쓰고,
나머지는 발표·비교 실험·안내 모양 만들기(tools/)용으로 남겨 두었다.

1. 그림자 지도: 빈 스크린(배경) 대비 얼마나 어두운지를 0~1로 나타낸다.
   밝은 방에서는 그림자가 배경보다 15% 정도만 어둡고, 어두운 방에서는 70~80% 어둡다.
   그래서 "가장 진한 부분 = 1"로 정규화해 두 조명에서 같은 모양으로 보이게 한다.
2. 모양 특징: 그림자 부분만 잘라 64x64 정사각형에 맞춘 뒤 HOG(윤곽 방향 분포) +
   Hu 모멘트(회전·크기에 강한 모양 수치)를 뽑는다.
3. 분류: tools/train_shadow.py가 학습한 SVM(assets/models/shadow_svm.joblib)으로 예측한다.

쓰는 법 (tools/make_guides.py):
    rec = ShadowRecognizer.load()
    rec.set_background([빈 스크린 프레임 여러 장])     # 시작할 때 한 번 (손을 치운 상태)
    name, prob = rec.predict(frame)                    # 그림자가 없거나 애매하면 name은 None
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

WORK_W = 480            # 처리 해상도(가로). 작게 줄여서 빠르고 잡음에 강하게
PEAK_FLOOR = 0.06       # 배경보다 이만큼도 안 어두우면 그림자가 없는 것으로 본다
SHAPE = 64              # 특징을 뽑을 정사각형 크기
MODEL_PATH = Path(__file__).resolve().parent.parent / "assets" / "models" / "shadow_svm.joblib"



def hog(img: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """HOG(윤곽 방향 히스토그램). OpenCV 5에는 HOGDescriptor가 없어서 직접 계산한다.
    8x8 칸마다 방향 9개 히스토그램, 2x2 칸 묶음마다 L2 정규화 (64x64 → 1764차원)."""
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180) / (180 / bins)   # 0 ~ bins
    lo = np.floor(ang).astype(int) % bins
    w_hi = ang - np.floor(ang)
    hi = (lo + 1) % bins
    ch, cw = f.shape[0] // cell, f.shape[1] // cell
    hist = np.zeros((ch, cw, bins), np.float32)
    cy = (np.arange(f.shape[0]) // cell)[:, None].repeat(f.shape[1], 1)
    cx = (np.arange(f.shape[1]) // cell)[None, :].repeat(f.shape[0], 0)
    np.add.at(hist, (cy, cx, lo), mag * (1 - w_hi))
    np.add.at(hist, (cy, cx, hi), mag * w_hi)
    blocks = np.concatenate([hist[:-1, :-1], hist[1:, :-1], hist[:-1, 1:], hist[1:, 1:]], axis=2)
    blocks /= np.sqrt((blocks ** 2).sum(axis=2, keepdims=True) + 1e-6)
    return np.minimum(blocks, 0.2).ravel()


def to_gray(frame: np.ndarray) -> np.ndarray:
    """컬러/흑백 프레임을 WORK_W 폭의 float32 흑백으로."""
    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = int(round(WORK_W * g.shape[0] / g.shape[1]))
    return cv2.resize(g, (WORK_W, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def screen_region(bg: np.ndarray) -> np.ndarray:
    """배경에서 밝은 스크린 영역(가장 큰 밝은 덩어리)을 0/1로. 테두리는 조금 깎는다."""
    _, scr = cv2.threshold(bg.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(scr)
    if n > 1:
        scr = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return cv2.erode(scr, np.ones((11, 11), np.uint8))


def darkness(gray: np.ndarray, bg: np.ndarray, screen: np.ndarray) -> tuple[np.ndarray, float]:
    """배경 대비 어두운 정도(0~1)와 그 최댓값(peak)을 돌려준다.

    카메라 자동 노출로 전체 밝기가 오르내리는 것은 스크린의 밝은 쪽(80번째 백분위) 값으로 나눠
    없앤다. 중앙값을 쓰면 그림자가 스크린 절반을 덮을 때(새) 그림자 자신이 기준이 되어 지워진다.
    """
    r = cv2.GaussianBlur(gray, (0, 0), 1.5) / np.maximum(bg, 1)
    on = screen > 0
    r /= max(float(np.percentile(r[on], 80)), 1e-3)
    d = np.clip(1 - r, 0, 1) * screen
    peak = float(np.percentile(d[on], 99.5))
    return d, peak


def soft_map(d: np.ndarray, peak: float) -> np.ndarray:
    """가장 진한 부분이 1이 되도록 정규화한 그림자 지도."""
    return np.clip(d / max(peak, PEAK_FLOOR), 0, 1).astype(np.float32)


def shadow_select(soft: np.ndarray, thr: float = 0.35) -> np.ndarray | None:
    """그림자 덩어리 영역(bool). 가장 큰 덩어리와, 그 15% 이상 크기의 덩어리
    (손가락 끝처럼 떨어져 보이는 부분)를 함께 쓴다. 너무 작으면 None."""
    m = (soft > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.nonzero(areas >= 0.15 * areas.max())[0]
    sel = np.isin(labels, keep)
    return sel if sel.sum() >= 50 else None


def shadow_crop(soft: np.ndarray, thr: float = 0.35,
                sel: np.ndarray | None = None) -> np.ndarray | None:
    """그림자 덩어리만 잘라 SHAPE x SHAPE 정사각형 가운데에 비율을 유지해 넣는다."""
    if sel is None:
        sel = shadow_select(soft, thr)
    if sel is None:
        return None
    ys, xs = np.nonzero(sel)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = np.where(sel, soft, 0)[y0:y1, x0:x1]
    side = int(max(patch.shape) * 1.08) + 2
    canvas = np.zeros((side, side), np.float32)
    oy, ox = (side - patch.shape[0]) // 2, (side - patch.shape[1]) // 2
    canvas[oy:oy + patch.shape[0], ox:ox + patch.shape[1]] = patch
    return cv2.resize(canvas, (SHAPE, SHAPE), interpolation=cv2.INTER_AREA)


def shape_features(crop: np.ndarray) -> np.ndarray:
    """64x64 그림자 조각 → 특징 벡터 (HOG 1764 + Hu 7 + 채움 비율 1)."""
    u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
    h = hog(u8)
    hu =cv2.HuMoments(cv2.moments((u8 > 90).astype(np.uint8), binaryImage=True)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    hu = np.clip(hu, -30, 30) / 10
    fill = np.array([(u8 > 90).mean()], np.float32)
    return np.concatenate([h, hu, fill]).astype(np.float32)


class ShadowRecognizer:
    """학습된 모델로 웹캠 프레임의 그림자 모양을 맞힌다."""

    def __init__(self, model) -> None:
        self.model = model                     # sklearn Pipeline (predict_proba 지원)
        self.classes = [str(c) for c in model.classes_]
        self.bg: np.ndarray | None = None
        self.screen: np.ndarray | None = None

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "ShadowRecognizer":
        import joblib
        return cls(joblib.load(path))

    def set_background(self, frames: list[np.ndarray]) -> None:
        """손을 치운 빈 스크린 프레임들로 배경을 정한다."""
        g = np.median(np.stack([to_gray(f) for f in frames]), axis=0)
        self.bg = cv2.GaussianBlur(g, (0, 0), 2)
        self.screen = screen_region(self.bg)

    def predict(self, frame: np.ndarray, min_prob: float = 0.6) -> tuple[str | None, float]:
        """(이름, 확률). 그림자가 없거나 확신이 min_prob보다 낮으면 이름은 None."""
        name, prob, _ = self.predict_detail(frame, min_prob)
        return name, prob

    def predict_detail(self, frame: np.ndarray, min_prob: float = 0.6
                       ) -> tuple[str | None, float, np.ndarray | None]:
        """(이름, 확률, 그림자 마스크 0/255). 그림자가 없으면 (None, 0, None).
        frame은 BGR 컬러 또는 흑백 (웹캠 RGB 프레임은 흑백으로 바꿔서 넘긴다)."""
        if self.bg is None:
            raise RuntimeError("set_background()를 먼저 불러야 한다")
        gray = to_gray(frame)
        if gray.shape != self.bg.shape:
            return None, 0.0, None
        d, peak = darkness(gray, self.bg, self.screen)
        if peak < PEAK_FLOOR:
            return None, 0.0, None
        soft = soft_map(d, peak)
        sel = shadow_select(soft)
        crop = shadow_crop(soft, sel=sel) if sel is not None else None
        if crop is None:
            return None, 0.0, None
        p = self.model.predict_proba(shape_features(crop)[None])[0]
        i = int(np.argmax(p))
        mask = (sel * 255).astype(np.uint8)
        return (self.classes[i] if p[i] >= min_prob else None), float(p[i]), mask
