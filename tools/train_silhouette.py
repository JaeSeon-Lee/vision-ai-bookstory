"""그림자 윤곽 분류기(bookstory/silhouette.py)를 학습한다 → assets/models/shadow_silhouette.joblib

영상 프레임에 웹캠 교란을 넣고 앱과 같은 방식으로 그림자를 뽑는다. 다섯 가지 그림자에
"없음"(다섯 가지가 아닌 모양)을 더해 학습한다. 채점은 tools/silhouette_eval.py.

실행: .venv/Scripts/python.exe tools/train_silhouette.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed, dump

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import train_shadow as T  # noqa: E402
from bookstory.recognizer import PEAK_FLOOR, darkness, screen_region, shadow_crop, soft_map  # noqa: E402
from bookstory.silhouette import (MODEL_PATH, NONE, crop_to_screen, features, find_shadow,  # noqa: E402
                                  make_background, screen_mask)

TRAIN = [("clean", 0), ("mild", 1), ("hard", 2)]    # (교란 정도, 난수)
TEST = [("clean", 0), ("mild", 11), ("hard", 12)]
PER_CLASS, N_NONE = 1000, 1500


class Cond:
    """웹캠에서 생기는 교란: 잡음, 흐림, 노출, 조명 기울기, 카메라 이동."""

    def __init__(self, kind: str, seed: int) -> None:
        self.r = r = np.random.default_rng(seed)
        self.gain = r.uniform(0.88, 1.1) if kind != "clean" else 1.0
        self.angle = r.uniform(0, 6.28)
        self.shift = r.uniform(-3, 3, 2) * {"clean": 0, "mild": 0.5, "hard": 1}[kind]
        self.amp = {"clean": 0, "mild": 0.03, "hard": 0.07}[kind]
        self.sigma = {"clean": 0, "mild": 5, "hard": 8}[kind]
        self.blur = {"clean": 0, "mild": 0.8, "hard": 1.5}[kind]

    def noise(self, f: np.ndarray) -> np.ndarray:
        f = f.astype(np.float32)
        if self.blur:
            f = cv2.GaussianBlur(f, (0, 0), self.blur)
        if self.sigma:
            n = self.r.normal(0, self.sigma, (f.shape[0] // 2, f.shape[1] // 2)).astype(np.float32)
            f = f + cv2.resize(n, f.shape[::-1])
        return np.clip(f, 0, 255)

    def __call__(self, f: np.ndarray) -> np.ndarray:
        f = f.astype(np.float32)
        h, w = f.shape
        if self.amp:
            yy, xx = np.mgrid[0:h, 0:w]
            tilt = np.cos(self.angle) * (xx / w - 0.5) + np.sin(self.angle) * (yy / h - 0.5)
            f = f * (1 + 2 * self.amp * tilt) * self.gain
            f = cv2.warpAffine(f.astype(np.float32), np.float32([[1, 0, self.shift[0]], [0, 1, self.shift[1]]]),
                               (w, h), borderMode=cv2.BORDER_REPLICATE)
        return self.noise(f)


def aug(m: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """좌우 반전·회전·크기·기울기·팔 길이·윤곽 번짐을 무작위로 바꾼다."""
    if r.random() < 0.5:
        m = m[:, ::-1]
    m = cv2.copyMakeBorder(np.ascontiguousarray(m), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=0)
    h, w = m.shape
    a = np.eye(3)
    a[:2] = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-22, 22), r.uniform(0.75, 1.15))
    s = np.array([[r.uniform(0.88, 1.13), r.uniform(-0.15, 0.15), 0], [r.uniform(-0.1, 0.1), 1, 0], [0, 0, 1]])
    s[0, 2] = w / 2 - s[0, 0] * w / 2 - s[0, 1] * h / 2
    s[1, 2] = h / 2 - s[1, 0] * w / 2 - s[1, 1] * h / 2
    m = cv2.warpAffine(m, (s @ a)[:2], (w, h))
    ys, xs = np.nonzero(m > 127)
    if r.random() < 0.5 and len(ys) > 100:      # 팔이 들어오는 쪽을 조금 잘라 팔 길이를 바꾼다
        side = [(m[ys.min():ys.min() + 3] > 127).sum(), (m[ys.max() - 2:ys.max() + 1] > 127).sum(),
                (m[:, xs.min():xs.min() + 3] > 127).sum(), (m[:, xs.max() - 2:xs.max() + 1] > 127).sum()]
        ux, uy = [(0, -1), (0, 1), (-1, 0), (1, 0)][int(np.argmax(side))]
        t = r.uniform(-0.5, 0.5)
        ux, uy = ux * np.cos(t) - uy * np.sin(t), ux * np.sin(t) + uy * np.cos(t)
        along = (xs - xs.mean()) * ux + (ys - ys.mean()) * uy
        cut = along > np.percentile(along, r.uniform(85, 99))
        m[ys[cut], xs[cut]] = 0
    g = cv2.GaussianBlur(m, (0, 0), r.uniform(0.5, 4)).astype(np.float32)
    if r.random() < 0.3:                        # 거친 윤곽
        g += cv2.GaussianBlur(r.normal(0, 60, g.shape).astype(np.float32), (0, 0), 2) * 2
    return (g > r.uniform(80, 175)).astype(np.uint8) * 255


def noise_shape(r: np.random.Generator, masks: list[np.ndarray]) -> np.ndarray | None:
    """다섯 가지 그림자가 아닌 모양("없음" 학습용): 얼룩, 띠, 팔뚝, 주먹, 편 손, 다각형, 손의 일부."""
    w, h = 420, 250
    m = np.zeros((h, w), np.uint8)
    t = r.integers(0, 8)

    def ip(p) -> tuple[int, int]:
        return int(p[0]), int(p[1])

    if t == 0:
        g = cv2.GaussianBlur(r.normal(0, 1, (h, w)).astype(np.float32), (0, 0), r.uniform(6, 40))
        m = ((g > np.percentile(g, r.uniform(55, 90))) * 255).astype(np.uint8)
    elif t == 1:
        a = r.uniform(0, 6.28)
        yy, xx = np.mgrid[0:h, 0:w]
        v = np.cos(a) * (xx - w / 2) + np.sin(a) * (yy - h / 2)
        lo = r.uniform(-60, 120)
        m = ((v > lo) & (v < lo + r.uniform(30, 250))).astype(np.uint8) * 255
    elif t == 2:
        cv2.line(m, ip((r.uniform(0, w), r.choice([0, h]))), ip((r.uniform(0, w), r.uniform(0.2, 0.8) * h)),
                 255, int(r.uniform(15, 90)))
    elif t == 3:
        c, rad = ip((r.uniform(0.25, 0.75) * w, r.uniform(0.3, 0.6) * h)), r.uniform(30, 75)
        cv2.ellipse(m, c, ip((rad * r.uniform(0.75, 1), rad * r.uniform(1, 1.3))), r.uniform(0, 180),
                    0, 360, 255, -1)
        cv2.line(m, c, ip((c[0] + r.uniform(-150, 150), h)), 255, int(rad * r.uniform(0.7, 1.8)))
    elif t == 4:
        c, rad = np.array([r.uniform(0.3, 0.7) * w, r.uniform(0.45, 0.75) * h]), r.uniform(28, 50)
        cv2.circle(m, ip(c), int(rad), 255, -1)
        b, spread = r.uniform(-0.6, 0.6) - np.pi / 2, r.uniform(0.12, 0.42)
        for f in range(r.integers(0, 6)):
            an = b + (f - 2) * spread
            tip = c + rad * r.uniform(1.8, 2.8) * np.array([np.cos(an), np.sin(an)])
            cv2.line(m, ip(c), ip(tip), 255, int(rad * 0.42))
        elbow = c - rad * 8 * np.array([np.cos(b), np.sin(b)])
        cv2.line(m, ip(c), ip(elbow), 255, int(rad * r.uniform(1, 1.6)))
    elif t == 5:
        n = r.integers(3, 8)
        an = np.sort(r.uniform(0, 6.28, n))
        p = np.array([w / 2, h / 2]) + np.stack([np.cos(an), np.sin(an)], 1) * r.uniform(30, 120, (n, 1))
        cv2.fillPoly(m, [p.astype(np.int32)], 255)
    else:
        b = aug(masks[r.integers(len(masks))], r)
        ys, xs = np.nonzero(b > 127)
        m = np.zeros_like(b)
        if len(ys) > 200:
            a = r.uniform(0, 6.28)
            v = np.cos(a) * xs + np.sin(a) * ys
            keep = v < np.percentile(v, r.uniform(15, 40))
            m[ys[keep], xs[keep]] = 255
    m = cv2.GaussianBlur(np.ascontiguousarray(m), (0, 0), r.uniform(0.5, 4))
    m = (m > r.uniform(90, 160)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(labels, 1 + np.nonzero(areas >= 0.15 * areas.max())[0]).astype(np.uint8) * 255


def load(light: str, animal: str) -> list[dict]:
    cv2.setNumThreads(1)
    path = ROOT / "assets" / "video" / light / f"{animal}.mp4"
    v = T.process_video(path, 2)
    frames = T.read_gray(path, 2)
    state, take = v["label_state"], dict(zip(v["index"], v["take"]))
    empty = np.nonzero(state == 0)[0]
    if len(empty) < 8:
        empty = np.argsort(v["peaks"])[:8]
    picks = ([("pos", i) for i in v["index"][::3]] + [("empty", i) for i in empty[8:][::3]]
             + [("trans", i) for i in np.nonzero(state == -1)[0]])
    out = []
    for cond, seed in sorted(set(TRAIN + TEST)):
        c = Cond(cond, seed * 97 + ord(animal[0]) + (light == "dark") * 7)
        bg = make_background([c.noise(frames[i]) for i in empty[:8]])
        screen, old_screen = screen_mask(bg), screen_region(bg)
        for kind, i in picks:
            g = c(frames[i])
            s = find_shadow(g, bg, screen)
            d, peak = darkness(g, bg, old_screen)           # 기존 방식 (비교 채점용)
            near = v["index"][np.argmin(np.abs(v["index"] - i))]   # 시간상 가장 가까운 동작에 붙인다
            out.append({"kind": kind, "animal": animal, "light": light, "cond": cond, "seed": seed,
                        "take": f"{light}-{animal}-{take[near]}",
                        "old": shadow_crop(soft_map(d, peak)) if peak >= PEAK_FLOOR else None,
                        "mask": crop_to_screen(s.mask, screen) if s.valid else None})
    return out


def make_svm():
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), PCA(150, random_state=0),
                         SVC(C=10, gamma="scale", class_weight="balanced"))


def build_xy(train: list[dict], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    xs, ys = [], []
    for a in T.ANIMALS:
        masks = [d["mask"] for d in train if d["animal"] == a]
        for _ in range(PER_CLASS):
            f = features(aug(masks[r.integers(len(masks))], r))
            if f is not None:
                xs.append(f), ys.append(a)
    masks = [d["mask"] for d in train]
    while ys.count(NONE) < N_NONE:
        m = noise_shape(r, masks)
        f = None if m is None else features(m)
        if f is not None:
            xs.append(f), ys.append(NONE)
    return np.array(xs), np.array(ys)


def training_set(data: list[dict], skip: str | None = None) -> list[dict]:
    return [d for d in data if d["kind"] == "pos" and d["take"] != skip and d["mask"] is not None
            and (d["cond"], d["seed"]) in TRAIN]


def main() -> None:
    data = sum(Parallel(n_jobs=4)(delayed(load)(light, a) for light in T.LIGHTS for a in T.ANIMALS), [])
    x, y = build_xy(training_set(data))
    dump(make_svm().fit(x, y), MODEL_PATH)
    print(f"모델 저장: {MODEL_PATH}")


if __name__ == "__main__":
    main()
