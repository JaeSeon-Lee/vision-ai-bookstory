"""그림자 영상에서 "따라 만들 손그림자" 안내 모양을 만든다 → assets/images/guides/<동물>.png

그림자를 기다리는 동안 앱이 동물 자리에 이 모양을 보여 준다. 인식기가 학습한 바로 그 손 모양이라
관객(아이)이 보고 따라 하면 인식도 잘 된다.

프레임 고르기: 그림자가 또렷한 프레임 중 학습된 모델이 그 동물이라고 가장 확신하는 것들에서,
그림자 크기가 영상의 중간값에 가장 가까운(= 가장 전형적인 자세) 프레임을 쓴다.

실행: .venv/Scripts/python.exe tools/make_guides.py [--light dark] [--preview 폴더]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from bookstory.recognizer import (PEAK_FLOOR, ShadowRecognizer, darkness, screen_region,  # noqa: E402
                                  shadow_crop, shadow_select, shape_features, soft_map)
from bookstory.shadow import round_cut_edges  # noqa: E402
from train_shadow import estimate_background, read_gray  # noqa: E402

OUT = ROOT / "assets" / "images" / "guides"
ANIMALS = ["bird", "rabbit", "fox", "deer", "moon"]
UPSCALE = 3   # 480폭에서 뽑은 마스크를 키워 매끈하게


def best_mask(path: Path, animal: str, rec: ShadowRecognizer) -> np.ndarray:
    frames = read_gray(path, 2)
    bg = estimate_background(frames)
    screen = screen_region(bg)
    cands = []   # (확률, 면적, 마스크)
    for g in frames:
        d, peak = darkness(g, bg, screen)
        if peak < PEAK_FLOOR:
            continue
        soft = soft_map(d, peak)
        sel = shadow_select(soft, 0.45)
        if sel is None:
            continue
        crop = shadow_crop(soft, sel=sel)
        p = rec.model.predict_proba(shape_features(crop)[None])[0]
        cands.append((p[rec.classes.index(animal)], int(sel.sum()), sel, soft))
    sure = [c for c in cands if c[0] >= 0.95] or sorted(cands, key=lambda c: -c[0])[:10]
    med = np.median([c[1] for c in sure])
    _, _, sel, soft = min(sure, key=lambda c: abs(c[1] - med))

    # 스크린 안쪽만 남기고(팔이 스크린 테두리에서 잘린 부분은 둥글게), 크게 키워 매끈하게
    ys, xs = np.nonzero(screen)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = np.where(sel, soft, 0)[y0:y1, x0:x1]
    m = cv2.resize(m, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(m, (0, 0), UPSCALE * 0.8)
    mask = ((m > 0.45) * 255).astype(np.uint8)
    # 손가락 사이 작은 구멍은 메우고, 동물 눈처럼 큰 구멍은 남긴다
    holes = cv2.bitwise_not(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60 * UPSCALE * UPSCALE:
            mask[labels == i] = 255
    return round_cut_edges(mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", default="dark", choices=["dark", "bright"], help="어느 조명 영상에서 뽑을지")
    ap.add_argument("--preview", type=Path, help="모아 보기 이미지를 저장할 폴더")
    args = ap.parse_args()
    rec = ShadowRecognizer.load()
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for a in ANIMALS:
        mask = best_mask(ROOT / "assets" / "video" / args.light / f"{a}.mp4", a, rec)
        cv2.imwrite(str(OUT / f"{a}.png"), mask)
        print(f"{a}: {OUT / f'{a}.png'}  {mask.shape[1]}x{mask.shape[0]}")
        ys, xs = np.nonzero(mask)
        t = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        s = 300 / max(t.shape)
        t = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)))
        tile = np.zeros((320, 320), np.uint8)
        tile[(320 - t.shape[0]) // 2:][:t.shape[0], (320 - t.shape[1]) // 2:][:, :t.shape[1]] = t
        tiles.append(tile)
    if args.preview:
        args.preview.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview / "guides.png"), np.hstack(tiles))


if __name__ == "__main__":
    main()
