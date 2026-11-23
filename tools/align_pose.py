"""같은 동물의 다른 자세 그림을 기준 그림에 맞춰 놓을 자리를 계산한다.

자세마다 그림 크기와 여백이 달라서 그대로 바꿔 끼우면 동물이 커지거나 튄다.
- 배율·가로 위치: 두 그림에서 얼굴(HEAD 범위)이 같은 크기·같은 자리에 오도록 맞춘다
  (여러 배율로 템플릿 매칭).
  얼굴 범위가 "feet"이면 (돌아서는 모습처럼 얼굴로 맞출 수 없을 때) 키(귀 끝~발끝)를 같게
  하고, 땅에 닿은 발들의 가운데를 기준 그림의 발 가운데에 맞춘다. 몸을 돌려도 서 있는 자리가
  그대로다. 기준 그림의 발 가운데는 좌우 뒤집기 축으로도 쓴다 (cuts.py의 FOX_PIVOT).
- 세로 위치: 발끝(그림 아래 끝)을 기준 그림의 아래 끝에 맞춘다. 땅에 선 높이가 같아야
  착지할 때 떠 보이거나 파묻히지 않는다.

결과: layers/<자세>.json = {"base": 기준 그림, "size": 기준 그림 크기, "box": [x0, y0, x1, y1]}
box는 자세 그림이 덮을 범위를 기준 그림 픽셀 좌표로 나타낸 것. 앱은 이 범위에 그린다.

실행: .venv/Scripts/python.exe tools/align_pose.py [--preview 출력폴더]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LAYERS = ROOT / "assets" / "images" / "layers"

# 자세 그림: (기준 그림, 기준 그림에서 얼굴 범위 x0, y0, x1, y1)
POSES = {
    "animal_rabbit_land.png": ("animal_rabbit.png", (1180, 430, 1603, 860)),
    # 여우: 오른쪽으로 걷는 옆모습, 돌아서는 중간, 숲으로 들어가는 뒷모습 (서 있는 자리를 발로 맞춤)
    "animal_fox_right.png": ("animal_fox.png", "feet"),
    "animal_fox_mid.png": ("animal_fox.png", "feet"),
    "animal_fox_back.png": ("animal_fox.png", "feet"),
}


def flat_windows(target: np.ndarray, th: int, tw: int, min_std: float = 8.0) -> np.ndarray:
    """matchTemplate 결과와 같은 크기로, 창 안 밝기 변화가 거의 없는(투명한 빈 곳) 자리 표시.
    그런 곳은 정규화 상관값이 계산 오차로 1에 가깝게 나와서 후보에서 빼야 한다."""
    k = (tw, th)
    mean = cv2.boxFilter(target, -1, k, normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    sq = cv2.boxFilter(target * target, -1, k, normalize=True, anchor=(0, 0),
                       borderType=cv2.BORDER_CONSTANT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return std[:target.shape[0] - th + 1, :target.shape[1] - tw + 1] < min_std


def rgba(name: str) -> np.ndarray:
    return np.array(Image.open(LAYERS / name).convert("RGBA"))


def features(img: np.ndarray) -> np.ndarray:
    """배경(투명)은 검게 둔 밝기 그림. 템플릿 매칭용."""
    g = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g * (img[..., 3].astype(np.float32) / 255)


def feet_x(img: np.ndarray, band: float = 0.15) -> float:
    """땅에 닿은 발들의 가로 가운데 (그림 아래 band 비율 높이 안의 불투명 픽셀 평균 x). 뒷발이 앞발보다 위에 그려지는 옆모습도 포함되게 넉넉히 잡는다."""
    ys = np.nonzero(img[..., 3].max(axis=1) > 8)[0]
    y0 = int(ys.max() - (ys.max() - ys.min()) * band)
    xs = np.nonzero(img[y0:ys.max() + 1, :, 3] > 128)[1]
    return float(xs.mean())


def align(pose: str) -> dict:
    base, head = POSES[pose]
    b, p = rgba(base), rgba(pose)
    ph, pw = p.shape[:2]
    bh, bw = b.shape[:2]

    def bottom(img: np.ndarray) -> int:
        return int(np.nonzero(img[..., 3].max(axis=1) > 8)[0].max()) + 1

    def top(img: np.ndarray) -> int:
        return int(np.nonzero(img[..., 3].max(axis=1) > 8)[0].min())

    if head == "feet":
        s = (bottom(p) - top(p)) / (bottom(b) - top(b))
        x0 = feet_x(b) - feet_x(p) / s
        y1 = bottom(b)
        y0 = y1 - bottom(p) / s
        box = [round(x0, 1), round(y0, 1), round(x0 + pw / s, 1), round(y0 + ph / s, 1)]
        print(f"{pose}: 키·발 맞춤, 배율 {s:.2f}, box {box}  "
              f"(기준 그림 발 가운데 = 폭의 {feet_x(b) / bw - 0.5:+.3f})")
        return {"base": base, "size": [bw, bh], "box": box}
    hx0, hy0, hx1, hy1 = head
    templ = features(b)[hy0:hy1, hx0:hx1]
    target = features(p)
    best = (-1.0, 1.0, (0, 0))
    for s in np.arange(0.6, 1.6, 0.01):   # s = 자세 그림 픽셀 / 기준 그림 픽셀
        t = cv2.resize(templ, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= target.shape[0] or t.shape[1] >= target.shape[1]:
            break
        res = cv2.matchTemplate(target, t, cv2.TM_CCOEFF_NORMED)
        res[~np.isfinite(res) | flat_windows(target, *t.shape)] = -1   # 투명한 빈 곳은 빼고
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, float(s), loc)
    score, s, (mx, my) = best
    # 가로: 얼굴 왼쪽 끝이 같은 자리 -> 자세 그림 x=mx 가 기준 그림 x=hx0
    x0 = hx0 - mx / s
    # 세로: 아래 끝(발끝)을 맞춘다. 두 그림 모두 투명 여백 없이 잘려 있다
    y1 = bottom(b)
    y0 = y1 - bottom(p) / s
    box = [round(x0, 1), round(y0, 1), round(x0 + pw / s, 1), round(y0 + ph / s, 1)]
    print(f"{pose}: 얼굴 일치 {score:.3f}, 배율 {s:.2f}, box {box}")
    if score < 0.5:
        print("  !! 얼굴이 잘 맞지 않습니다. POSES의 얼굴 범위를 확인하세요")
    return {"base": base, "size": [bw, bh], "box": box}


def preview(pose: str, meta: dict, out_dir: Path) -> None:
    """기준 그림과 자세 그림을 반투명으로 겹쳐 본다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    b = Image.open(LAYERS / meta["base"]).convert("RGBA")
    p = Image.open(LAYERS / pose).convert("RGBA")
    x0, y0, x1, y1 = meta["box"]
    pad = 200
    canvas = Image.new("RGBA", (b.width + 2 * pad, b.height + 2 * pad), (50, 50, 60, 255))
    ghost = b.copy()
    ghost.putalpha(Image.fromarray((np.array(b)[..., 3] * 0.45).astype(np.uint8)))
    canvas.alpha_composite(ghost, (pad, pad))
    pr = p.resize((round(x1 - x0), round(y1 - y0)), Image.LANCZOS)
    arr = np.array(pr)
    arr[..., 3] = (arr[..., 3] * 0.75).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(arr), (round(x0) + pad, round(y0) + pad))
    canvas.resize((canvas.width // 2, canvas.height // 2)).save(out_dir / f"align_{Path(pose).stem}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path, help="겹쳐 본 그림을 저장할 폴더")
    args = ap.parse_args()
    for pose in POSES:
        meta = align(pose)
        (LAYERS / Path(pose).with_suffix(".json").name).write_text(
            json.dumps(meta, indent=1), encoding="utf-8")
        if args.preview:
            preview(pose, meta, args.preview)


if __name__ == "__main__":
    main()
