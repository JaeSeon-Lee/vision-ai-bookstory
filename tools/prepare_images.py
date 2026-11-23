"""생성기에서 받은 이미지를 앱에서 쓸 수 있게 정리한다.

사용법 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe tools\\prepare_images.py

- assets/images/raw/bg_*.png|jpg|webp   -> assets/images/scenes/bg_*.png  (1920x1080, 16:9 중앙 크롭)
- assets/images/raw/cloud_*.png|jpg|webp -> assets/images/layers/cloud_*.png (초록 배경 제거, 투명 PNG)
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "images" / "raw"
SCENES = ROOT / "assets" / "images" / "scenes"
LAYERS = ROOT / "assets" / "images" / "layers"
TARGET = (1920, 1080)
EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def fit_16x9(img: Image.Image, v_anchor: float = 0.5) -> Image.Image:
    """16:9로 가운데를 잘라 1920x1080으로 맞춘다. v_anchor: 0=위, 0.5=가운데, 1=아래."""
    w, h = img.size
    target_ratio = TARGET[0] / TARGET[1]
    if w / h > target_ratio:  # 너무 넓음 -> 좌우 자르기
        nw = round(h * target_ratio)
        left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    else:  # 너무 높음 -> 위아래 자르기
        nh = round(w / target_ratio)
        top = round((h - nh) * v_anchor)
        img = img.crop((0, top, w, top + nh))
    return img.resize(TARGET, Image.LANCZOS)


def remove_green(img: Image.Image) -> Image.Image:
    """초록(#00FF00) 배경을 투명하게 만든다. 가장자리는 부드럽게, 초록 번짐은 제거한다."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # 초록이 다른 채널보다 얼마나 강한지 -> 클수록 배경
    greenness = g - np.maximum(r, b)
    alpha = 1.0 - np.clip((greenness - 4.0) / 60.0, 0.0, 1.0)
    # 구름 본체와 떨어진 조각(배경 얼룩, 잔상)은 버린다: 큰 덩어리만 남김
    solid = (alpha > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_ids = 1 + np.flatnonzero(areas >= areas.max() * 0.05)
        keep = np.isin(labels, keep_ids).astype(np.uint8)
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        alpha *= cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3)
    # 반투명 가장자리에 남은 초록 기운 제거 (despill)
    g_fixed = np.minimum(g, np.maximum(r, b))
    out = np.dstack([r, g_fixed, b, alpha * 255.0]).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(out, "RGBA")
    bbox = result.getbbox()  # 투명 여백 잘라내기
    return result.crop(bbox) if bbox else result


def main() -> None:
    SCENES.mkdir(parents=True, exist_ok=True)
    LAYERS.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in RAW.glob("*") if p.suffix.lower() in EXTS) if RAW.exists() else []
    if not files:
        print(f"처리할 이미지가 없습니다: {RAW}")
        return
    for src in files:
        img = Image.open(src)
        if src.stem.startswith(("cloud", "animal")):
            dst = LAYERS / f"{src.stem}.png"
            remove_green(img).save(dst)
        else:
            dst = SCENES / f"{src.stem}.png"
            fit_16x9(img.convert("RGB")).save(dst)
        print(f"{src.name} {img.size} -> {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
