"""animal_bird.png를 날갯짓용 부위로 나누고, 날갯짓 프레임을 만든다.

1. 부위 나누기: bird_body.png / bird_wing_l.png / bird_wing_r.png
   - 세 장 모두 원본과 같은 캔버스 크기·위치를 유지한다 (겹치면 원본이 된다).
   - 날개 뿌리는 어깨 근처에서 몸 안쪽으로 이어 그린다 (날개는 몸 뒤에 그린다).
2. 날갯짓 프레임: bird_flap/flap_XX.png + frames.json
   - 날개를 어깨(SHOULDER_L/R) 기준으로 "휘게" 돌린다. 뿌리(BEND_R0 안쪽)는 고정이고
     BEND_R까지 회전량이 부드럽게 늘어난다. 통째로 돌리면 어깨 이음새가 벌어지기 때문.
   - 각도 0 = 원본 자세, + = 날개를 들어 올림, - = 내려침. 앱은 가장 가까운 각도의 프레임을 그린다.

실행: .venv/Scripts/python.exe tools/split_bird.py [--preview 출력폴더] [--from-parts]
  --from-parts : 부위 PNG를 다시 자르지 않고 layers/에 있는 것을 쓴다
                 (그림판 프로그램에서 부위 PNG를 손본 뒤 프레임만 다시 만들 때)
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
SRC = LAYERS / "animal_bird.png"

# 날개 영역 경계 (원본 1574x1333 픽셀 좌표). 바깥쪽은 투명한 곳을 지나 캔버스 끝까지 감싼다.
WING_L = [(0, 0), (765, 0), (765, 420), (778, 452), (812, 480), (815, 505), (790, 525),
          (747, 557), (700, 610), (677, 670), (663, 723), (653, 763), (648, 800), (0, 800)]
WING_R = [(1574, 0), (1200, 0), (1200, 330), (1062, 440), (1053, 477), (1050, 530),
          (1045, 590), (1033, 620), (1007, 670), (980, 723), (967, 763), (972, 785),
          (1574, 785)]

SHOULDER_L = (784, 468)
SHOULDER_R = (1052, 490)

ROOT_EXTEND = 45   # 날개 뿌리를 몸 안쪽으로 이어 그리는 폭 (px)
ROOT_RADIUS = 120  # 어깨에서 이 거리 안쪽만 뿌리를 이어 그린다 (멀면 회전할 때 크게 밀려 나온다)
ROOT_INSET = 22    # 몸 외곽선에서 이만큼 안쪽에만 이어 그린다 (돌아도 몸 밖으로 안 삐져나오게)
BODY_FEATHER = 3   # 몸통이 잘린 가장자리를 부드럽게 하는 폭 (px)

BEND_R0 = 60       # 어깨에서 이 거리까지는 날개가 움직이지 않는다 (px)
BEND_R = 420       # 이 거리부터는 날개가 각도만큼 다 돌아간다 (px)
FLAP_ANGLES = list(range(12, -37, -2))   # 날개 각도 (도). 0이 반드시 들어 있어야 한다
FRAME_SCALE = 0.55  # 프레임 그림 배율 (화면에서 새는 최대 560px 정도로 그려진다)
FRAME_PAD = 250     # 날개가 원본 캔버스 밖으로 나가도 잘리지 않게 사방에 두는 여백 (px)
FLAP_DIR = LAYERS / "bird_flap"


def polygon_mask(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    return m.astype(bool)


def extend_root(rgba: np.ndarray, wing: np.ndarray, alpha: np.ndarray,
                shoulder: tuple[int, int]) -> np.ndarray:
    """날개 그림을 만들고, 날개와 맞닿은 몸 안쪽 띠를 가장 가까운 날개 픽셀 색으로 채운다."""
    out = np.zeros_like(rgba)
    out[wing] = rgba[wing]
    near = cv2.dilate(wing.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=ROOT_EXTEND).astype(bool)
    body = (alpha > 0) & ~wing
    inner = cv2.erode(body.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=ROOT_INSET).astype(bool)
    yy, xx = np.mgrid[:alpha.shape[0], :alpha.shape[1]]
    close = (xx - shoulder[0]) ** 2 + (yy - shoulder[1]) ** 2 <= ROOT_RADIUS ** 2
    band = near & inner & close
    # 각 픽셀에서 가장 가까운 날개 픽셀 찾기 (날개 = 0인 곳)
    _, labels = cv2.distanceTransformWithLabels(
        (~wing).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(~((~wing).astype(bool)))   # 날개 픽셀 좌표, 라벨 순서대로
    lut_y = np.zeros(labels.max() + 1, np.int64)
    lut_x = np.zeros(labels.max() + 1, np.int64)
    # DIST_LABEL_PIXEL 라벨은 0인 픽셀을 행 우선 순서로 1부터 매긴다
    lut_y[1:len(ys) + 1] = ys
    lut_x[1:len(xs) + 1] = xs
    by, bx = np.nonzero(band)
    lab = labels[by, bx]
    out[by, bx, :3] = rgba[lut_y[lab], lut_x[lab], :3]
    out[by, bx, 3] = 255
    # 채운 띠의 색 경계가 각지지 않게 살짝 흐린다 (띠 안쪽만)
    blur = cv2.GaussianBlur(out[..., :3], (0, 0), 2.0)
    out[band, :3] = blur[band]
    return out


def split(src: Path = SRC) -> dict[str, np.ndarray]:
    rgba = np.array(Image.open(src).convert("RGBA"))
    alpha = rgba[..., 3]
    h, w = alpha.shape
    left = polygon_mask((h, w), WING_L) & (alpha > 0)
    right = polygon_mask((h, w), WING_R) & (alpha > 0)
    body = rgba.copy()
    cut = left | right
    body[cut] = 0
    # 날개와 잘린 가장자리를 살짝 부드럽게 (투명 배경과 맞닿은 원래 외곽선은 그대로)
    dist = cv2.distanceTransform((~cut).astype(np.uint8), cv2.DIST_L2, 5)
    ramp = np.clip(dist / BODY_FEATHER, 0, 1)
    body[..., 3] = (body[..., 3] * ramp).astype(np.uint8)
    return {
        "bird_body.png": body,
        "bird_wing_l.png": extend_root(rgba, left, alpha, SHOULDER_L),
        "bird_wing_r.png": extend_root(rgba, right, alpha, SHOULDER_R),
    }


def bend(rgba: np.ndarray, angle: float, pivot: tuple[int, int]) -> np.ndarray:
    """pivot을 중심으로 날개를 휘게 돌린다. angle도: 시계 방향 + (화면 좌표, Qt와 같음).
    어깨에서의 거리 r에 따라 회전량이 0(BEND_R0)에서 angle(BEND_R)까지 부드럽게 늘어난다."""
    if angle == 0:
        return rgba.copy()
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - pivot[0], yy - pivot[1]
    t = np.clip((np.hypot(dx, dy) - BEND_R0) / (BEND_R - BEND_R0), 0, 1)
    a = -np.radians(angle) * t * t * (3 - 2 * t)   # 출력 -> 입력: 반대로 돌린 자리에서 가져온다
    c, s = np.cos(a), np.sin(a)
    mx = (pivot[0] + c * dx - s * dy).astype(np.float32)
    my = (pivot[1] + s * dx + c * dy).astype(np.float32)
    # 알파를 곱한 상태로 옮겨야 투명한 가장자리의 색이 번지지 않는다
    f = rgba.astype(np.float32)
    f[..., :3] *= f[..., 3:4] / 255
    o = cv2.remap(f, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    o = np.clip(o, 0, 255)
    al = o[..., 3:4]
    o[..., :3] = np.where(al > 0, o[..., :3] * 255 / np.maximum(al, 1), 0)
    return np.clip(o, 0, 255).astype(np.uint8)


def pose(parts: dict[str, np.ndarray], angle: float, pad: int = 0) -> Image.Image:
    """날개 각도 angle의 새 한 장 (날개 -> 몸 순서로 겹친다). + = 들어 올림, - = 내려침.
    pad: 사방 여백. 그림의 원본 (0, 0)은 (pad, pad)에 놓인다."""
    def padded(name: str) -> np.ndarray:
        return np.pad(parts[name], ((pad, pad), (pad, pad), (0, 0)))
    sl = (SHOULDER_L[0] + pad, SHOULDER_L[1] + pad)
    sr = (SHOULDER_R[0] + pad, SHOULDER_R[1] + pad)
    body = Image.fromarray(padded("bird_body.png"))
    img = Image.new("RGBA", body.size, (0, 0, 0, 0))
    img.alpha_composite(Image.fromarray(bend(padded("bird_wing_l.png"), angle, sl)))
    img.alpha_composite(Image.fromarray(bend(padded("bird_wing_r.png"), -angle, sr)))
    img.alpha_composite(body)
    return img


def write_frames(parts: dict[str, np.ndarray]) -> list[Image.Image]:
    """프레임을 저장한다. 모든 자세를 감싸는 같은 범위로 잘라서, 프레임끼리 위치가 같다.
    frames.json의 box = 그 범위의 원본 픽셀 좌표 [x0, y0, x1, y1] (원본 캔버스 밖이면 음수 등),
    size = 원본 캔버스 크기. 앱은 원본 캔버스가 레이어 자리에 오도록 box 범위로 그린다."""
    assert 0 in FLAP_ANGLES, "각도 0(원본 자세) 프레임이 있어야 그림자 연출에서 자연스럽게 이어진다"
    FLAP_DIR.mkdir(parents=True, exist_ok=True)
    for old in FLAP_DIR.glob("flap_*.png"):
        old.unlink()
    h, w = parts["bird_body.png"].shape[:2]
    big = [pose(parts, ang, FRAME_PAD) for ang in FLAP_ANGLES]
    ys, xs = np.nonzero(np.max([np.array(b)[..., 3] for b in big], axis=0) > 0)
    x0, y0 = max(xs.min() - 4, 0), max(ys.min() - 4, 0)
    x1, y1 = min(xs.max() + 5, big[0].width), min(ys.max() + 5, big[0].height)
    for edge, v, lim in (("left", x0, 0), ("top", y0, 0),
                         ("right", x1, big[0].width), ("bottom", y1, big[0].height)):
        assert v != lim, f"날개가 여백({edge})까지 닿았다. FRAME_PAD를 늘리세요"
    out_w, out_h = round((x1 - x0) * FRAME_SCALE), round((y1 - y0) * FRAME_SCALE)
    frames, files = [], []
    for i, img in enumerate(big):
        img = img.crop((x0, y0, x1, y1)).resize((out_w, out_h), Image.LANCZOS)
        name = f"flap_{i:02d}.png"
        img.save(FLAP_DIR / name, optimize=True)
        frames.append(img)
        files.append(name)
    box = [int(x0 - FRAME_PAD), int(y0 - FRAME_PAD), int(x1 - FRAME_PAD), int(y1 - FRAME_PAD)]
    (FLAP_DIR / "frames.json").write_text(json.dumps(
        {"angles": FLAP_ANGLES, "files": files, "size": [w, h], "box": box}, indent=1),
        encoding="utf-8")
    print(f"saved {len(files)} frames ({out_w}x{out_h}, box {box}) ->", FLAP_DIR)
    return frames


def preview(parts: dict[str, np.ndarray], frames: list[Image.Image], out_dir: Path) -> None:
    """부위 색 구분 그림과 프레임 모음을 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = parts["bird_body.png"].shape[1::-1]
    tint = Image.new("RGBA", size, (60, 60, 60, 255))
    for name, col in (("bird_wing_l.png", (255, 80, 80)), ("bird_wing_r.png", (80, 160, 255)),
                      ("bird_body.png", None)):
        arr = parts[name].astype(np.float32)
        if col:
            arr[..., :3] = arr[..., :3] * 0.5 + np.array(col) * 0.5
        tint.alpha_composite(Image.fromarray(arr.astype(np.uint8)))
    tint.save(out_dir / "parts.png")

    cols = 6
    thumbs = []
    for f in frames:
        t = Image.new("RGBA", f.size, (60, 60, 60, 255))
        t.alpha_composite(f)
        thumbs.append(t.resize((f.width // 3, f.height // 3)))
    tw, th = thumbs[0].size
    sheet = Image.new("RGBA", (tw * cols, th * ((len(thumbs) + cols - 1) // cols)), (40, 40, 40, 255))
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
    sheet.save(out_dir / "frames.png")
    print("preview ->", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path, help="확인용 그림을 저장할 폴더")
    ap.add_argument("--from-parts", action="store_true", help="layers/의 부위 PNG를 그대로 쓴다")
    args = ap.parse_args()
    names = ("bird_body.png", "bird_wing_l.png", "bird_wing_r.png")
    if args.from_parts:
        parts = {n: np.array(Image.open(LAYERS / n).convert("RGBA")) for n in names}
    else:
        parts = split()
        for name, arr in parts.items():
            Image.fromarray(arr).save(LAYERS / name)
            print("saved", LAYERS / name)
    frames = write_frames(parts)
    if args.preview:
        preview(parts, frames, args.preview)


if __name__ == "__main__":
    main()
