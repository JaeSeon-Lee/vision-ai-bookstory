"""동물 그림의 꼬리를 떼어 좌우로 흔드는 프레임을 만든다 (새 날갯짓과 같은 frames.json 형식).

- 꼬리(TAIL 다각형)를 떼어내고, 꼬리를 흔들 때 드러날 몸 쪽 자리(꼬리 안에서 몸과 FILL_BAND px
  이내)만 주변 털 색으로 메운다.
- 꼬리는 뿌리(ROOT) 근처는 고정하고 끝으로 갈수록 더 돌아가게 휘어 흔든다.
- 꼬리는 몸 앞에 그린다 (뒷모습에서 꼬리가 뒷다리보다 관객 쪽에 있다).
- 각도 0 = 원본 그림. 앱은 Layer.flap 폴더로 쓰고, "flap" 흔들림 +1/-1이 양 끝 각도다.

실행: .venv/Scripts/python.exe tools/tail_frames.py [--preview 폴더]
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

TAILS = {
    # 숲으로 걸어 들어가는 여우 뒷모습. 꼬리는 엉덩이에서 왼쪽 아래로 늘어진다
    "fox_back_tail": dict(
        src="animal_fox_back.png",
        tail=[(322, 660), (518, 645), (540, 780), (518, 900), (450, 990), (360, 1095),
              (255, 1200), (120, 1240), (0, 1240), (0, 1035), (120, 915), (225, 780)],
        root=(450, 690),       # 꼬리 뿌리 (회전 중심)
        bend=(60, 520),        # 뿌리에서 이 거리까지 회전량이 0 -> 다 돌아감 (px)
        angles=list(range(14, -15, -2)),   # + = 시계 방향 (꼬리 끝이 왼쪽 위로)
        fill_band=90,
        scale=0.5,
        pad=300,
    ),
}


def bend(rgba: np.ndarray, angle: float, pivot: tuple[float, float], r0: float, r1: float
         ) -> np.ndarray:
    """pivot 중심으로 휘게 돌린다 (시계 방향 +). r0 안쪽은 고정, r1부터 angle만큼 다 돌아간다."""
    if angle == 0:
        return rgba.copy()
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - pivot[0], yy - pivot[1]
    t = np.clip((np.hypot(dx, dy) - r0) / (r1 - r0), 0, 1)
    a = -np.radians(angle) * t * t * (3 - 2 * t)
    c, s = np.cos(a), np.sin(a)
    mx = (pivot[0] + c * dx - s * dy).astype(np.float32)
    my = (pivot[1] + s * dx + c * dy).astype(np.float32)
    f = rgba.astype(np.float32)
    f[..., :3] *= f[..., 3:4] / 255
    o = np.clip(cv2.remap(f, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT), 0, 255)
    al = o[..., 3:4]
    o[..., :3] = np.where(al > 0, o[..., :3] * 255 / np.maximum(al, 1), 0)
    return np.clip(o, 0, 255).astype(np.uint8)


def split_tail(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(꼬리 뗀 몸, 꼬리) 그림."""
    src = np.array(Image.open(LAYERS / cfg["src"]).convert("RGBA"))
    h, w = src.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly, [np.array(cfg["tail"], np.int32)], 1)
    tail_m = (poly > 0) & (src[..., 3] > 0)
    tail = np.zeros_like(src)
    tail[tail_m] = src[tail_m]
    # 꼬리 가장자리를 1~2px 부드럽게
    soft = cv2.GaussianBlur(tail_m.astype(np.float32), (0, 0), 1.0)
    tail[..., 3] = (src[..., 3] * soft).astype(np.uint8)

    body = src.copy()
    body_m = (src[..., 3] > 128) & ~tail_m
    # 꼬리를 떼고 남은 털 끝 조각(몸과 떨어진 작은 덩어리)은 버리고 가장 큰 덩어리만 몸으로 본다
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body_m.astype(np.uint8))
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    stray = (src[..., 3] > 0) & ~tail_m & (lab != biggest)
    stray = cv2.dilate(stray.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~(lab == biggest)
    body[stray] = 0
    body_m = lab == biggest
    # 꼬리 자리 중 몸 실루엣의 볼록 껍질 안(엉덩이와 뒷다리 사이 등 몸이 있을 자리)이면서
    # 몸과 가까운 띠만 메운다. 나머지는 원래 배경이므로 투명하게 둔다
    pts = cv2.findNonZero(body_m.astype(np.uint8))
    hull = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
    dist = cv2.distanceTransform((~body_m).astype(np.uint8), cv2.DIST_L2, 5)
    fill = tail_m & (hull > 0) & (dist <= cfg["fill_band"])
    # 색은 가장 가까운 몸 픽셀에서 가져온다 (투명 배경의 색이 섞이지 않게). 외곽선의 노르스름한
    # 테두리 빛은 빼고 몇 px 안쪽 털에서만 가져온다
    core = cv2.erode(body_m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=8) > 0
    _, labels = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    by, bx = np.nonzero(core)
    lut = np.zeros((labels.max() + 1, 2), np.int64)
    lut[1:len(by) + 1, 0], lut[1:len(by) + 1, 1] = by, bx
    # 몸 밖 모든 곳을 가장 가까운 몸 색으로 채운 뒤 크게 흐려서 메운다 (줄무늬가 생기지 않게)
    near = src[..., :3].copy()
    oy, ox = np.nonzero(~core)
    src_yx = lut[labels[oy, ox]]
    near[oy, ox] = src[src_yx[:, 0], src_yx[:, 1], :3]
    blur = cv2.GaussianBlur(near, (0, 0), 12)
    body[fill, :3] = blur[fill]
    body[tail_m & ~fill] = 0
    # 메운 자리의 바깥 끝(껍질 경계·띠 끝)은 부드럽게 사라지게
    edge = cv2.distanceTransform(fill.astype(np.uint8), cv2.DIST_L2, 5)
    body[fill, 3] = (255 * np.clip(edge[fill] / 8, 0, 1)).astype(np.uint8)
    body[fill & (dist <= 2), 3] = 255
    return body, tail


def make(name: str, cfg: dict, preview: Path | None = None) -> None:
    body, tail = split_tail(cfg)
    pad = cfg["pad"]
    h, w = body.shape[:2]
    padded = lambda a: np.pad(a, ((pad, pad), (pad, pad), (0, 0)))
    pb, pt = Image.fromarray(padded(body)), padded(tail)
    root = (cfg["root"][0] + pad, cfg["root"][1] + pad)
    big = []
    for ang in cfg["angles"]:
        img = pb.copy()
        img.alpha_composite(Image.fromarray(bend(pt, ang, root, *cfg["bend"])))
        big.append(img)
    ys, xs = np.nonzero(np.max([np.array(b)[..., 3] for b in big], axis=0) > 0)
    x0, y0 = max(int(xs.min()) - 4, 0), max(int(ys.min()) - 4, 0)
    x1, y1 = min(int(xs.max()) + 5, big[0].width), min(int(ys.max()) + 5, big[0].height)
    assert x0 > 0 and y0 > 0 and x1 < big[0].width and y1 < big[0].height, "pad를 늘리세요"
    s = cfg["scale"]
    out_w, out_h = round((x1 - x0) * s), round((y1 - y0) * s)
    folder = LAYERS / name
    folder.mkdir(exist_ok=True)
    for old in folder.glob("tail_*.png"):
        old.unlink()
    files = []
    for i, img in enumerate(big):
        f = f"tail_{i:02d}.png"
        img.crop((x0, y0, x1, y1)).resize((out_w, out_h), Image.LANCZOS).save(folder / f, optimize=True)
        files.append(f)
    box = [x0 - pad, y0 - pad, x1 - pad, y1 - pad]
    (folder / "frames.json").write_text(json.dumps(
        {"angles": cfg["angles"], "files": files, "size": [w, h], "box": box}, indent=1),
        encoding="utf-8")
    print(f"{name}: {len(files)} frames {out_w}x{out_h} box {box}")
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
        thumbs = []
        for i in (0, len(big) // 2, len(big) - 1):
            bg = Image.new("RGBA", big[i].size, (50, 50, 60, 255))
            bg.alpha_composite(big[i])
            thumbs.append(bg.crop((x0, y0, x1, y1)).resize((out_w // 2, out_h // 2)))
        sheet = Image.new("RGB", (sum(t.width for t in thumbs), thumbs[0].height))
        for k, t in enumerate(thumbs):
            sheet.paste(t.convert("RGB"), (k * t.width, 0))
        sheet.save(preview / f"{name}.png")
        Image.fromarray(body).save(preview / f"{name}_body.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=Path)
    args = ap.parse_args()
    for name, cfg in TAILS.items():
        make(name, cfg, args.preview)


if __name__ == "__main__":
    main()
