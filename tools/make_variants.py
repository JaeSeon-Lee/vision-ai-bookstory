"""생성한 동물 자세·표정 그림을 원본 그림에 맞게 다듬는다 (generate_images.py로 만든 raw 그림).

- animal_fox_front : 정면 여우. 옆으로 뻗은 꼬리를 지우고, 코를 중심으로 왼쪽 절반을 거울처럼
                     붙여 완전한 좌우 대칭으로 만든다 (돌아설 때 좌우를 뒤집어도 티가 나지 않게).
                     색감은 옆모습 여우에 맞춘다.
- animal_deer_smile : 웃는 사슴. 생성 그림 전체를 쓰면 털 무늬가 미묘하게 달라 바뀌는 순간 티가
                     나므로, 얼굴로 위치를 맞춘 뒤 입 둘레만 원본 사슴에 옮겨 붙인다.

- animal_fox_turn1~3 : 옆모습 -> 정면 사이 돌아서는 모습. turn1은 생성 그림, turn2·3은 turn1의
                     가슴 뒤쪽을 가로로 줄여 만든다 (몸이 정면으로 돌아오며 뒤쪽이 짧아 보이는 원근).
- animal_deer_laugh : 입을 벌리고 눈을 감고 웃는 사슴. 벌어진 턱까지 윤곽째, 감은 눈과 함께
                     옮겨 붙인다.

- animal_fox_right/mid/back : 오른쪽으로 걷는 옆모습, 돌아서는 중간, 뒷모습. 배경을 지우고 색감을
                     옆모습에 맞춘다. (turn1~3·front는 예전 돌아서기 방식용으로, 지금은 쓰지 않는다)

실행: .venv/Scripts/python.exe tools/make_variants.py [fox] [deer] [laugh] [foxpose] [turns] [--preview 폴더]
그 뒤 tools/align_pose.py 로 자세 그림의 자리(json)를 다시 계산한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_images import LAYERS, RAW, remove_green

FOX_TAIL_CUT_X = 1660          # raw 정면 여우에서 이 x부터 오른쪽(옆으로 뻗은 꼬리)을 지운다
FOX_CHEST_X = 560             # turn1 그림에서 가슴·앞다리가 끝나는 x (이 뒤를 줄인다)
FOX_SQUEEZE = {2: 0.6, 3: 0.3}  # turn2·3에서 가슴 뒤쪽 가로 배율
DEER_HEAD = (1100, 150, 1507, 450)   # 원본 사슴 얼굴 범위 (위치 맞추기용)
# 원본 사슴에서 웃는 입으로 바꿀 영역 (입꼬리부터 입선 끝까지, 코는 피한다)
DEER_MOUTH = [(1345, 290), (1372, 276), (1440, 296), (1497, 272), (1507, 292), (1500, 330),
              (1420, 345), (1352, 326)]
DEER_FEATHER = 8                     # 입 둘레 경계를 부드럽게 하는 폭 (px)
# 웃는 사슴은 눈을 감고 있어 얼굴로는 잘 안 맞는다. 표정과 상관없는 뿔·귀로 위치를 맞춘다
DEER_LAUGH_REF = (820, 0, 1330, 420)
# 웃으며 감은 눈으로 바꿀 영역 (중심 x, y, 반지름 x, y, 기울기)
DEER_LAUGH_EYE = (1213, 268, 80, 54, -8)
# 입을 벌리고 웃는 얼굴로 바꿀 영역 (벌어진 아래턱까지 넉넉히, 코는 피한다)
DEER_LAUGH_MOUTH = [(1340, 290), (1372, 274), (1440, 292), (1497, 270), (1507, 285),
                    (1507, 380), (1440, 400), (1360, 370), (1335, 330)]


def flat_windows(target: np.ndarray, th: int, tw: int, min_std: float = 8.0) -> np.ndarray:
    """matchTemplate 결과와 같은 크기로, 창 안 밝기 변화가 거의 없는(투명한 빈 곳) 자리 표시.
    그런 곳은 정규화 상관값이 계산 오차로 1에 가깝게 나와서 후보에서 빼야 한다."""
    k = (tw, th)
    mean = cv2.boxFilter(target, -1, k, normalize=True, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
    sq = cv2.boxFilter(target * target, -1, k, normalize=True, anchor=(0, 0),
                       borderType=cv2.BORDER_CONSTANT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return std[:target.shape[0] - th + 1, :target.shape[1] - tw + 1] < min_std


def match_colors(img: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None,
                 ref_mask: np.ndarray | None = None) -> np.ndarray:
    """img의 색(LAB 평균·표준편차)을 ref에 맞춘다. 마스크가 있으면 그 부분만 보고 계산."""
    a = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(ref[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    ma = mask if mask is not None else img[..., 3] > 128
    mb = ref_mask if ref_mask is not None else ref[..., 3] > 128
    for c in range(3):
        sa, sb = a[..., c][ma], b[..., c][mb]
        a[..., c] = (a[..., c] - sa.mean()) * (sb.std() / max(sa.std(), 1e-3)) + sb.mean()
    out = img.copy()
    out[..., :3] = cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out


def fox_front() -> Image.Image:
    a = np.array(Image.open(RAW / "animal_fox_front.png").convert("RGBA"))
    a[:, FOX_TAIL_CUT_X:] = (0, 255, 0, 255)   # 초록으로 덮어 배경과 함께 지운다
    a = np.array(remove_green(Image.fromarray(a)))
    # 코(얼굴 높이에서 가장 어두운 덩어리) 가운데를 대칭축으로
    h = a.shape[0]
    y0, y1 = int(h * 0.30), int(h * 0.40)
    g = a[y0:y1, ..., :3].mean(axis=2)
    xs = np.nonzero((g < 50) & (a[y0:y1, ..., 3] > 200))[1]
    cx = int(round(xs.mean()))
    left = a[:, :cx + 1]
    sym = np.concatenate([left, left[:, -2::-1]], axis=1)
    side = np.array(Image.open(LAYERS / "animal_fox.png").convert("RGBA"))
    sym = match_colors(sym, side)
    img = Image.fromarray(sym)
    return img.crop(img.getbbox())


def locate(templ: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    """templ(원본 일부)이 target 안에서 보이는 (배율, x, y, 일치도). 배율 = target / templ."""
    def feat(x: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(x[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
        return g * (x[..., 3].astype(np.float32) / 255)
    t0, tg = feat(templ), feat(target)
    best = (-1.0, 1.0, 0, 0)
    for s in np.arange(0.5, 2.0, 0.01):
        t = cv2.resize(t0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= tg.shape[0] or t.shape[1] >= tg.shape[1]:
            break
        res = cv2.matchTemplate(tg, t, cv2.TM_CCOEFF_NORMED)
        res[~np.isfinite(res) | flat_windows(tg, *t.shape)] = -1   # 투명한 빈 곳은 빼고
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, float(s), *loc)
    score, s, x, y = best
    return s, x, y, score


def fox_pose(name: str) -> Image.Image:
    """생성한 여우 자세 그림 (오른쪽 옆모습·돌아서는 중간·뒷모습): 배경을 지우고 색감을 옆모습에 맞춘다."""
    a = np.array(remove_green(Image.open(RAW / f"{name}.png")))
    side = np.array(Image.open(LAYERS / "animal_fox.png").convert("RGBA"))
    img = Image.fromarray(match_colors(a, side))
    return img.crop(img.getbbox())


def fox_turn(i: int) -> Image.Image:
    """돌아서는 중간 모습. turn1은 생성 그림(배경 제거, 색감을 옆모습에 맞춤).
    turn2·3은 turn1에서 가슴 뒤쪽(몸통·뒷다리·꼬리)을 가로로 줄여 만든다: 몸이 정면으로
    돌아올수록 뒤쪽이 짧아 보이는 원근. 같은 그림에서 나와 털 무늬가 그대로라 빠르게 넘겨도
    깜빡이지 않는다 (생성으로는 이 각도들이 잘 나오지 않았다)."""
    if i == 1:
        a = np.array(remove_green(Image.open(RAW / "animal_fox_turn1.png")))
        side = np.array(Image.open(LAYERS / "animal_fox.png").convert("RGBA"))
        img = Image.fromarray(match_colors(a, side))
        return img.crop(img.getbbox())
    return squeeze_behind(np.array(fox_turn(1)), FOX_CHEST_X, FOX_SQUEEZE[i])


def squeeze_behind(a: np.ndarray, cx: int, k: float, blend: int = 60) -> Image.Image:
    """x > cx 부분을 가로로 k배로 줄인다. cx 근처 blend px에 걸쳐 서서히 줄어들어 꺾이지 않는다."""
    h, w = a.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    t = np.clip((xs - (cx - blend / 2)) / blend, 0, 1)
    rate = 1 - (1 - k) * t * t * (3 - 2 * t)          # 원본 1px이 결과에서 차지하는 폭
    fwd = np.concatenate([[0], np.cumsum(rate)[:-1]])  # 원본 x -> 결과 x
    out_w = int(np.ceil(fwd[-1] + rate[-1]))
    src_x = np.interp(np.arange(out_w), fwd, xs).astype(np.float32)
    map_x = np.repeat(src_x[None, :], h, axis=0)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], out_w, axis=1)
    f = a.astype(np.float32)
    f[..., :3] *= f[..., 3:4] / 255
    o = cv2.remap(f, map_x, map_y, cv2.INTER_AREA if k < 0.5 else cv2.INTER_LINEAR)
    al = o[..., 3:4]
    o[..., :3] = np.where(al > 0, o[..., :3] * 255 / np.maximum(al, 1), 0)
    img = Image.fromarray(np.clip(o, 0, 255).astype(np.uint8))
    return img.crop(img.getbbox())


def deer_laugh() -> Image.Image:
    """입을 벌리고 웃는 사슴: 아래턱이 내려가 윤곽이 바뀌므로 입 둘레는 알파까지 옮겨 붙인다."""
    return deer_smile("animal_deer_laugh.png", DEER_LAUGH_MOUTH, with_alpha=True,
                      ref=DEER_LAUGH_REF, eye=DEER_LAUGH_EYE)


def deer_smile(raw: str = "animal_deer_smile.png", region=None, with_alpha: bool = False,
               ref=None, eye=None) -> Image.Image:
    """원본 사슴에 생성 그림의 입(region 다각형)과 눈(eye 타원, 있으면)을 옮겨 붙인다."""
    region = region or DEER_MOUTH
    base = np.array(Image.open(LAYERS / "animal_deer.png").convert("RGBA"))
    gen = np.array(remove_green(Image.open(RAW / raw)))
    hx0, hy0, hx1, hy1 = ref or DEER_HEAD
    s, mx, my, score = locate(base[hy0:hy1, hx0:hx1], gen)
    print(f"deer: 얼굴 일치 {score:.3f}, 배율 {s:.3f}")
    # 생성 그림을 원본 좌표로 옮긴다: 원본 (hx0, hy0) <- 생성 (mx, my), 원본 1px = 생성 s px
    m = np.float32([[1 / s, 0, hx0 - mx / s], [0, 1 / s, hy0 - my / s]])
    h, w = base.shape[:2]
    warped = cv2.warpAffine(gen, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    # 입 둘레 타원 마스크 (부드러운 경계). 원본 실루엣 밖은 건드리지 않는다
    mask = np.zeros((h, w), np.float32)
    cv2.fillPoly(mask, [np.array(region, np.int32)], 1.0, cv2.LINE_AA)
    if eye:
        ex, ey, rx, ry, ang = eye
        cv2.ellipse(mask, (ex, ey), (rx, ry), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), DEER_FEATHER / 2)
    if not with_alpha:
        mask *= (base[..., 3] > 0) & (warped[..., 3] > 0)
    # 옮겨 올 부분의 색을 원본 같은 자리에 맞춘다 (입선처럼 어두운 곳은 빼고 털 색으로 계산)
    ring = ((mask > 0.05) & (base[..., :3].mean(axis=2) > 90)
            & (warped[..., :3].mean(axis=2) > 90))
    warped = match_colors(warped, base, ring, ring)
    out = base.astype(np.float32)
    k = mask[..., None]
    if with_alpha:
        # 색은 알파를 곱한 상태로 섞어야 윤곽이 바뀌는 곳(벌린 턱)에 테두리 색이 번지지 않는다
        pa, pb = out[..., 3:4] / 255, warped[..., 3:4].astype(np.float32) / 255
        # 윤곽은 또렷하게 바꾼다 (부드럽게 섞으면 턱 밑에 반투명한 번짐이 생긴다)
        ka = np.clip((k - 0.5) * 4 + 0.5, 0, 1)
        alpha = pa * (1 - ka) + pb * ka
        rgb = ((out[..., :3] * pa * (1 - k) + warped[..., :3] * pb * k)
               / np.maximum(pa * (1 - k) + pb * k, 1e-3))
        out[..., :3], out[..., 3:4] = rgb, alpha * 255
    else:
        out[..., :3] = out[..., :3] * (1 - k) + warped[..., :3].astype(np.float32) * k
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=["fox", "deer", "laugh", "foxpose"])
    ap.add_argument("--preview", type=Path)
    args = ap.parse_args()
    made = {}
    if "fox" in args.which:
        made["animal_fox_front.png"] = fox_front()
    if "deer" in args.which:
        made["animal_deer_smile.png"] = deer_smile()
    if "laugh" in args.which:
        made["animal_deer_laugh.png"] = deer_laugh()
    if "foxpose" in args.which:
        for n in ("animal_fox_right", "animal_fox_mid", "animal_fox_back"):
            made[f"{n}.png"] = fox_pose(n)
    for i in (1, 2, 3):
        if f"turn{i}" in args.which or "turns" in args.which:
            made[f"animal_fox_turn{i}.png"] = fox_turn(i)
    for name, img in made.items():
        img.save(LAYERS / name)
        print("saved", LAYERS / name, img.size)
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=True)
            bg = Image.new("RGBA", img.size, (50, 50, 60, 255))
            bg.alpha_composite(img)
            bg.save(args.preview / name)


if __name__ == "__main__":
    main()
