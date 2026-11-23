"""그림자 모양 처리: 사진(나중에는 웹캠 프레임)에서 그림자 마스크를 뽑고,
그 모양을 동물 실루엣으로 서서히 바꾸는(모핑) 프레임을 만든다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_photo(path: Path) -> Image.Image:
    """휴대폰 사진의 회전 정보(EXIF)를 반영해 연다."""
    return ImageOps.exif_transpose(Image.open(path))


def mask_from_photo(path: Path, crop: tuple[int, int, int, int] | None = None,
                    ratio: float = 0.6) -> np.ndarray:
    """벽에 비친 손그림자 사진에서 그림자 영역을 0/255 마스크로 뽑는다.

    crop=(x, y, w, h)는 원본 픽셀 기준. 그림자 외의 물체(포스터·액자 등)를 빼는 데 쓴다.
    벽 밝기의 ratio배보다 어두운 부분 중 가장 큰 덩어리를 그림자로 본다.
    (뒤에서 비추는 스크린처럼 그림자가 옅으면 ratio를 높인다)
    """
    gray = np.asarray(load_photo(path).convert("L")).astype(np.float32)
    if crop:
        x, y, w, h = crop
        gray = gray[y:y + h, x:x + w]
    # 조명이 고르지 않아 벽도 구석은 어둡다. 작게 줄인 뒤 크게 팽창(최댓값)시켜 그림자를 지운
    # "벽만의 밝기"를 추정하고, 그 밝기의 60%보다 어두운 곳을 그림자로 본다
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    k = max(41, (max(small.shape) // 3) | 1)  # 그림자 폭보다 넓게
    wall = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    wall = cv2.GaussianBlur(wall, (0, 0), k / 3)
    wall = cv2.resize(wall, (w, h), interpolation=cv2.INTER_LINEAR)
    dark = (gray < wall * ratio).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if count <= 1:
        return dark * 255
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return round_cut_edges(((labels == biggest) * 255).astype(np.uint8))


def round_cut_edges(mask: np.ndarray) -> np.ndarray:
    """화면 가장자리에서 잘린 팔뚝은 직선으로 끊겨 보인다. 가장자리 띠를 걷어 내고
    둥글게 다듬어 손목이 자연스럽게 끝나도록 한다."""
    h, w = mask.shape
    band = max(8, min(h, w) // 25)
    on = mask > 0
    # 그림자가 실제로 닿아 잘린 가장자리만 다룬다 (손가락 끝 등 다른 곳은 건드리지 않음)
    sides = {"top": on[0].any(), "bottom": on[-1].any(), "left": on[:, 0].any(), "right": on[:, -1].any()}
    if not any(sides.values()):
        return mask
    m = mask.copy()
    near_edge = np.zeros_like(m)
    e = band * 4
    if sides["top"]:
        m[:band, :] = 0
        near_edge[:e, :] = 1
    if sides["bottom"]:
        m[-band:, :] = 0
        near_edge[-e:, :] = 1
    if sides["left"]:
        m[:, :band] = 0
        near_edge[:, :e] = 1
    if sides["right"]:
        m[:, -band:] = 0
        near_edge[:, -e:] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band * 4 + 1, band * 4 + 1))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    # 열림 연산은 가는 부분을 지우므로, 잘린 가장자리 근처에만 그 결과를 쓴다
    return np.where(near_edge > 0, opened, m).astype(np.uint8)


def mask_from_rgba(path: Path) -> np.ndarray:
    """투명 PNG(동물 그림)의 알파를 마스크로."""
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    return ((a > 127) * 255).astype(np.uint8)


def fit_mask(mask: np.ndarray, size: tuple[int, int], margin: float = 0.04) -> np.ndarray:
    """마스크의 그림자 부분만 잘라 size(w, h) 캔버스 가운데에 비율을 유지해 맞춘다."""
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    w, h = size
    scale = min(w * (1 - 2 * margin) / m.shape[1], h * (1 - 2 * margin) / m.shape[0])
    m = cv2.resize(m, (max(1, int(m.shape[1] * scale)), max(1, int(m.shape[0] * scale))),
                   interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w), np.uint8)
    ox, oy = (w - m.shape[1]) // 2, (h - m.shape[0]) // 2
    canvas[oy:oy + m.shape[0], ox:ox + m.shape[1]] = m
    return canvas


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """안쪽은 +, 바깥쪽은 − 인 거리 지도."""
    inside = (mask > 127).astype(np.uint8)
    d_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    return d_in - d_out


def morph_frames(src: np.ndarray, dst: np.ndarray, steps: int = 40,
                 feather: float = 2.5) -> list[np.ndarray]:
    """src 모양에서 dst 모양으로 서서히 바뀌는 알파(0~1) 프레임들.

    두 모양의 거리 지도를 섞어서, 중간 모양이 녹아내리듯 자연스럽게 이어지게 한다.
    src와 dst는 같은 크기여야 한다.
    """
    s_src, s_dst = signed_distance(src), signed_distance(dst)
    frames = []
    for i in range(steps + 1):
        a = i / steps
        a = a * a * (3 - 2 * a)  # 처음과 끝을 부드럽게
        sdf = (1 - a) * s_src + a * s_dst
        frames.append(np.clip(sdf / feather + 0.5, 0, 1).astype(np.float32))
    return frames
