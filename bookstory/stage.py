"""컷을 화면에 그리는 위젯과 렌더러."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QImage, QKeyEvent, QPainter,
                           QPainterPath, QPen)
from PySide6.QtWidgets import QWidget

from .camera import Camera
from .watcher import ShadowWatcher
from .narrator import Narrator
from .talk import Talk
from .shadow import fit_mask, morph_frames
from .timeline import Cut, ShadowSpec, ease_in_out, narration_page_at, narration_pages

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "assets" / "images" / "scenes"
LAYERS = ROOT / "assets" / "images" / "layers"
PHOTOS = ROOT / "assets" / "images"
FONTS = ROOT / "assets" / "fonts"   # narration.ttf를 넣으면 자막 글꼴로 우선 사용

STAGE_W, STAGE_H = 1920, 1080
NIGHT = QColor(14, 18, 48)

TITLE_TEXT = "기다림의 달"
TITLE_FONT = "Jua-Regular.ttf"   # assets/fonts/ 안의 제목 글꼴 (GowunBatang-Bold.ttf로 바꿔도 됨)
TITLE_POS = (720, 230)                # bg_title 왼쪽 위 빈 하늘
ASK_TEXT = "동화를 들으면서 궁금한 게 있었나요?"   # 이야기가 끝난 뒤 화면 가운데 안내
ASK_POS = (STAGE_W / 2, 400)          # 연못 위 밤하늘 (달과 겹치지 않게 아래쪽 자막과도 떨어뜨림)
ASK_FADE = 1.2                        # 안내가 떠오르는 시간 (초)
START_FADE = 2.0                      # 메인 화면에서 이야기로 넘어가는 시간 (초)
CAM_VIEW_W, CAM_MARGIN = 400, 36      # 웹캠 화면 폭과 가장자리 여백 (1920x1080 기준)
NOTICE_SECONDS = 2.2                  # S 키를 눌렀을 때 켜짐·꺼짐 안내를 보여 주는 시간
HELP_BUTTON = QRectF(STAGE_W - 128, STAGE_H - 128, 76, 76)   # 메인 화면 오른쪽 아래 ? 버튼

# 단축키 안내 창에 보여 줄 목록: (키들, 설명)
HELP_KEYS = [
    (["Space"], "이야기 시작\n재생 · 일시정지"),
    (["→", "←"], "다음 장면 · 이전 장면"),
    (["1", "~", "5"], "그림자 (꾹 누르면 유지)\n새·토끼·여우·사슴·달"),
    (["M"], "질문 녹음 시작 · 끝\n(이야기가 끝난 뒤에만)"),
    (["R"], "메인 화면으로 돌아가기"),
    (["C"], "웹캠 켜기 · 끄기\n(켜면 그림자 인식)"),
    (["B"], "빈 스크린 다시 찍기\n(손을 치우고 누르기)"),
    (["D"], "그림자 인식 진단\n(웹캠이 켜져 있을 때)"),
    (["N"], "자막 켜기 · 끄기"),
    (["S"], "자막 읽어 주기\n켜기 · 끄기"),
    (["F"], "전체 화면"),
    (["V"], "화면 맞추기 · 채우기"),
    (["?"], "단축키 안내 열기 · 닫기"),
    (["Esc"], "안내 닫기 · 앱 종료"),
]

SHADOW_NAMES = {"bird": "새", "rabbit": "토끼", "fox": "여우", "deer": "사슴", "moon": "달"}
SHADOW_GRACE = 0.6          # 인식기가 이만큼(초) 그림자를 못 보면 그림자를 푼 것으로 본다
SHADOW_RGB = (14, 16, 34)   # 그림자 색 (거의 검은 남색)
GUIDE_RIM_RGB = (255, 240, 200)  # 안내 모양 테두리 (달빛 크림색)
GUIDE_ALPHA = 0.7           # 기다리는 동안 보여 주는 안내 손그림자의 진하기
SHADOW_ALPHA = 0.92         # 인식된 손그림자의 진하기


def load_qimage(path: Path, tint: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> QImage:
    arr = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32)
    if tint != (1.0, 1.0, 1.0):
        arr[..., :3] *= np.array(tint, np.float32)
    return array_to_qimage(arr.clip(0, 255).astype(np.uint8))


def array_to_qimage(rgba: np.ndarray) -> QImage:
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QImage(rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()


def shadow_qimage(alpha: np.ndarray) -> QImage:
    """0~1 알파 지도를 그림자 색 이미지로."""
    h, w = alpha.shape
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = SHADOW_RGB
    rgba[..., 3] = (alpha * 255).clip(0, 255).astype(np.uint8)
    return array_to_qimage(rgba)


def guide_qimage(mask: np.ndarray) -> QImage:
    """따라 만들 손그림자 안내 모양: 그림자 색으로 채우고 바깥에 달빛 테두리와 은은한 빛번짐을
    두른다 (어두운 풀밭·숲 위에서도 모양이 또렷하게 보이도록)."""
    inside = (mask > 127).astype(np.float32)
    inside = cv2.GaussianBlur(inside, (0, 0), 1.2)
    solid = (mask > 127).astype(np.uint8)
    ring = cv2.dilate(solid, np.ones((9, 9), np.uint8)).astype(np.float32)
    ring = cv2.GaussianBlur(np.clip(ring - inside, 0, 1), (0, 0), 2.5)
    glow = cv2.dilate(solid, np.ones((21, 21), np.uint8)).astype(np.float32)
    glow = cv2.GaussianBlur(np.clip(glow - inside, 0, 1), (0, 0), 7) * 0.45
    ring = np.maximum(ring, glow)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), np.float32)
    rgba[..., :3] = np.array(SHADOW_RGB, np.float32) * inside[..., None] \
        + np.array(GUIDE_RIM_RGB, np.float32) * (1 - inside[..., None])
    rgba[..., 3] = np.clip(inside + ring * 0.9, 0, 1) * 255
    return array_to_qimage(rgba.clip(0, 255).astype(np.uint8))


class ShadowRig:
    """동물 하나의 그림자 연출 자료.

    guide  : 기다리는 동안 보여 주는 "따라 만들 손그림자" 모양
    frames : 인식된 손그림자 → 동물 실루엣으로 바뀌는 프레임들
    art    : 마지막에 색이 입혀진 동물 그림
    """

    WORK_W = 480  # 모핑 계산 해상도 (화면에는 확대해서 그린다)

    def __init__(self, spec: ShadowSpec, guide_mask: np.ndarray | None = None) -> None:
        self.spec = spec
        if not spec.art:   # 신호 그림자 (달): 동그란 모양을 안내로 쓴다
            self.size = (self.WORK_W, self.WORK_W)
            circle = np.zeros((self.WORK_W, self.WORK_W), np.uint8)
            c = self.WORK_W // 2
            cv2.circle(circle, (c, c), int(self.WORK_W * 0.42), 255, -1, cv2.LINE_AA)
            self.target, self.art = circle, None
            self._set_guide(fit_mask(guide_mask, self.size) if guide_mask is not None else circle)
            self.frames = [shadow_qimage(circle.astype(np.float32) / 255)]
            if guide_mask is not None:
                self.set_source(guide_mask)
            return
        art = Image.open(LAYERS / spec.art).convert("RGBA")
        self.size = (self.WORK_W, round(self.WORK_W * art.height / art.width))
        alpha = np.asarray(art.resize(self.size, Image.LANCZOS))[..., 3].astype(np.float32) / 255
        self.target = ((alpha > 0.5) * 255).astype(np.uint8)
        self.art = load_qimage(LAYERS / spec.art)
        if guide_mask is not None:
            self._set_guide(fit_mask(guide_mask, self.size))
            self.set_source(guide_mask)   # 실제 인식 모양이 오기 전까지는 안내 모양에서 시작
        else:                             # 안내 사진이 없으면 동물 실루엣을 안내로 쓴다
            self._set_guide(self.target)
            self.frames = [shadow_qimage(alpha)]

    def _set_guide(self, mask: np.ndarray) -> None:
        self.guide = guide_qimage(mask)

    def set_source(self, mask: np.ndarray) -> None:
        """인식된 손그림자 마스크(0/255)로 모핑 프레임을 새로 만든다."""
        src = fit_mask(mask, self.size)
        self.frames = [shadow_qimage(a) for a in morph_frames(src, self.target)]

    def rect(self) -> QRectF:
        w = self.spec.width
        h = w * self.size[1] / self.size[0]
        return QRectF(self.spec.x - w / 2, self.spec.y - h / 2, w, h)

    def paint(self, p: QPainter, t: float, now: float) -> None:
        s, r = self.spec, self.rect()
        if t < s.wait_at:                       # 실루엣이 서서히 나타남
            p.setOpacity(GUIDE_ALPHA * ease_in_out(t / s.wait_at))
            p.drawImage(r, self.guide)
        elif t == s.wait_at:                    # 그림자를 기다리는 중: 은은하게 숨 쉬기
            p.setOpacity(GUIDE_ALPHA * (0.75 + 0.25 * math.sin(now * 2.2)))
            p.drawImage(r, self.guide)
        else:
            e = t - s.wait_at
            if e < s.APPEAR:                    # 손그림자가 실루엣 자리에 나타남
                k = ease_in_out(e / s.APPEAR)
                p.setOpacity(GUIDE_ALPHA * (1 - k))
                p.drawImage(r, self.guide)
                p.setOpacity(SHADOW_ALPHA * k)
                p.drawImage(r, self.frames[0])
            elif self.art is None:              # 신호 그림자: 빛 속으로 사라짐
                k = ease_in_out(min(1.0, (e - s.APPEAR) / s.FADE))
                if k < 1.0:
                    p.setOpacity(SHADOW_ALPHA * (1 - k))
                    p.drawImage(r, self.frames[0])
            elif e < s.APPEAR + s.MORPH:        # 손그림자 → 동물 모양
                k = (e - s.APPEAR) / s.MORPH
                p.setOpacity(SHADOW_ALPHA)
                p.drawImage(r, self.frames[round(k * (len(self.frames) - 1))])
            else:                               # 색이 입혀지며 완성
                k = ease_in_out(min(1.0, (e - s.APPEAR - s.MORPH) / s.COLOR))
                p.setOpacity(SHADOW_ALPHA * (1 - k))
                p.drawImage(r, self.frames[-1])
                p.setOpacity(k)
                p.drawImage(r, self.art)
        p.setOpacity(1.0)


class Renderer:
    """컷의 특정 시점을 1920x1080 좌표계로 그린다. 창 출력과 스냅샷 저장에 같이 쓴다.

    guides: {동물: 마스크 PNG (assets/images/ 기준)} — 따라 만들 손그림자 모양.
            시험용 키로 인식을 흉내 내면 이 모양이 인식된 것으로 보고 동물로 바꾼다.
    """

    def __init__(self, guides: dict | None = None) -> None:
        self._cache: dict[tuple[str, tuple[float, float, float]], QImage] = {}
        self._flaps: dict[str, dict] = {}
        self._pose_rects: dict[str, tuple[float, float, float, float]] = {}
        self.rigs: dict[str, ShadowRig] = {}
        self.guides = guides or {}
        self.show_narration = True
        self.hide_title = False   # 단축키 안내 창이 열리면 뒤쪽 제목을 숨긴다
        self.ask_dim = 1.0        # 질문 안내 진하기 배율 (AI가 대답하는 동안 옅게)

    def image(self, folder: Path, name: str, tint=(1.0, 1.0, 1.0)) -> QImage:
        key = (str(folder / name), tint)
        if key not in self._cache:
            self._cache[key] = load_qimage(folder / name, tint)
        return self._cache[key]

    def flap_frames(self, folder: str) -> dict:
        """날갯짓 프레임 폴더의 frames.json (tools/split_bird.py가 만든다).
        angles: 날개 각도 (+ 들어 올림, - 내려침), files: 프레임 경로 (layers/ 기준),
        size: 원본 그림 크기, box: 프레임이 덮는 범위 (원본 픽셀 좌표, 날개가 밖으로 나가는 만큼 넓다)"""
        if folder not in self._flaps:
            meta = json.loads((LAYERS / folder / "frames.json").read_text(encoding="utf-8"))
            meta["angles"] = [float(a) for a in meta["angles"]]
            meta["files"] = [f"{folder}/{f}" for f in meta["files"]]
            self._flaps[folder] = meta
        return self._flaps[folder]

    def pose_rect(self, name: str) -> tuple[float, float, float, float]:
        """자세 그림을 기준 그림 위에 그릴 범위 (기준 그림 크기 = 1). layers/<그림>.json의 box.
        json이 없으면 기준 그림과 같은 캔버스로 본다 (표정만 바꾼 그림 등)."""
        if name not in self._pose_rects:
            path = LAYERS / Path(name).with_suffix(".json").name
            if not path.exists():
                self._pose_rects[name] = (0.0, 0.0, 1.0, 1.0)
                return self._pose_rects[name]
            meta = json.loads(path.read_text(encoding="utf-8"))
            (w, h), (x0, y0, x1, y1) = meta["size"], meta["box"]
            self._pose_rects[name] = (x0 / w, y0 / h, x1 / w, y1 / h)
        return self._pose_rects[name]

    def layer_images(self, layer, t: float) -> tuple[list[tuple[QImage, float, tuple]], float]:
        """레이어가 t초에 그릴 그림들 [(그림, 진하기, 범위)]과 기준 그림의 세로/가로 비.
        범위는 기준 그림(layer.image) 크기를 1로 본 (x0, y0, x1, y1). 기준 그림이면 (0, 0, 1, 1).
        날갯짓은 가장 가까운 각도의 프레임을 그리고, 그 옆 프레임을 거리만큼 살짝 겹쳐
        프레임 사이를 부드럽게 잇는다 (가운데에서 두 프레임 진하기가 같아져 튀지 않는다)."""
        base = self.image(LAYERS, layer.image, layer.tint)
        aspect = base.height() / base.width()
        if not layer.flap:
            out = []
            for name, k in layer.images_at(t):
                if name == layer.image:
                    out.append((base, k, (0.0, 0.0, 1.0, 1.0)))
                else:
                    out.append((self.image(LAYERS, name, layer.tint), k, self.pose_rect(name)))
            return out, aspect
        meta = self.flap_frames(layer.flap)
        angles, files = meta["angles"], meta["files"]
        (w, h), (x0, y0, x1, y1) = meta["size"], meta["box"]
        rect = (x0 / w, y0 / h, x1 / w, y1 / h)
        v = layer.flap_at(t)
        target = v * min(angles) if v > 0 else -v * max(angles)
        near = min(range(len(angles)), key=lambda k: abs(angles[k] - target))
        out = [(self.image(LAYERS, files[near], layer.tint), 1.0, rect)]
        side = near + (1 if (angles[near] - target) * (angles[0] - angles[-1]) > 0 else -1)
        if 0 <= side < len(angles) and angles[near] != target:
            weight = abs(angles[near] - target) / abs(angles[side] - angles[near])
            out.append((self.image(LAYERS, files[side], layer.tint), weight, rect))
        return out, h / w

    def preload(self, cuts: list[Cut]) -> None:
        for cut in cuts:
            self.image(SCENES, cut.background)
            if cut.next_background:
                self.image(SCENES, cut.next_background)
            for layer in cut.layers + (cut.shadow_hold.layers if cut.shadow_hold else []):
                self.image(LAYERS, layer.image, layer.tint)
                if layer.flap:
                    for f in self.flap_frames(layer.flap)["files"]:
                        self.image(LAYERS, f, layer.tint)
                for _, art in layer.poses:
                    self.image(LAYERS, art, layer.tint)
                    if art != layer.image:
                        self.pose_rect(art)
            if cut.shadow and cut.shadow.art:
                self.image(LAYERS, cut.shadow.art)
            if cut.shadow and cut.shadow.animal not in self.rigs:
                guide = None
                if cut.shadow.animal in self.guides:
                    guide = cv2.imread(str(PHOTOS / self.guides[cut.shadow.animal]), cv2.IMREAD_GRAYSCALE)
                self.rigs[cut.shadow.animal] = ShadowRig(cut.shadow, guide)

    def paint(self, p: QPainter, cut: Cut, t: float, now: float = 0.0,
              hold: float | None = None) -> None:
        """그리는 순서: 배경 → (아래) 다음 배경 → 레이어 → (위) 다음 배경 → 어둠
        → 어둠 위 레이어(동물) → 그림자 연출 → 따뜻한 빛 → 자막

        hold: 그림자를 유지해서 머무는 중이면 머문 지 몇 초인지 (t는 cut.shadow_hold.at).
              레이어는 머무는 동안의 되풀이 움직임으로 그리고, 자막은 마지막 장을 계속 보여 준다."""
        layers, lt = cut.layers, t
        if hold is not None and cut.shadow_hold:
            h = cut.shadow_hold
            u = hold % h.loop
            layers, lt = (h.layers, u) if h.layers else (cut.layers, h.at + u)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        full = QRectF(0, 0, STAGE_W, STAGE_H)
        p.drawImage(full, self.image(SCENES, cut.background))

        mix = cut.next_mix.at(t) if cut.next_background else 0.0
        if mix > 0 and cut.next_under_layers:
            p.setOpacity(mix)
            p.drawImage(full, self.image(SCENES, cut.next_background))
            p.setOpacity(1.0)

        self._paint_layers(p, [l for l in layers if not l.above_dark], lt)

        if mix > 0 and not cut.next_under_layers:
            p.setOpacity(mix)
            p.drawImage(full, self.image(SCENES, cut.next_background))
            p.setOpacity(1.0)

        dark = cut.darkness.at(t)
        if dark > 0:
            c = QColor(NIGHT)
            c.setAlphaF(dark)
            p.fillRect(full, c)

        # 동물은 어둠 위에 그려서 밝고 또렷하게 보이게 한다
        self._paint_layers(p, [l for l in layers if l.above_dark and not l.after_shadow], lt)
        # 그림자 연출이 끝나면 완성된 동물 그림 대신 움직이는 동물 레이어(after_shadow)가 이어받는다
        takeover = [l for l in layers if l.after_shadow]
        shadow_done = cut.shadow is None or t >= cut.shadow.done_at
        if cut.shadow and cut.shadow.animal in self.rigs and not (takeover and shadow_done):
            self.rigs[cut.shadow.animal].paint(p, t, now)
        if takeover and shadow_done:
            self._paint_layers(p, takeover, lt)

        warm = cut.warmth.at(t)
        if warm > 0:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SoftLight)
            c = QColor(255, 180, 90)
            c.setAlphaF(warm)
            p.fillRect(full, c)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        title = 0.0 if self.hide_title else cut.title_alpha.at(t)
        if title > 0:
            self._paint_title(p, title, now if cut.hold else None)

        if self.show_narration and cut.narration:
            nt = t
            if hold is not None:   # 머무는 동안 마지막 자막이 사라지지 않게
                nt = min(t, (cut.narration_until or cut.duration) - 1e-3)
            self._paint_narration_pages(p, cut, nt)

        if cut.ask_from is not None and t >= cut.ask_from and not self.hide_title:
            self._paint_ask(p, ease_in_out((t - cut.ask_from) / ASK_FADE) * self.ask_dim)

    def _paint_ask(self, p: QPainter, alpha: float) -> None:
        """이야기가 끝난 뒤 화면 가운데의 질문 안내."""
        if alpha <= 0:
            return
        p.setOpacity(alpha)
        self._outlined_text(p, self._title_font(84), ASK_TEXT, *ASK_POS, 7,
                            QColor(255, 248, 228), glow=True)
        p.setOpacity(1.0)

    def _title_font(self, size: int) -> QFont:
        """제목 글꼴. 굵은 스타일만 든 글꼴도 있어서, 등록된 스타일 이름을 그대로 쓴다."""
        if not hasattr(self, "_title_fam"):
            self._title_fam, self._title_style = self._font_family(), ""
            path = FONTS / TITLE_FONT
            if path.exists():
                fid = QFontDatabase.addApplicationFont(str(path))
                fams = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                if fams:
                    self._title_fam = fams[0]
                    styles = QFontDatabase.styles(fams[0])
                    self._title_style = styles[0] if styles else ""
        font = QFont(self._title_fam, size)
        if self._title_style:
            font.setStyleName(self._title_style)
        return font

    def _outlined_text(self, p: QPainter, font: QFont, text: str, cx: float, cy: float,
                       outline: float, fill: QColor, glow: bool = False) -> None:
        """(cx, cy)를 가운데로 외곽선 글자를 그린다.

        굵은 외곽선·번짐을 여러 겹 긋는 일(strokePath)은 아주 느려서(긴 문장이면 한 번에 100ms 넘게)
        매 프레임 하지 않는다. 한 번 이미지로 그려 두고 붙이기만 한다. 진하기는 p의 opacity를 따른다.
        """
        # 창이 무대보다 크면(4K 등) 그만큼 촘촘하게 그려 둬야 흐려지지 않는다
        scale = max(1.0, math.ceil(p.transform().m11() * 2) / 2)
        key = (font.toString(), font.styleName(), text, outline, fill.rgba(), glow, scale)
        cache = self.__dict__.setdefault("_text_cache", {})
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = self._render_outlined_text(font, text, outline, fill, glow, scale)
        img, rect = cache[key]
        p.drawImage(rect.translated(cx, cy), img)

    @staticmethod
    def _render_outlined_text(font: QFont, text: str, outline: float, fill: QColor, glow: bool,
                              scale: float) -> tuple[QImage, QRectF]:
        """외곽선 글자를 투명 이미지로 그린다. (이미지, 글자 가운데 기준 무대 좌표 범위)."""
        path = QPainterPath()
        path.addText(0, 0, font, text)
        br = path.boundingRect()
        path.translate(-br.width() / 2 - br.x(), -br.y() - br.height() / 2)
        margin = 30 + outline                       # 번짐(44/2)·그림자(4, 6)가 잘리지 않게
        rect = path.boundingRect().adjusted(-margin, -margin, margin, margin)
        img = QImage(math.ceil(rect.width() * scale), math.ceil(rect.height() * scale),
                     QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        q = QPainter(img)
        q.setRenderHint(QPainter.RenderHint.Antialiasing)
        q.scale(scale, scale)
        q.translate(-rect.x(), -rect.y())
        pen = lambda c, w: QPen(c, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                                Qt.PenJoinStyle.RoundJoin)
        q.strokePath(path.translated(4, 6), pen(QColor(14, 18, 48, 110), outline * 2))  # 그림자
        if glow:                                                     # 달빛 번짐
            for width, a in ((44, 16), (30, 26), (18, 40)):
                q.strokePath(path, pen(QColor(255, 236, 190, a), width))
        q.strokePath(path, pen(QColor(26, 32, 78, 230), outline))
        q.fillPath(path, fill)
        q.end()
        return img, QRectF(rect.x(), rect.y(), img.width() / scale, img.height() / scale)

    def _paint_title(self, p: QPainter, alpha: float, now: float | None) -> None:
        """메인 화면 제목. 은은한 달빛 번짐 + 남색 테두리로 수채화 배경 위에서도 또렷하게.
        now가 있으면(메인 화면) 아래에 시작 안내를 은은하게 깜박인다."""
        p.setOpacity(alpha)
        self._outlined_text(p, self._title_font(140), TITLE_TEXT, *TITLE_POS, 7,
                            QColor(255, 248, 228), glow=True)
        if now is not None:
            hint = QFont(self._font_family(), 28)
            hint.setBold(True)
            p.setOpacity(alpha * (0.6 + 0.35 * math.sin(now * 2.0)))
            self._outlined_text(p, hint, "스페이스바를 누르면 이야기가 시작돼요",
                                TITLE_POS[0], TITLE_POS[1] + 140, 6, QColor(255, 248, 228))
        p.setOpacity(1.0)

    def _paint_layers(self, p: QPainter, layers: list, t: float) -> None:
        for layer in layers:
            op = layer.opacity.at(t)
            w = layer.width.at(t)
            if op <= 0 or w <= 0:
                continue
            images, aspect = self.layer_images(layer, t)
            h = w * aspect
            cx, cy, rot, sx, sy = layer.pose(t)
            # 중심(pivot)을 기준으로 돌리고 늘인다. 그림은 원본 크기(w, h) 기준 범위에 그린다
            ox, oy = layer.pivot[0] * w, layer.pivot[1] * h
            p.save()
            p.translate(cx + ox, cy + oy)
            if rot:
                p.rotate(rot)
            if sx != 1 or sy != 1:
                p.scale(sx, sy)
            for img, k, (fx0, fy0, fx1, fy1) in images:
                p.setOpacity(op * min(k, 1.0))
                p.drawImage(QRectF(-w / 2 - ox + fx0 * w, -h / 2 - oy + fy0 * h,
                                   (fx1 - fx0) * w, (fy1 - fy0) * h), img)
            p.restore()
        p.setOpacity(1.0)

    def _paint_narration_pages(self, p: QPainter, cut: Cut, t: float) -> None:
        """자막 페이지를 차례로 보여 준다. 각 페이지는 글자 수에 맞춰 시간을 나눠 받는다."""
        at = narration_page_at(cut, t)
        if at is None:
            return
        i, start = at
        out = min(1.0, (cut.narration_until - t) / 0.6) if cut.narration_until else 1.0   # 사라질 때
        first = (i == 0 and cut.narration_from == 0)
        fade = 1.0 if first else min(1.0, (t - start) / 0.4)
        p.setOpacity(fade * out)
        self._paint_narration(p, narration_pages(cut.narration)[i])
        p.setOpacity(1.0)

    def _font_family(self) -> str:
        """한글 글꼴을 파일에서 직접 등록한다 (창 없는 스냅샷 모드에서도 한글이 나오도록)."""
        if not hasattr(self, "_family"):
            self._family = "Malgun Gothic"
            for path in (FONTS / "narration.ttf", Path("C:/Windows/Fonts/malgunbd.ttf"),
                         Path("C:/Windows/Fonts/malgun.ttf")):
                if path.exists():
                    fid = QFontDatabase.addApplicationFont(str(path))
                    families = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
                    if families:
                        self._family = families[0]
                        break
        return self._family

    def _paint_narration(self, p: QPainter, text: str) -> None:
        """화면 아래에 자막. 외곽선을 둘러 어떤 배경에서도 읽히게 한다."""
        font = QFont(self._font_family(), 34)
        font.setBold(True)
        lines = text.split("\n")
        line_h = 58
        y0 = STAGE_H - 70 - line_h * (len(lines) - 1)
        for i, line in enumerate(lines):
            path = QPainterPath()
            path.addText(0, 0, font, line)
            br = path.boundingRect()
            path.translate((STAGE_W - br.width()) / 2 - br.x(), y0 + i * line_h)
            p.strokePath(path, QPen(QColor(10, 14, 40, 220), 8, Qt.PenStyle.SolidLine,
                                    Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.fillPath(path, QColor(255, 250, 235))

    def snapshot(self, cut: Cut, t: float, out: Path) -> None:
        """t가 그림자 대기 시점(wait_at)보다 크면 그림자가 인식된 것으로 그린다."""
        img = QImage(STAGE_W, STAGE_H, QImage.Format.Format_RGB32)
        p = QPainter(img)
        self.paint(p, cut, t)
        p.end()
        img.save(str(out))


class Stage(QWidget):
    """컷을 자동으로 이어서 재생하는 창.

    메인 화면(hold 컷)에서 Space를 누르면 시작한다. 컷이 끝나면 다음 컷으로 넘어가고,
    마지막 컷이 끝나면 메인 화면으로 돌아간다. shadow가 있는 컷은 wait_at 시점에서 시간이
    멈추고, 그 그림자가 인식되면(on_shadow 호출) 이어서 진행한다.

    shadow_hold가 있는 컷은 그 시점에 그림자를 아직 유지하고 있으면 머무르며 되풀이 움직임을
    보여 주고, 그림자를 풀면 이어서 진행한다. 인식기는 그림자가 보이는 동안 on_shadow를 계속
    호출하고, SHADOW_GRACE초 동안 호출이 없거나 on_shadow_lost를 부르면 푼 것으로 본다.

    마지막 컷(ask_from)은 자동으로 넘어가지 않는다. 질문 안내가 떠오른 뒤에만 대화형 AI가 켜지고
    (M 키로 녹음 시작·끝), Space를 누르면 메인 화면으로 돌아간다. 이야기 중에는 M을 눌러도 무시한다.

    운영자 키:
      Space : 메인 화면에서는 시작, 이야기 중에는 재생/일시정지, 질문 받기 중에는 메인 화면으로
      M : 질문 받기 중에 녹음 시작 / 끝 (끝내면 AI가 목소리와 자막으로 대답한다)
      1~5 : 그림자 인식 흉내 (1 새, 2 토끼, 3 여우, 4 사슴, 5 달) — 웹캠 없이 시험할 때.
            누르고 있는 동안은 그림자를 유지하는 것으로, 떼면 푼 것으로 본다
      → : 다음 컷      ← : 이전 컷      R : 메인 화면으로
      N : 자막 켜기/끄기      S : 자막 읽어 주기 켜기/끄기 (Leda 목소리)      F : 전체 화면      V : 맞추기/채우기
      C : 웹캠 켜기/끄기 (오른쪽 위). 켜면 그림자 인식기(watcher.py)도 함께 돌아,
          지금 장면이 요구하는 그림자(cut.shadow / shadow_hold의 동물)가 보이면 on_shadow를 부른다.
          다른 동물 그림자는 무시한다 (예: 첫 그림자 장면은 새만 받는다)
      B : 빈 스크린(배경) 다시 찍기 — 손을 치우고 누른다      ? : 단축키 안내      Esc : 종료
      D : 그림자 인식 진단 켜기/끄기 — 인식기가 보는 것을 겹쳐 보여 준다 (저장하지 않음)
    메인 화면 오른쪽 아래의 ? 버튼을 눌러도 단축키 안내가 열린다.
    """

    TEST_KEYS = {Qt.Key.Key_1: "bird", Qt.Key.Key_2: "rabbit",
                 Qt.Key.Key_3: "fox", Qt.Key.Key_4: "deer", Qt.Key.Key_5: "moon"}

    def __init__(self, cuts: list[Cut], start: int = 0, guides: dict | None = None) -> None:
        super().__init__()
        self.cuts = cuts
        self.index = start
        self.renderer = Renderer(guides)
        self.renderer.preload(cuts)
        self.recognized: set[str] = set()   # 지금까지 인식된 그림자
        self._seen: dict[str, float] = {}   # 그림자별로 인식기가 마지막으로 본 때 (실제 시각)
        self._keys_down: set[str] = set()   # 시험용 키를 누르고 있는 그림자
        self._hold_checked = False           # 이 컷의 머물 시점을 지났는지
        self._hold_start: float | None = None  # 머물기 시작한 이야기 시계 (머무는 중이 아니면 None)
        self._hold_until: float | None = None  # 그림자를 푼 뒤 이어서 진행할 때 (머문 때부터 초)
        self.paused_at: float | None = None  # 일시정지한 시각 (재생 중이면 None)
        self.fade_from: QImage | None = None  # 부드럽게 넘어갈 때 사라지는 앞 화면
        self.fill_screen = False              # False=맞추기(여백), True=채우기(가장자리 잘림)
        self.camera = Camera()                # C 키로 켜고 끈다
        self.watcher = ShadowWatcher(self.camera)   # 카메라와 함께 켜지는 그림자 인식
        self.talk = Talk()                    # 이야기가 끝난 뒤의 대화형 AI
        self.narrator = Narrator(cuts)        # S 키로 켜고 끄는 자막 읽어 주기
        self._notice, self._notice_at = "", -math.inf   # 잠깐 보여 주는 안내 (S 키)
        self.help_open = False                # 단축키 안내 창
        self.help_hover = False               # ? 버튼 위에 마우스가 있는지
        self._help_paused = False             # 안내 창을 열면서 이야기를 멈췄는지
        self._view = (1.0, 0.0, 0.0)          # (배율, x 여백, y 여백)
        self.setMouseTracking(True)
        self.fade_start = self.fade_len = 0.0
        self.started = time.perf_counter()
        self.resize(1280, 720)
        self.setStyleSheet("background: black;")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)
        self._update_title()

    @property
    def cut(self) -> Cut:
        return self.cuts[self.index]

    @property
    def paused(self) -> bool:
        return self.paused_at is not None

    def _clock(self) -> float:
        """이야기 시계. 일시정지 중에는 멈춘다."""
        return self.paused_at if self.paused_at is not None else time.perf_counter()

    @property
    def waiting(self) -> bool:
        """지금 컷이 그림자를 기다리느라 멈춰 있는지."""
        s = self.cut.shadow
        return bool(s and s.animal not in self.recognized
                    and self._clock() - self.started >= s.wait_at)

    @property
    def holding(self) -> bool:
        """그림자를 유지하고 있어서 머무는 중인지."""
        return self._hold_start is not None

    @property
    def hold_elapsed(self) -> float | None:
        return self._clock() - self._hold_start if self.holding else None

    @property
    def t(self) -> float:
        if self.waiting:
            return self.cut.shadow.wait_at
        h = self.cut.shadow_hold
        if h and (self.holding or not self._hold_checked):
            # 머무는 중이거나, 머물지 아직 정하지 않았으면(_tick 전 한 프레임) at을 넘지 않는다
            return min(self._clock() - self.started, h.at, self.cut.duration)
        return min(self._clock() - self.started, self.cut.duration)

    @property
    def asking(self) -> bool:
        """이야기가 끝나고 질문을 받는 중인지 (이때만 대화형 AI가 켜진다)."""
        a = self.cut.ask_from
        return a is not None and not self.paused and self.t >= a

    def is_held(self, name: str) -> bool:
        """그 그림자를 지금 유지하고 있는지."""
        return (name in self._keys_down
                or time.perf_counter() - self._seen.get(name, -math.inf) < SHADOW_GRACE)

    def _update_hold(self) -> None:
        """머물 시점에 그림자를 유지하고 있으면 머물고, 풀면 되풀이 움직임을 마저 끝내고 진행한다."""
        h = self.cut.shadow_hold
        if not h:
            return
        now = self._clock()
        if not self.holding:
            if not self._hold_checked and not self.waiting and now - self.started >= h.at:
                self._hold_checked = True
                if self.is_held(h.animal):
                    self._hold_start = self.started + h.at
            return
        u = now - self._hold_start
        if self.is_held(h.animal):
            self._hold_until = None             # 다시 그림자를 만들면 계속 머문다
        elif self._hold_until is None:
            self._hold_until = h.exit_after(u)
        if self._hold_until is not None and u >= self._hold_until:
            self.started += self._hold_until    # 머문 시간만큼 미뤄서 at부터 이어진다
            self._hold_start = self._hold_until = None

    # ---- 진행 ----
    def go(self, index: int, fade: float = 0.0, keep_voice: bool = False) -> None:
        """index 컷으로 간다. fade초를 주면 지금 화면에서 부드럽게 넘어가고,
        새 컷은 넘어가기가 끝난 뒤에 시작한다.
        keep_voice: 자연스럽게 다음 컷으로 이어질 때는 읽던 자막을 끊지 않는다."""
        if not keep_voice:
            self.narrator.stop()
        if fade > 0:
            self.fade_from = self._render_frame()
            self.fade_start, self.fade_len = time.perf_counter(), fade
        self.talk.cancel()                  # 질문 받기 화면을 떠나면 녹음·대답·대화 기록을 모두 멈춘다
        self.index = max(0, min(index, len(self.cuts) - 1))
        self.started = time.perf_counter() + fade
        self.paused_at = None
        self._hold_checked = False
        self._hold_start = self._hold_until = None
        # 그림자 컷에 들어올 때마다 새로 기다린다 (앞서 눌린 키나 잘못 잡힌 그림자로 건너뛰지 않게)
        if self.cut.shadow:
            self.recognized.discard(self.cut.shadow.animal)
        self._update_title()

    def restart(self, fade: float = 0.0) -> None:
        """메인 화면으로 돌아가 처음부터."""
        self.recognized.clear()
        self.go(0, fade)

    def _render_frame(self) -> QImage:
        img = QImage(STAGE_W, STAGE_H, QImage.Format.Format_RGB32)
        p = QPainter(img)
        self.renderer.paint(p, self.cut, self.t, time.perf_counter(), self.hold_elapsed)
        p.end()
        return img

    def toggle_pause(self) -> None:
        now = time.perf_counter()
        if self.paused_at is None:
            self.paused_at = now
        else:
            self.started += now - self.paused_at   # 멈춰 있던 시간만큼 미룬다
            if self._hold_start is not None:
                self._hold_start += now - self.paused_at
            self.paused_at = None
        self._update_title()

    def on_shadow(self, name: str, mask: np.ndarray | None = None) -> None:
        """그림자 인식기가 호출한다. mask(0/255)를 주면 그 모양에서 동물로 바뀐다.

        지금 컷이 기다리는 그림자일 때만 받는다. 다른 때 들어온 인식은 무시한다.
        기다리는 중이었다면 그 시점부터 이어서 진행하고, 안내 모양이 나타나는 중에
        미리 인식됐다면 기다리지 않고 바로 이어진다.
        """
        self._seen[name] = time.perf_counter()   # 계속 불리는 동안 그림자를 유지하는 것으로 본다
        s = self.cut.shadow
        if not s or s.animal != name or name in self.recognized:
            return
        if mask is not None and name in self.renderer.rigs:
            self.renderer.rigs[name].set_source(mask)
        was_waiting = self.waiting and self.cut.shadow.animal == name
        self.recognized.add(name)
        if was_waiting:
            self.started = self._clock() - self.cut.shadow.wait_at
        self._update_title()

    def on_shadow_lost(self, name: str) -> None:
        """인식기가 그림자가 사라졌다고 알릴 때 (부르지 않아도 SHADOW_GRACE초 뒤에 푼 것으로 본다)."""
        self._seen.pop(name, None)

    @property
    def wanted_shadow(self) -> str | None:
        """지금 장면이 요구하는 그림자 (기다리는 그림자, 또는 머물기 위해 유지할 그림자)."""
        s, h = self.cut.shadow, self.cut.shadow_hold
        if s and s.animal not in self.recognized:
            return s.animal
        if h and (self.holding or not self._hold_checked):
            return h.animal
        return None

    def _tick(self) -> None:
        seen = self.watcher.stable if self.watcher.running else None
        if seen:
            self.on_shadow(*seen)   # 지금 장면이 요구하는 동물이 아니면 on_shadow가 무시한다
        self._update_hold()
        self.talk.poll()
        self.narrator.update(self.cut, self.t, self.paused)
        cut = self.cut
        # AI가 대답하는 동안은 질문 안내를 부드럽게 숨기고 자막에 눈이 가게 한다
        target = 0.0 if self.talk.state == "speaking" else 1.0
        self.renderer.ask_dim += (target - self.renderer.ask_dim) * 0.12
        if (not cut.hold and cut.ask_from is None and not self.waiting and not self.holding
                and self.t >= cut.duration):
            if self.index + 1 < len(self.cuts):
                self.go(self.index + 1, keep_voice=True)
            else:
                self.restart()          # 이야기가 끝나면 메인 화면으로
        self.update()

    def _update_title(self) -> None:
        self.setWindowTitle(TITLE_TEXT + (" [일시정지]" if self.paused else ""))

    # ---- 그리기 ----
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        # 16:9 비율은 항상 지킨다 (늘리지 않음).
        #   맞추기: 화면 안에 다 보이게 넣고 남는 곳은 검은 여백
        #   채우기: 화면을 꽉 채우고 넘치는 가장자리는 잘라 냄
        sx, sy = self.width() / STAGE_W, self.height() / STAGE_H
        scale = max(sx, sy) if self.fill_screen else min(sx, sy)
        ox, oy = (self.width() - STAGE_W * scale) / 2, (self.height() - STAGE_H * scale) / 2
        self._view = (scale, ox, oy)          # 마우스 좌표를 무대 좌표로 바꿀 때 쓴다
        p.translate(ox, oy)
        p.scale(scale, scale)
        # 무대 밖(검은 여백)으로 나간 구름·새 등은 보이지 않게 자른다
        p.setClipRect(QRectF(0, 0, STAGE_W, STAGE_H))
        # 반짝임 같은 효과도 이야기 시계를 따른다 → 일시정지하면 함께 멈춘다
        self.renderer.paint(p, self.cut, self.t, self._clock(), self.hold_elapsed)
        if self.fade_from is not None:      # 앞 화면이 서서히 사라지며 넘어감
            k = (time.perf_counter() - self.fade_start) / self.fade_len
            if k >= 1:
                self.fade_from = None
            else:
                p.setOpacity(1 - ease_in_out(k))
                p.drawImage(QRectF(0, 0, STAGE_W, STAGE_H), self.fade_from)
                p.setOpacity(1.0)
        if self.camera.running or self.camera.status == "카메라 없음":
            self._paint_camera(p)
        if self.asking and not self.help_open:
            self._paint_talk(p)
        if self.paused and not self.help_open:
            self._paint_pause(p)
        if self.cut.hold and not self.help_open:
            self._paint_help_button(p)
        if self.help_open:
            self._paint_help(p)
        self._paint_notice(p)
        p.end()

    # ---- 대화형 AI ----
    def _paint_talk(self, p: QPainter) -> None:
        """질문 안내 아래의 상태 표시(듣는 중·생각하는 중)와 화면 아래 대답 자막."""
        talk, now = self.talk, time.perf_counter()
        fade = ease_in_out((self.t - self.cut.ask_from) / ASK_FADE)
        cx, cy = ASK_POS[0], ASK_POS[1] + 130
        font = QFont(self.renderer._font_family(), 30)
        font.setBold(True)
        cream = QColor(255, 248, 228)
        if talk.state == "recording":
            # 숨 쉬듯 커졌다 작아지는 빨간 점 + 안내
            text = "듣고 있어요   (M 키를 다시 누르면 끝나요)"
            pulse = 0.5 + 0.5 * math.sin(now * 5.0)
            p.setFont(font)
            tw = p.fontMetrics().horizontalAdvance(text)
            dot = QRectF(0, 0, 30, 30)
            dot.moveCenter(QRectF(cx - tw / 2 - 40, cy - 15, 30, 30).center())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(255, 110, 100, int(60 + 60 * pulse)))
            grow = 8 + 8 * pulse
            p.drawEllipse(dot.adjusted(-grow, -grow, grow, grow))
            p.setBrush(QColor(240, 80, 74))
            p.drawEllipse(dot)
            self.renderer._outlined_text(p, font, text, cx + 20, cy, 6, cream)
        elif talk.state == "thinking":
            dots = "." * (1 + int(now * 2.5) % 3)
            self.renderer._outlined_text(p, font, "생각하고 있어요" + dots.ljust(3, " "),
                                         cx, cy, 6, cream)
        elif talk.state == "error":
            self.renderer._outlined_text(p, font, talk.message, cx, cy, 6, cream)
        elif talk.state == "idle":
            p.setOpacity(fade * (0.6 + 0.35 * math.sin(now * 2.0)))
            hint = QFont(self.renderer._font_family(), 26)
            hint.setBold(True)
            self.renderer._outlined_text(p, hint, "M 키를 누르고 말해 보세요   ·   스페이스바를 누르면 처음으로",
                                         cx, cy, 6, cream)
            p.setOpacity(1.0)
        sub = talk.subtitle()
        if sub and self.renderer.show_narration:
            self.renderer._paint_narration(p, sub)

    # ---- 자막 읽어 주기 ----
    def toggle_narrator(self) -> None:
        self.narrator.toggle()
        self._notice = "자막 읽어 주기 켜짐" if self.narrator.enabled else "자막 읽어 주기 꺼짐"
        self._notice_at = time.perf_counter()

    def _paint_notice(self, p: QPainter) -> None:
        """화면 위쪽 가운데에 잠깐 떠올랐다 사라지는 안내 (켜짐·꺼짐)."""
        age = time.perf_counter() - self._notice_at
        if age >= NOTICE_SECONDS or not self._notice:
            return
        text = self._notice
        if self.narrator.enabled and self.narrator.missing:
            text += f"   (목소리 준비 중 {self.narrator.missing}개 남음)"
        p.setOpacity(min(1.0, age / 0.2, (NOTICE_SECONDS - age) / 0.5))
        font = QFont(self.renderer._font_family(), 28)
        font.setBold(True)
        self.renderer._outlined_text(p, font, text, STAGE_W / 2, 90, 6, QColor(255, 248, 228))
        p.setOpacity(1.0)

    # ---- 단축키 안내 ----
    def _paint_help_button(self, p: QPainter) -> None:
        """메인 화면 오른쪽 아래의 동그란 ? 버튼."""
        r = HELP_BUTTON
        c = r.center()
        if self.help_hover:  # 마우스를 올리면 달빛처럼 번진다
            for rad, a in ((r.width() / 2 + 18, 40), (r.width() / 2 + 10, 70)):
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 236, 190, a))
                p.drawEllipse(c, rad, rad)
        p.setPen(QPen(QColor(26, 32, 78, 200), 3))
        p.setBrush(QColor(255, 248, 228, 235 if self.help_hover else 205))
        p.drawEllipse(r)
        p.setPen(QColor(30, 38, 92))
        p.setFont(self.renderer._title_font(44))
        p.drawText(r.adjusted(0, -2, 0, 0), Qt.AlignmentFlag.AlignCenter, "?")

    def _paint_help(self, p: QPainter) -> None:
        """화면 가운데 단축키 안내 창."""
        full = QRectF(0, 0, STAGE_W, STAGE_H)
        p.fillRect(full, QColor(8, 10, 30, 150))                      # 뒤를 어둡게

        panel = QRectF(0, 0, 1280, 890)
        panel.moveCenter(full.center())
        p.setPen(Qt.PenStyle.NoPen)
        for grow, a in ((26, 18), (14, 30)):                          # 달빛 번짐
            p.setBrush(QColor(255, 236, 190, a))
            p.drawRoundedRect(panel.adjusted(-grow, -grow, grow, grow), 40 + grow, 40 + grow)
        p.setBrush(QColor(20, 26, 66, 248))
        p.setPen(QPen(QColor(255, 236, 196), 3))
        p.drawRoundedRect(panel, 40, 40)
        p.setPen(QPen(QColor(255, 236, 196, 90), 1.5))                # 안쪽 가는 선
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(panel.adjusted(14, 14, -14, -14), 30, 30)

        # 제목 + 작은 달
        title_font = self.renderer._title_font(50)
        moon_c = QRectF(0, 0, 44, 44)
        p.setFont(title_font)
        tw = p.fontMetrics().horizontalAdvance("단축키 안내")
        x0 = panel.center().x() - (tw + 64) / 2
        moon_c.moveCenter(QRectF(x0, panel.top() + 52, 44, 44).center())
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 236, 190, 60))
        p.drawEllipse(moon_c.adjusted(-8, -8, 8, 8))
        p.setBrush(QColor(255, 240, 196))
        p.drawEllipse(moon_c)
        p.setPen(QColor(255, 248, 228))
        p.drawText(QRectF(x0 + 64, panel.top() + 40, tw + 10, 70),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "단축키 안내")

        # 키 목록 (두 줄로 나눠 배치)
        key_font = self.renderer._title_font(26)
        desc_font = QFont(self.renderer._font_family(), 21)
        left = panel.left() + 80
        col_w = (panel.width() - 160) / 2
        key_area = 210                     # 키캡이 차지하는 폭 (설명은 그 오른쪽)
        top = panel.top() + 150
        row_h = 92
        half = (len(HELP_KEYS) + 1) // 2
        for i, (keys, desc) in enumerate(HELP_KEYS):
            col, row = divmod(i, half)
            x = left + col * col_w
            y = top + row * row_h
            kx = x
            for key in keys:
                p.setFont(key_font)
                kw = max(56, p.fontMetrics().horizontalAdvance(key) + 34)
                cap = QRectF(kx, y, kw, 56)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(4, 6, 20, 120))                    # 키캡 그림자
                p.drawRoundedRect(cap.translated(0, 5), 14, 14)
                p.setBrush(QColor(255, 246, 222))
                p.setPen(QPen(QColor(214, 196, 160), 2))
                p.drawRoundedRect(cap, 14, 14)
                p.setPen(QColor(30, 38, 92))
                p.drawText(cap, Qt.AlignmentFlag.AlignCenter, key)
                kx += kw + 10
            p.setFont(desc_font)
            p.setPen(QColor(255, 248, 228))
            p.drawText(QRectF(x + key_area, y - 12, col_w - key_area - 20, 80),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                       | Qt.TextFlag.TextWordWrap, desc)

        p.setFont(QFont(self.renderer._font_family(), 19))
        p.setPen(QColor(255, 236, 196, 170))
        p.drawText(QRectF(panel.left(), panel.bottom() - 78, panel.width(), 40),
                   Qt.AlignmentFlag.AlignCenter, "아무 곳이나 누르거나  ?  ·  Esc  키를 누르면 닫혀요")

    def toggle_help(self) -> None:
        if self.help_open:
            self.help_open = False
            if self._help_paused:        # 안내를 열면서 멈췄던 이야기는 다시 재생
                self.toggle_pause()
            self._help_paused = False
        else:
            self.help_open = True
            self._help_paused = not self.cut.hold and not self.paused
            if self._help_paused:
                self.toggle_pause()
        self.renderer.hide_title = self.help_open
        self.help_hover = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _to_stage(self, pos) -> tuple[float, float]:
        """창 좌표 → 무대(1920x1080) 좌표."""
        return ((pos.x() - self._view[1]) / self._view[0], (pos.y() - self._view[2]) / self._view[0])

    def mousePressEvent(self, e) -> None:
        if self.help_open:
            self.toggle_help()
            return
        x, y = self._to_stage(e.position())
        if self.cut.hold and HELP_BUTTON.adjusted(-8, -8, 8, 8).contains(x, y):
            self.toggle_help()

    def mouseMoveEvent(self, e) -> None:
        x, y = self._to_stage(e.position())
        hover = (self.cut.hold and not self.help_open
                 and HELP_BUTTON.adjusted(-8, -8, 8, 8).contains(x, y))
        if hover != self.help_hover:
            self.help_hover = hover
            self.setCursor(Qt.CursorShape.PointingHandCursor if hover else Qt.CursorShape.ArrowCursor)

    def _camera_rect(self) -> QRectF:
        w = CAM_VIEW_W
        h = w * 3 / 4
        return QRectF(STAGE_W - CAM_MARGIN - w, CAM_MARGIN, w, h)

    def _paint_camera(self, p: QPainter) -> None:
        """오른쪽 위 웹캠 화면 (무대 위에 직접 그린다)."""
        r = self._camera_rect()
        frame = self.camera.latest()
        p.setOpacity(1.0)
        clip = QPainterPath()
        clip.addRoundedRect(r, 14, 14)
        p.save()
        p.setClipPath(clip)
        p.fillRect(r, QColor(10, 12, 30, 230))
        if frame is not None:
            h, w = frame.shape[:2]
            img = QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888)
            # 4:3이 아닌 카메라도 찌그러지지 않게 가운데를 맞춰 채운다
            scale = max(r.width() / w, r.height() / h)
            dw, dh = w * scale, h * scale
            p.drawImage(QRectF(r.center().x() - dw / 2, r.center().y() - dh / 2, dw, dh), img)
            if self.watcher.diag:
                self._paint_diag(p, QRectF(r.center().x() - dw / 2, r.center().y() - dh / 2, dw, dh))
        else:
            p.setPen(QColor(255, 248, 228))
            p.setFont(QFont(self.renderer._font_family(), 22))
            msg = "카메라 연결 중…" if self.camera.status == "연결 중" else "카메라를 찾을 수 없어요"
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, msg)
        p.restore()
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 240, 200, 200), 3))
        p.drawRoundedRect(r, 14, 14)
        self._paint_watch_status(p, r)

    def _paint_diag(self, p: QPainter, cam: QRectF) -> None:
        """진단(D 키): 인식기가 찾은 스크린(초록 테두리)과 그림자(주황)를 카메라 화면에 겹친다."""
        screen, shadow = self.watcher.outline, self.watcher.shadow
        for m, rgb, fill in ((screen, (60, 220, 120), False), (shadow, (255, 120, 40), True)):
            if m is None:
                continue
            u = (m > 0).astype(np.uint8)
            if not fill:
                u = u - cv2.erode(u, np.ones((7, 7), np.uint8))
            rgba = np.zeros((*u.shape, 4), np.uint8)
            rgba[..., :3] = rgb
            rgba[..., 3] = u * (110 if fill else 230)
            rgba = np.ascontiguousarray(rgba)
            img = QImage(rgba.data, u.shape[1], u.shape[0], u.shape[1] * 4, QImage.Format.Format_RGBA8888).copy()
            p.drawImage(cam, img)

    def _paint_watch_status(self, p: QPainter, cam: QRectF) -> None:
        """웹캠 화면 아래: 그림자 인식 상태 (필요한 그림자 · 지금 보이는 그림자)."""
        w = self.watcher
        if w.status == "꺼짐" or self.camera.latest() is None:
            return
        if w.status == "준비 중":
            text = "그림자 인식 준비 중…"
        elif w.status == "배경 찍는 중":
            text = "빈 스크린을 찍는 중 · 손을 치워 주세요"
        elif w.status == "모델 없음":
            text = "인식 모델이 없어요 (tools/train_silhouette.py)"
        elif w.status == "오류":
            text = "인식 모델을 불러오지 못했어요"
        else:
            want = self.wanted_shadow
            name, prob = w.raw
            stable = w.stable
            seen = (f"{SHADOW_NAMES.get(stable[0], stable[0])} ✓" if stable
                    else f"{SHADOW_NAMES.get(name, name)}?" if name else "없음")
            text = (f"필요: {SHADOW_NAMES.get(want, want)}   보임: {seen}" if want
                    else f"보이는 그림자: {seen}")
        y = self._paint_text_box(p, cam, cam.bottom() + 10, [text], 24)
        if w.diag and w.status == "인식 중":
            info = [SHADOW_NAMES.get(line, line) for line in w.info]
            self._paint_text_box(p, cam, y + 8, ["진단 중", *info], 22)

    def _paint_text_box(self, p: QPainter, cam: QRectF, top: float, lines: list[str], px: int) -> float:
        """카메라 화면 폭의 어두운 상자에 글자를 여러 줄로 쓴다. 상자 아래쪽 y를 돌려준다.

        글자 크기는 픽셀로 정한다 (포인트로 정하면 화면 배율이 큰 모니터에서 커져 잘린다).
        한 줄이 상자보다 길면 그 줄을 줄이고, 그래도 길면 줄을 나누고 상자를 늘린다.
        """
        width, pad = cam.width() - 24, 10
        flags = int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap)
        laid = []
        for text in lines:
            font = QFont(self.renderer._font_family())
            size = px
            while True:
                font.setPixelSize(size)
                p.setFont(font)
                if size <= 16 or p.fontMetrics().horizontalAdvance(text) <= width:
                    break
                size -= 1
            h = p.boundingRect(QRectF(0, 0, width, 1000), flags, text).height()
            laid.append((QFont(font), text, h))
        box = QRectF(cam.left(), top, cam.width(), sum(h for _, _, h in laid) + 2 * pad)
        p.setOpacity(1.0)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(10, 12, 30, 200))
        p.drawRoundedRect(box, 12, 12)
        p.setPen(QColor(255, 248, 228))
        y = top + pad
        for font, text, h in laid:
            p.setFont(font)
            p.drawText(QRectF(box.left() + 12, y, width, h), flags, text)
            y += h
        return box.bottom()

    def _paint_pause(self, p: QPainter) -> None:
        """오른쪽 위에 작은 일시정지 표시 (두 막대). 카메라 화면이 있으면 그 왼쪽에."""
        right = STAGE_W - 48
        if self.camera.running or self.camera.status == "카메라 없음":
            right = self._camera_rect().left() - 24
        p.setOpacity(0.75)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 248, 228))
        p.drawRoundedRect(QRectF(right - 44, 40, 16, 52), 4, 4)
        p.drawRoundedRect(QRectF(right - 16, 40, 16, 52), 4, 4)
        p.setOpacity(1.0)

    def toggle_camera(self) -> None:
        if self.camera.running or self.camera.status == "카메라 없음":
            self.watcher.stop()
            self.camera.stop()
        else:
            self.camera.start()
            self.watcher.start()

    def closeEvent(self, e) -> None:
        self.watcher.stop()
        self.camera.stop()
        self.talk.cancel()
        self.narrator.stop()
        super().closeEvent(e)

    # ---- 키보드 ----
    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        if k in (Qt.Key.Key_Question, Qt.Key.Key_Slash):
            self.toggle_help()
            return
        if self.help_open:               # 안내 창이 열려 있으면 Esc·Space 등은 창만 닫는다
            if k in (Qt.Key.Key_Escape, Qt.Key.Key_Space, Qt.Key.Key_Return):
                self.toggle_help()
            elif k == Qt.Key.Key_F:
                self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        if k in self.TEST_KEYS:
            if not e.isAutoRepeat():
                self._keys_down.add(self.TEST_KEYS[k])   # 누르고 있는 동안 그림자 유지
                self.on_shadow(self.TEST_KEYS[k])
        elif k == Qt.Key.Key_Space:
            if self.cut.hold:
                self.go(self.index + 1, fade=START_FADE)  # 메인 화면 → 이야기 시작
            elif self.asking:
                self.restart(fade=START_FADE)             # 질문 받기 → 메인 화면
            else:
                self.toggle_pause()
        elif k == Qt.Key.Key_M:
            if self.asking and not e.isAutoRepeat():      # 이야기 중에는 AI가 듣지도 답하지도 않는다
                self.narrator.stop()                      # 자막을 읽던 중이면 멈추고 질문을 듣는다
                self.talk.toggle_record()
        elif k == Qt.Key.Key_Right:
            self.go(self.index + 1)
        elif k == Qt.Key.Key_Left:
            self.go(self.index - 1)
        elif k == Qt.Key.Key_R:
            self.restart(fade=1.0)
        elif k == Qt.Key.Key_N:
            self.renderer.show_narration = not self.renderer.show_narration
        elif k == Qt.Key.Key_S:
            if not e.isAutoRepeat():
                self.toggle_narrator()
        elif k == Qt.Key.Key_C:
            self.toggle_camera()
        elif k == Qt.Key.Key_B:
            if self.watcher.running:
                self.watcher.recapture()
        elif k == Qt.Key.Key_D:
            if self.watcher.running and not e.isAutoRepeat():
                self.watcher.toggle_diag()
        elif k == Qt.Key.Key_V:
            self.fill_screen = not self.fill_screen
        elif k == Qt.Key.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif k == Qt.Key.Key_Escape:
            self.close()

    def keyReleaseEvent(self, e: QKeyEvent) -> None:
        if e.key() in self.TEST_KEYS and not e.isAutoRepeat():
            name = self.TEST_KEYS[e.key()]
            self._keys_down.discard(name)
            self.on_shadow_lost(name)   # 시험용 키를 떼면 그림자를 푼 것
