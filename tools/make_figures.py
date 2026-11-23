"""발표용 그림을 만든다 → presentation/figures/ (정리된 그림), presentation/images/ (낱장 이미지)

먼저 tools/shadow_experiments.py 를 돌려 presentation/data/ 가 있어야 한다 (이미지만이면 --images).
실행: .venv/Scripts/python.exe tools/make_figures.py [--images | --charts]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import train_shadow as T  # noqa: E402
from bookstory.recognizer import (PEAK_FLOOR, darkness, hog, screen_region, shadow_crop,  # noqa: E402
                                  shadow_select, soft_map)
from shadow_experiments import METHODS, SEEDS, segment  # noqa: E402

P = ROOT / "presentation"
FIG, IMG, DATA = P / "figures", P / "images", P / "data"
KO, ANIMALS, LIGHTS = T.KO, T.ANIMALS, T.LIGHTS
LIGHT_KO = {"bright": "밝은 곳", "dark": "어두운 곳"}
METHOD_KO = {"fixed": "고정 문턱값\n(128)", "otsu_full": "Otsu\n(화면 전체)", "otsu_screen": "Otsu\n(스크린 안)",
             "adaptive": "적응형\n문턱값", "otsu_ratio": "배경 비율\n+ Otsu", "ours": "배경 비율 +\n진하기 정규화 (1차)"}
SPLIT_KO = {"random": "무작위 70/30\n(부풀려짐)", "time": "시간 70/30", "take": "동작 하나 빼기",
            "b2d": "밝음 → 어두움", "d2b": "어두움 → 밝음"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b9b8b2"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#0d366b"])
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 13,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=150)
    plt.close(fig)
    print("figures/" + name + ".png", flush=True)


def title(fig, text: str, size: int = 21) -> None:
    fig.suptitle(text, fontsize=size, x=0.02, ha="left", weight="bold")


def imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())


def show(ax, img, label=None, vmax=1.0, size=13):
    """흑백은 vmax를 흰색으로, 컬러는 그대로."""
    ax.imshow(img) if img.ndim == 3 else ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(GRID)
    if label:
        ax.set_title(label, fontsize=size, color=INK)


class Sample:
    """영상 하나의 대표 프레임(가장 긴 동작의 가운데)과 빈 화면."""

    def __init__(self, light: str, animal: str):
        path = ROOT / "assets" / "video" / light / f"{animal}.mp4"
        v = T.process_video(path, 2)
        frames = T.read_gray(path, 2)
        tk = v["take"]
        big = max(set(tk.tolist()), key=lambda t: (tk == t).sum())
        use = v["index"][tk == big]
        self.i = int(use[len(use) // 2])
        self.bg = T.estimate_background(frames)
        self.screen = screen_region(self.bg)
        self.gray = frames[self.i]
        empty = np.nonzero(v["label_state"] == 0)[0]
        self.empty = frames[int(empty[np.argmin(v["peaks"][empty])])] if len(empty) else None
        cap = cv2.VideoCapture(str(path))
        for _ in range(self.i * 2 + 1):      # 건너뛰기 없이 읽어야 같은 프레임이 나온다
            _, f = cap.read()
        cap.release()
        self.color = cv2.cvtColor(cv2.resize(f, self.gray.shape[::-1], interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2RGB)
        self.d, self.peak = darkness(self.gray, self.bg, self.screen)
        self.soft = soft_map(self.d, self.peak)
        self.sel = shadow_select(self.soft)
        self.crop = shadow_crop(self.soft, sel=self.sel)

    def seg(self, method: str, gray=None) -> np.ndarray:
        g = self.gray if gray is None else gray
        s = segment(method, g, self.bg, self.screen)
        if method == "ours":   # 1차 인식기와 같이: 배경과 차이가 거의 없으면 그림자 없음
            sel = shadow_select(s)
            _, peak = darkness(g, self.bg, self.screen)
            return (sel if sel is not None and peak >= PEAK_FLOOR else np.zeros(s.shape)).astype(np.float32)
        return s

    def overlay(self) -> np.ndarray:
        out = self.color.copy()
        cs, _ = cv2.findContours(self.sel.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        return cv2.drawContours(out, cs, -1, (235, 104, 52), 2, cv2.LINE_AA)


def hog_picture(crop: np.ndarray, px: int = 60) -> np.ndarray:
    """8x8 칸마다 윤곽 방향을 선으로 (밝을수록 강함). 묶음 계산상 마지막 줄·칸은 뺀다."""
    cells = hog((crop * 255).astype(np.uint8)).reshape(7, 7, 4, 9)[:, :, 0]
    big = cv2.resize(crop[:56, :56], (7 * px, 7 * px)) * 70
    canvas = np.repeat(big[..., None], 3, 2).astype(np.uint8)
    for r in range(7):
        for c in range(7):
            for b, v in enumerate(cells[r, c]):
                if v < 0.03:
                    continue
                th = np.radians(b * 20 + 90)       # 기울기에 수직 = 윤곽 방향
                half = px * 0.45 * v / 0.2
                dx, dy = np.cos(th) * half, np.sin(th) * half
                x, y = (c + 0.5) * px, (r + 0.5) * px
                col = tuple(int(k * (0.4 + 0.6 * v / 0.2)) for k in (255, 200, 90))
                cv2.line(canvas, (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy)), col, 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------- 이미지 그림
def fig_pipeline(S):
    fig, ax = plt.subplots(2, 7, figsize=(19.2, 4.9), width_ratios=[1.75] * 5 + [1.1, 1.1])
    for r, light in enumerate(("dark", "bright")):
        s = S[light, "deer"]
        cols = [(s.color, "① 원본", 1), (s.gray, "② 흑백", 255), (s.bg, "③ 빈 스크린 배경", 255),
                (s.d, "④ 배경보다 어두운 정도", 0.85), (s.soft, "⑤ 진하기 정규화", 1),
                (s.crop, "⑥ 64×64로 자르기", 1), (hog_picture(s.crop), "⑦ 윤곽 방향 (HOG)", 1)]
        for c, (im, label, vmax) in enumerate(cols):
            show(ax[r, c], im, label if r == 0 else None, vmax)
        ax[r, 3].text(0.97, 0.08, f"최대 {s.peak:.0%}", transform=ax[r, 3].transAxes, ha="right",
                      color="white", fontsize=14)
        ax[r, 0].set_ylabel(LIGHT_KO[light], fontsize=17, color=INK)
    title(fig, "그림자 전처리 과정 (사슴)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "01_전처리_과정")


def fig_otsu_hist(S):
    fig, ax = plt.subplots(2, 4, figsize=(19.2, 7.4))
    for r, light in enumerate(("dark", "bright")):
        s = S[light, "bird"]
        on = s.screen > 0
        g8 = np.clip(cv2.GaussianBlur(s.gray, (0, 0), 1.5), 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(g8[on], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        show(ax[r, 0], s.color, "원본", size=15)
        ax[r, 0].set_ylabel(LIGHT_KO[light], fontsize=18, color=INK)
        a = ax[r, 1]
        a.hist(g8[on], bins=np.arange(0, 257, 4), color=BLUE, rwidth=0.85)
        a.axvline(t, color=ORANGE, lw=2)
        a.annotate(f"Otsu 문턱값 = {t:.0f}", (t, a.get_ylim()[1] * 0.93), xytext=(8 if t < 170 else -8, 0),
                   textcoords="offset points", ha="left" if t < 170 else "right", fontsize=14, color=INK)
        a.set_title("스크린 안 밝기 분포", fontsize=15)
        a.set_yticks([])
        a.set_xlim(0, 256)
        a.set_xlabel("밝기 (0 = 검정, 255 = 흰색)")
        a.spines["left"].set_visible(False)
        show(ax[r, 2], s.seg("otsu_screen"), "Otsu 이진화 결과", size=15)
        show(ax[r, 3], s.seg("ours"), "1차 방법 결과", size=15)
    title(fig, "Otsu 이진화: 어두운 곳과 밝은 곳 (새)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "02_Otsu_히스토그램")


def fig_methods(S):
    rows = [(S["dark", "bird"], "새\n어두운 곳", None), (S["bright", "bird"], "새\n밝은 곳", None),
            (S["dark", "fox"], "여우\n어두운 곳", None), (S["bright", "fox"], "여우\n밝은 곳", None),
            (S["bright", "rabbit"], "빈 스크린\n밝은 곳", S["bright", "rabbit"].empty)]
    fig, ax = plt.subplots(len(rows), 7, figsize=(19.2, 9.2))
    for r, (s, name, g) in enumerate(rows):
        show(ax[r, 0], s.gray if g is None else g, "흑백 원본" if r == 0 else None, 255, 14)
        ax[r, 0].set_ylabel(name, fontsize=15, color=INK)
        for c, m in enumerate(METHODS, 1):
            show(ax[r, c], s.seg(m, g), METHOD_KO[m] if r == 0 else None, 1, 14)
    title(fig, "그림자 분리 방법 비교 (흰색 = 그림자로 판단한 곳)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "03_분리_방법_비교")


def fig_all(S):
    for k, light in enumerate(("dark", "bright")):
        fig, ax = plt.subplots(3, 5, figsize=(16, 6.9))
        for c, a in enumerate(ANIMALS):
            s = S[light, a]
            show(ax[0, c], s.color, KO[a], size=18)
            show(ax[1, c], s.seg("otsu_screen"))
            show(ax[2, c], s.seg("ours"))
        for j, n in enumerate(("원본", "Otsu", "1차 방법")):
            ax[j, 0].set_ylabel(n, fontsize=15, color=INK)
        title(fig, f"동물별 그림자 분리 결과: {LIGHT_KO[light]}", 20)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"04{'ab'[k]}_동물별_분리_{'어두운곳' if light == 'dark' else '밝은곳'}")


def fig_augment(S):
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(2, 4, figsize=(12.8, 7.2))
    for r, a in enumerate(("rabbit", "fox")):
        x, _ = T.augment(S["dark", a].crop[None], np.array([a]), rng)
        for j, (im, t) in enumerate(zip(x, ("원본", "좌우 반전", "회전·크기", "반전 + 회전·크기"))):
            show(ax[r, j], im, t if r == 0 else None, size=15)
        ax[r, 0].set_ylabel(KO[a], fontsize=16, color=INK)
    title(fig, "학습 데이터 늘리기 (1장 → 4장)", 19)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "05_데이터_증강")


def loose_images(S):
    def u8(m):
        return (np.clip(m, 0, 1) * 255).astype(np.uint8)

    for (light, a), s in S.items():
        d = IMG / f"{light}_{a}"
        imwrite(d / "1_original.png", cv2.cvtColor(s.color, cv2.COLOR_RGB2BGR))
        imwrite(d / "2_gray.png", s.gray.astype(np.uint8))
        imwrite(d / "3_background.png", s.bg.astype(np.uint8))
        for m in METHODS:
            imwrite(d / f"4_{m}.png", u8(s.seg(m)))
        imwrite(d / "5_darkness_normalized.png", u8(s.soft))
        imwrite(d / "6_contour.png", cv2.cvtColor(s.overlay(), cv2.COLOR_RGB2BGR))
        imwrite(d / "7_crop64.png", cv2.resize(u8(s.crop), (512, 512), interpolation=cv2.INTER_NEAREST))
        imwrite(d / "8_hog.png", cv2.cvtColor(hog_picture(s.crop), cv2.COLOR_RGB2BGR))
    print(f"images/ 낱장 {len(list(IMG.rglob('*.png')))}개")


# ---------------------------------------------------------------- 결과 차트
def arrs(E, n, kind, seed=0):
    r = E["predictions"][f"{n}|{kind}|{seed}"]
    return np.array(r["true"]), np.array(r["pred"]), np.array(r["conf"]), np.array(r["take"])


def acc(E, n, kind, seed=0) -> float:
    t, p, *_ = arrs(E, n, kind, seed)
    return float((t == p).mean())


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)


def fig_counts(E):
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    w = 0.34
    for k, (l, col) in enumerate((("bright", BLUE), ("dark", ORANGE))):
        x = np.arange(5) + (k - 0.5) * (w + 0.03)
        v = [E["data"][f"{l}|{a}"]["used"] for a in ANIMALS]
        tk = [E["data"][f"{l}|{a}"]["takes"] for a in ANIMALS]
        ax.bar(x, v, w, color=col, label=LIGHT_KO[l])
        for xi, vi, ti in zip(x, v, tk):
            ax.text(xi, vi + 6, f"{vi}장\n(동작 {ti})", ha="center", fontsize=12.5, color=INK)
    n = sum(d["used"] for d in E["data"].values())
    t = sum(d["takes"] for d in E["data"].values())
    ax.set_xticks(range(5), [KO[a] for a in ANIMALS], fontsize=16, color=INK)
    ax.set_ylim(0, 530)
    style(ax)
    ax.legend(frameon=False, fontsize=14, loc="upper left")
    ax.set_title(f"학습 데이터: 프레임 {n:,}장, 실제 동작은 {t}개", fontsize=19, loc="left", weight="bold", pad=16)
    fig.tight_layout()
    save(fig, "06_데이터_수")


def fig_accuracy(E):
    order, w = list(SPLIT_KO), 0.36
    fig, ax = plt.subplots(figsize=(14.4, 7.2))
    for k, (kind, col, name) in enumerate((("svm", BLUE, "SVM (1차)"), ("rf", ORANGE, "랜덤포레스트"))):
        v = [np.mean([acc(E, n, kind, s) for s in SEEDS]) for n in order]
        x = np.arange(len(order)) + (k - 0.5) * (w + 0.03)
        ax.bar(x, v, w, color=col, label=name)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.012, f"{vi:.1%}", ha="center", fontsize=14, color=INK)
    ax.set_xticks(range(len(order)), [SPLIT_KO[n] for n in order], fontsize=14, color=INK)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    style(ax)
    ax.legend(frameon=False, fontsize=14, loc="upper right", ncol=2)
    ax.set_title("검증 방식별 정확도 (난수 3회 평균)", fontsize=19, loc="left", weight="bold", pad=16)
    fig.tight_layout()
    save(fig, "07_검증_방식별_정확도")


def fig_confusion(E, kind, name, text):
    fig, axs = plt.subplots(1, 4, figsize=(21, 7.0), layout="constrained")
    for ax, n in zip(axs, ("time", "take", "b2d", "d2b")):
        t, p, *_ = arrs(E, n, kind)
        cm = np.array([[int(((t == a) & (p == b)).sum()) for b in ANIMALS] for a in ANIMALS])
        r = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(r, cmap=SEQ, vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                            color="white" if r[i, j] > 0.5 else INK)
        names = [KO[a] for a in ANIMALS]
        ax.set_xticks(range(5), names, fontsize=14)
        ax.set_yticks(range(5), names, fontsize=14)
        ax.set_xticks(np.arange(6) - 0.5, minor=True)
        ax.set_yticks(np.arange(6) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, lw=3)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("예측", fontsize=14)
        ax.set_title(f"{SPLIT_KO[n]}  ·  {(t == p).mean():.1%}", fontsize=16, pad=10)
    axs[0].set_ylabel("정답", fontsize=14)
    title(fig, text + " (칸 안 = 프레임 수, 난수 첫 회 기준)", 20)
    save(fig, name)


def fig_methods_acc(E):
    M = E["methods"]
    fig, axs = plt.subplots(1, 3, figsize=(19.2, 7.6), sharey=True)
    y = np.arange(len(METHODS))[::-1]
    for ax, n in zip(axs, ("time", "b2d", "d2b")):
        v = [M[m][n] for m in METHODS]
        ax.barh(y, v, 0.55, color=[BLUE if m == "ours" else GRAY for m in METHODS])
        for yi, vi, m in zip(y, v, METHODS):
            text = f"{vi:.1%}"      # 0%는 못 맞힌 것이 아니라 그림자를 하나도 찾지 못한 것이다
            if n == "b2d" and M[m]["found"]["bright"] == 0:
                text = "학습할 그림자를 찾지 못함"
            elif n == "d2b" and M[m]["found"]["bright"] == 0:
                text = "시험 그림자를 찾지 못함 (0%)"
            ax.text(vi + 0.015, yi, text, va="center", fontsize=14, color=INK)
        ax.set_yticks(y, [METHOD_KO[m] for m in METHODS], fontsize=14, color=INK)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_visible(False)
        ax.set_title(SPLIT_KO[n], fontsize=16)
    title(fig, "그림자 분리 방법에 따른 인식 정확도 (SVM, 같은 프레임, 난수 첫 회 기준)", 20)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "10_분리_방법별_정확도")


def fig_errors(E):
    z = np.load(DATA / "dataset.npz")
    t, p, _, take = arrs(E, "take", "svm")
    # 시험 순서 = 동작 이름순, 동작 안에서는 원래 순서
    order = np.concatenate([np.nonzero(z["take"] == k)[0] for k in sorted(set(z["take"].tolist()))])
    assert (z["y"][order] == t).all() and (z["take"][order] == take).all()
    groups: dict = {}
    for w in np.nonzero(t != p)[0]:
        groups.setdefault((str(take[w]), str(p[w])), []).append(w)
    top = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]
    fig, axs = plt.subplots(2, 4, figsize=(14.4, 9.0), layout="constrained")
    for ax in axs.flat:
        ax.axis("off")
    for ax, ((tk, pr), ws) in zip(axs.flat, top):
        ax.axis("on")
        light, a, _ = tk.split("-")
        show(ax, z["crops"][order[ws[len(ws) // 2]]],
             f"{KO[a]} → {KO[pr]}(으)로 오인, {len(ws)}장\n{LIGHT_KO[light]}", 255, 14)
    title(fig, "틀린 예: 동작 하나 빼기 검증 (SVM)", 19)
    save(fig, "11_틀린_예")


def write_csv(name: str, head: list, rows: list) -> None:
    with open(DATA / name, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows([head, *rows])


def tables(E):
    flat = {k: v.replace("\n", " ") for k, v in SPLIT_KO.items()}
    total, per = [], []
    for n in SPLIT_KO:
        for kind in ("svm", "rf"):
            v = [acc(E, n, kind, s) for s in SEEDS]
            t, p, *_ = arrs(E, n, kind)
            f1s = []
            for a in ANIMALS:
                tp = int(((t == a) & (p == a)).sum())
                pr, rc = tp / max(1, (p == a).sum()), tp / max(1, (t == a).sum())
                f1s.append(2 * pr * rc / max(1e-9, pr + rc))
                per.append([flat[n], kind, KO[a], round(float(pr), 4), round(float(rc), 4), round(float(f1s[-1]), 4), int((t == a).sum())])
            total.append([flat[n], kind, len(t)] + [round(float(x), 4) for x in (np.mean(v), min(v), max(v), np.mean(f1s))])
    write_csv("정확도.csv", ["검증 방식", "모델", "시험 프레임", "정확도(3회 평균)", "최소", "최대", "macro F1(첫 회)"], total)
    write_csv("동물별.csv", ["검증 방식", "모델", "동물", "정밀도", "재현율", "F1", "시험 프레임"], per)
    write_csv("분리방법.csv", ["방법", "시간 70/30", "밝음→어두움", "어두움→밝음"],
              [[METHOD_KO[m].replace("\n", " ")] + [round(E["methods"][m][n], 4) for n in ("time", "b2d", "d2b")]
               for m in METHODS])
    for row in total + [r for r in per if r[0] == flat["take"]]:
        print(row)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if "--charts" not in sys.argv:
        S = {(l, a): Sample(l, a) for l in LIGHTS for a in ANIMALS}
        for f in (fig_pipeline, fig_otsu_hist, fig_methods, fig_all, fig_augment, loose_images):
            f(S)
    if "--images" not in sys.argv:
        E = json.loads((DATA / "experiments.json").read_text(encoding="utf-8"))
        fig_counts(E)
        fig_accuracy(E)
        fig_confusion(E, "svm", "08_혼동행렬_SVM", "혼동 행렬: SVM (1차 모델)")
        fig_confusion(E, "rf", "09_혼동행렬_랜덤포레스트", "혼동 행렬: 랜덤포레스트")
        fig_methods_acc(E)
        fig_errors(E)
        tables(E)


if __name__ == "__main__":
    main()
