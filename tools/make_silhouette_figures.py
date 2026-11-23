"""새 인식기(silhouette.py) 발표 그림을 만든다 → presentation/figures/12~15, presentation/data/새_인식기.csv

12는 영상만 있으면 되고, 13~15는 tools/silhouette_eval.py 결과가 필요하다.
실행: .venv/Scripts/python.exe tools/make_silhouette_figures.py [--images]
"""
from __future__ import annotations

import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

import make_figures as F  # noqa: E402  (글꼴·색·저장 함수를 함께 쓴다)
import train_shadow as T
from bookstory.recognizer import PEAK_FLOOR, darkness, screen_region, shadow_select, soft_map
from bookstory.silhouette import MODEL_PATH, NONE, classify, crop_to_screen, find_shadow, make_background, screen_mask
from train_silhouette import Cond

KO = F.KO | {NONE: "없음"}
COND_KO = {"clean": "교란 없음", "mild": "약한 교란", "hard": "심한 교란"}
MODELS = (("old", F.GRAY, "기존 방식"), ("new_svm", F.BLUE, "새 방식 (SVM, 앱에서 사용)"),
          ("new_rf", F.ORANGE, "새 방식 (랜덤포레스트)"))


def fig_before_after() -> None:
    """같은 교란 화면을 기존 방식과 새 방식이 어떻게 보는지. 이름은 전체 영상으로 학습한 모델의 답이다
    (처음 보는 동작이 아니므로 정확도의 근거로 쓰지 않는다)."""
    import joblib
    old_model, new_model = joblib.load(F.ROOT / "assets/models/shadow_svm.joblib"), joblib.load(MODEL_PATH)
    rows = [("moon", "empty"), ("deer", "empty"), ("deer", "pos"), ("rabbit", "pos"), ("bird", "pos")]
    fig, ax = plt.subplots(len(rows), 3, figsize=(13.5, 13.2))
    for r, (a, kind) in enumerate(rows):
        path = F.ROOT / "assets" / "video" / "bright" / f"{a}.mp4"
        v, frames = T.process_video(path, 2), T.read_gray(path, 2)
        empty = np.nonzero(v["label_state"] == 0)[0]
        if len(empty) < 8:
            empty = np.argsort(v["peaks"])[:8]
        c = Cond("hard", 12 * 97 + ord(a[0]))              # 채점의 "심한 교란"과 같은 조건
        bg = make_background([c.noise(frames[i]) for i in empty[:8]])
        g = c(frames[empty[len(empty) // 2] if kind == "empty" else v["index"][len(v["index"]) // 2]])
        screen = screen_region(bg)
        d, peak = darkness(g, bg, screen)
        soft = soft_map(d, peak)
        sel = shadow_select(soft) if peak >= PEAK_FLOOR else None
        old = NONE
        if sel is not None:
            p = old_model.predict_proba(T.shape_features(F.shadow_crop(soft, sel=sel))[None])[0]
            old = str(old_model.classes_[p.argmax()]) if p.max() >= 0.7 else NONE
        s2 = screen_mask(bg)
        s = find_shadow(g, bg, s2)
        new = classify(new_model, crop_to_screen(s.mask, s2))[0] if s.valid else NONE
        F.show(ax[r, 0], g, "녹화 영상 + 합성 교란" if r == 0 else None, 255, 15)
        ax[r, 0].set_ylabel("빈 스크린" if kind == "empty" else KO[a], fontsize=17, color=F.INK)
        for k, (m, name, head) in enumerate(((sel, old, "기존 방식"), (s.mask if s.valid else None, new, "새 방식")), 1):
            F.show(ax[r, k], np.zeros(g.shape) if m is None else (m > 0).astype(np.float32),
                   head if r == 0 else None, 1, 15)
            ok = name == (NONE if kind == "empty" else a)
            ax[r, k].text(0.03, 0.06, f"{'○' if ok else '×'} {KO[name]}", transform=ax[r, k].transAxes,
                          fontsize=17, color="white", weight="bold",
                          bbox=dict(facecolor="#0b0b0b", edgecolor="none", alpha=0.6, pad=3))
    F.title(fig, "같은 화면을 보는 두 방식 (밝은 곳, 학습에 쓴 영상의 예시)", 20)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    F.save(fig, "12_기존과_새_방식_비교")


def rate(rows, m, f) -> float:
    return float(np.mean([f(r[m], r["animal"]) for r in rows])) if rows else 0.0


def fig_accuracy(E) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.2), width_ratios=[1.75, 1])
    pos, empty = [[r for r in E if r["kind"] == k] for k in ("pos", "empty")]
    w = 0.26
    for ax, rows, f, text in ((axs[0], pos, lambda p, a: p == a, "처음 보는 동작의 정답률 (높을수록 좋음)"),
                              (axs[1], empty, lambda p, a: p != NONE, "빈 화면을 그림자로 오인 (낮을수록 좋음)")):
        for k, (m, col, name) in enumerate(MODELS):
            v = [rate([r for r in rows if r["cond"] == c], m, f) for c in COND_KO]
            x = np.arange(3) + (k - 1) * (w + 0.03)
            ax.bar(x, v, w, color=col, label=name)
            for xi, vi in zip(x, v):
                ax.text(xi, vi + 0.012, f"{vi:.1%}", ha="center", fontsize=11.5, color=F.INK)
        ax.set_xticks(range(3), list(COND_KO.values()), fontsize=14, color=F.INK)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        F.style(ax)
        ax.set_title(text, fontsize=15, loc="left")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=14, loc="lower center", ncol=3)
    F.title(fig, f"기존 방식과 새 방식: 동작 하나 빼기 검증 (그림자 {len(pos) // 3}장·빈 화면 {len(empty) // 3}장에 교란 3종)", 20)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    F.save(fig, "13_기존과_새_방식_정확도")


def fig_confusion(E) -> None:
    cols = F.ANIMALS + [NONE]
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.4), layout="constrained")
    for ax, (m, _, name) in zip(axs, MODELS):
        pos = [r for r in E if r["kind"] == "pos"]
        cm = np.array([[sum(r["animal"] == a and r[m] == b for r in pos) for b in cols] for a in F.ANIMALS])
        q = cm / cm.sum(1, keepdims=True)
        ax.imshow(q, cmap=F.SEQ, vmin=0, vmax=1)
        for i in range(5):
            for j in range(6):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                            color="white" if q[i, j] > 0.5 else F.INK)
        ax.set_xticks(range(6), [KO[b] for b in cols], fontsize=14)
        ax.set_yticks(range(5), [KO[a] for a in F.ANIMALS], fontsize=14)
        ax.set_xticks(np.arange(7) - 0.5, minor=True)
        ax.set_yticks(np.arange(6) - 0.5, minor=True)
        ax.grid(which="minor", color=F.SURFACE, lw=3)
        ax.tick_params(which="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("예측", fontsize=14)
        ax.set_title(f"{name}  ·  {np.trace(cm[:, :5]) / cm.sum():.1%}", fontsize=15, pad=10)
    axs[0].set_ylabel("정답", fontsize=14)
    F.title(fig, "혼동 행렬: 동작 하나 빼기 (칸 안 = 프레임 수, 같은 751장을 교란 3종으로 시험)", 20)
    F.save(fig, "14_혼동행렬_새_방식")


def fig_animals(E) -> None:
    fig, ax = plt.subplots(figsize=(14.4, 7.2))
    pos, w = [r for r in E if r["kind"] == "pos"], 0.26
    for k, (m, col, name) in enumerate(MODELS):
        v = [rate([r for r in pos if r["animal"] == a], m, lambda p, t: p == t) for a in F.ANIMALS]
        x = np.arange(5) + (k - 1) * (w + 0.03)
        ax.bar(x, v, w, color=col, label=name)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.012, f"{vi:.0%}", ha="center", fontsize=13, color=F.INK)
    ax.set_xticks(range(5), [KO[a] for a in F.ANIMALS], fontsize=16, color=F.INK)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    F.style(ax)
    ax.legend(frameon=False, fontsize=14, loc="upper left", ncol=3)
    ax.set_title("동물별 정답률: 동작 하나 빼기 (같은 751장을 교란 3종으로 시험)", fontsize=19, loc="left", weight="bold", pad=16)
    fig.tight_layout()
    F.save(fig, "15_동물별_정답률_새_방식")


def table(E) -> None:
    out = []
    for m, _, name in MODELS:
        for c in (*COND_KO, "all"):
            z = [r for r in E if c in ("all", r["cond"])]
            pos, empty, trans = [[r for r in z if r["kind"] == k] for k in ("pos", "empty", "trans")]
            out.append([name, COND_KO.get(c, "합계"), len(pos)]
                       + [round(rate(pos, m, lambda p, a: p == a), 4),
                          round(rate(pos, m, lambda p, a: p not in (a, NONE)), 4),
                          round(rate(pos, m, lambda p, a: p == NONE), 4)]
                       + [round(rate([r for r in pos if r["animal"] == a], m, lambda p, t: p == t), 4)
                          for a in F.ANIMALS]
                       + [sum(r[m] != NONE for r in empty), len(empty),
                          sum(r[m] not in (NONE, r["animal"]) for r in trans), len(trans)])
    F.write_csv("새_인식기.csv", ["방식", "교란", "시험 프레임", "정답률", "다른 동물로 오답", "없음으로 놓침",
                              *[f"{KO[a]} 정답률" for a in F.ANIMALS], "빈 화면 오인식", "빈 화면 수",
                              "전환 오답", "전환 수"], out)
    for r in out:
        print(r)


def main() -> None:
    fig_before_after()
    if "--images" not in sys.argv:
        E = json.loads((F.DATA / "silhouette_eval.json").read_text(encoding="utf-8"))
        fig_accuracy(E)
        fig_confusion(E)
        fig_animals(E)
        table(E)


if __name__ == "__main__":
    main()
