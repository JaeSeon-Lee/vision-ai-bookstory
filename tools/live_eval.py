"""실제 웹캠 진단 기록(D 키)을 채점한다 → presentation/figures/16~18, presentation/data/실제_웹캠.csv

조명을 켜고 스크린에 그림자를 비춘 기록은 1123_150605 하나다. 저장된 프레임을 사람이 보고
정답 구간을 나눴다(LABELS, 프레임 번호는 양 끝 포함). 손이 들어오거나 모양을 만드는 구간은 채점하지 않는다.
앱이 그때 실제로 낸 답(log.csv)을 그대로 채점한다. 다시 인식시키지 않는다.

원본 기록(assets/diagnostics)은 2026-11-23에 지웠고 D 키도 더는 저장하지 않으므로 다시 실행할 수 없다.
채점 방법과 정답 구간을 보여 주려고 남겨 둔다.
"""
from __future__ import annotations

import csv
from collections import Counter, deque

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import make_figures as F
from bookstory.watcher import STABLE_K, STABLE_N

DIAG = F.ROOT / "assets" / "diagnostics"
SESSION = "1123_150605"
# (시작, 끝, 정답). "empty"는 빈 스크린, None은 채점하지 않음
LABELS = [(0, 473, "empty"), (474, 477, None), (478, 582, "bird"), (583, 761, "empty"), (762, 799, None),
          (800, 1050, "rabbit"), (1051, 1256, "empty"), (1257, 1261, None), (1262, 1499, "fox")]
KO = F.KO | {"empty": "빈 스크린"}
CROP = (0.06, 0.10, 0.95, 0.89)     # 스크린 안쪽만 (화면 가장자리에 사람이 비친다)


def read(session: str) -> list[dict]:
    with open(DIAG / session / "log.csv", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def picture(session: str, frame: int) -> np.ndarray:
    im = Image.open(DIAG / session / f"f{frame:05d}.jpg").convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = CROP
    return np.asarray(im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))))


def stable_names(rows: list[dict]) -> list[str]:
    """앱의 8프레임 중 6번 규칙을 기록에 그대로 적용한 결과 (없으면 "")."""
    hist: deque = deque(maxlen=STABLE_N)
    out = []
    for r in rows:
        hist.append(r["name"] if r["name"] not in ("", "none") else None)
        top, n = Counter(hist).most_common(1)[0]
        out.append(top if top and n >= STABLE_K else "")
    return out


def score() -> list[dict]:
    rows = read(SESSION)
    stable = stable_names(rows)
    out = []
    enter = 0       # 손이 스크린에 들어오기 시작한 프레임
    for a, b, truth in LABELS:
        if truth is None:
            enter = a
            continue
        seg = range(a, min(b, len(rows) - 1) + 1)
        raw = [rows[i]["name"] for i in seg]
        animal = [n for n in raw if n not in ("", "none")]
        right = sum(n == truth for n in raw)
        st = [stable[i] for i in seg]
        first = next((i for i in seg if stable[i] == truth), None)
        out.append({"truth": truth, "range": f"{a}-{b}", "n": len(raw), "right": right,
                    "wrong": len(animal) - right, "missed": len(raw) - len(animal),
                    "wrong_names": Counter(n for n in animal if n != truth),
                    "missed_why": Counter((rows[i]["reason"] or "다섯 모양이 아님") for i in seg
                                          if rows[i]["name"] in ("", "none")),
                    "stable_right": sum(s == truth for s in st),
                    "stable_wrong": sum(s not in ("", truth) for s in st),
                    "first_s": None if first is None else float(rows[first]["time"]) - float(rows[enter]["time"])})
    return out


def fig_accuracy(res: list[dict]) -> None:
    z = [r for r in res if r["truth"] != "empty"][::-1]
    fig, ax = plt.subplots(figsize=(14.4, 5.4))
    left = np.zeros(len(z))
    for key, col, name in (("right", F.BLUE, "정답"), ("wrong", F.ORANGE, "다른 동물로 오답"),
                           ("missed", F.GRAY, "놓침 (아무 동물도 아니라고 함)")):
        v = np.array([r[key] / r["n"] for r in z])
        ax.barh(range(len(z)), v, 0.56, left=left, color=col, label=name, edgecolor=F.SURFACE, lw=2)
        for i, (x, w) in enumerate(zip(left, v)):
            if w >= 0.045:
                ax.text(x + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=15,
                        color="white" if key == "right" else F.INK)
        left += v
    ax.set_yticks(range(len(z)), [f"{KO[r['truth']]}\n({r['n']}프레임)" for r in z], fontsize=15, color=F.INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=14, loc="upper center", bbox_to_anchor=(0.45, -0.02), ncol=3)
    ax.set_title("실제 웹캠에서 앱이 낸 답: 조명을 켜고 그림자를 비춘 한 번의 시연", fontsize=19, loc="left",
                 weight="bold", pad=16)
    fig.tight_layout()
    F.save(fig, "16_실제_웹캠_정확도")


def fig_examples() -> None:
    cells = [(530, "새", True), (552, "새", False), (940, "토끼", True), (866, "토끼", False),
             (1340, "여우", True), (1270, "여우", False)]
    rows = read(SESSION)
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for k, (f, who, ok) in enumerate(cells):
        ax = axs[0 if ok else 1, k // 2]
        F.show(ax, picture(SESSION, f), who if ok else None, 255, 17)
        said = rows[f]["name"]
        text = {"none": "다섯 모양이 아님"}.get(said) or KO.get(said, rows[f]["reason"])
        ax.text(0.03, 0.06, f"{'○' if ok else '×'} {text}", transform=ax.transAxes, fontsize=17, color="white",
                weight="bold", bbox=dict(facecolor="#0b0b0b", edgecolor="none", alpha=0.6, pad=4))
    F.title(fig, "실제 웹캠 화면: 맞힌 프레임(위)과 틀린 프레임(아래)", 20)
    fig.tight_layout(rect=(0, 0, 1, 1))
    F.save(fig, "17_실제_웹캠_예")


def fig_conditions() -> None:
    """그림자 인식이 성립하지 않는 화면들 (얼굴이 나온 프레임은 쓰지 않는다)."""
    cells = [(SESSION, 1340, "조명 켬 + 스크린 뒤의 손", "기대한 사용법: 또렷한 그림자"),
             ("1123_150956", 108, "스크린 뒤 조명 없음", "손이 비쳐 보일 뿐 그림자가 흐림"),
             ("1123_151036", 378, "카메라와 스크린 사이의 맨손", "그림자가 아닌데도 '여우'로 답함")]
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
    for ax, (s, f, head, note) in zip(axs, cells):
        F.show(ax, picture(s, f), head, 255, 17)
        name, why = next((r["name"], r["reason"]) for r in read(s) if int(r["frame"]) == f)
        ax.set_xlabel(f"앱의 답: {'다섯 모양이 아님' if name == 'none' else KO.get(name) or '그림자로 인정 안 함 (' + why + ')'}\n{note}",
                      fontsize=14, color=F.INK, labelpad=10)
    F.title(fig, "인식 결과는 촬영 환경에 크게 좌우된다", 20)
    fig.tight_layout(rect=(0, 0, 1, 1))
    F.save(fig, "18_촬영_환경")


def bare_hand() -> None:
    """스크린 앞 맨손에 동물 이름이 나온 프레임 수 (1123_151036, 사람이 확인한 구간)."""
    for s, a, b in (("1123_151036", 364, 402), ("1123_151036", 838, 880)):
        print(s, f"{a}-{b}", dict(Counter(r["name"] or r["reason"] for r in read(s)[a:b + 1])))


def main() -> None:
    res = score()
    for r in res:
        print({k: (dict(v) if isinstance(v, Counter) else v) for k, v in r.items()})
    F.write_csv("실제_웹캠.csv",
                ["정답", "프레임 구간", "프레임 수", "정답률", "다른 동물로 오답", "놓침",
                 "8중 6 규칙 정답 프레임", "8중 6 규칙 오답 프레임", "손이 들어온 뒤 인정까지(초)"],
                [[KO[r["truth"]], r["range"], r["n"],
                  "" if r["truth"] == "empty" else round(r["right"] / r["n"], 4), r["wrong"], r["missed"],
                  "" if r["truth"] == "empty" else r["stable_right"], r["stable_wrong"],
                  "" if r["first_s"] is None else round(r["first_s"], 1)] for r in res])
    pos, empty = [[r for r in res if (r["truth"] == "empty") == e] for e in (False, True)]
    n = sum(r["n"] for r in pos)
    print("합계", sum(r["right"] for r in pos), n, sum(r["right"] for r in pos) / n,
          "동물별 평균", np.mean([r["right"] / r["n"] for r in pos]))
    print("빈 스크린 오인식", sum(r["wrong"] for r in empty), sum(r["n"] for r in empty),
          "8중 6:", sum(r["stable_wrong"] for r in empty))
    bare_hand()
    fig_accuracy(res)
    fig_examples()
    fig_conditions()


if __name__ == "__main__":
    main()
