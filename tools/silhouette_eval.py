"""기존 인식기와 새 인식기를 같은 시험 프레임에서 채점한다 → presentation/data/silhouette_eval.json

동작(그림자를 넣었다 뺀 한 덩어리)을 하나씩 통째로 빼고 학습한 뒤, 뺀 동작만 시험한다.
빈 화면과 들어오고 나가는 프레임은 시간상 가장 가까운 동작에 붙여 한 번씩만 시험한다.
- old     : 기존 앱 방식 (진하기 정규화, "없음" 없음, 확률 0.7 이상만 받음)
- new_svm : 새 앱 방식 (윤곽 특징, "없음" 클래스, 달 원형 규칙)
- new_rf  : new_svm과 특징·규칙이 같고 분류기만 랜덤포레스트

실행: .venv/Scripts/python.exe tools/silhouette_eval.py [--jobs 5]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import train_shadow as T  # noqa: E402
from bookstory.silhouette import NONE, classify  # noqa: E402
from train_silhouette import TEST, build_xy, load, make_svm, training_set  # noqa: E402

OUT = ROOT / "presentation" / "data" / "silhouette_eval.json"


class ForestScores:
    """랜덤포레스트를 classify()가 쓰는 모양으로 감싼다."""

    def __init__(self, x, y) -> None:
        from sklearn.ensemble import RandomForestClassifier
        self.m = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=1, random_state=0).fit(x, y)
        self.classes_ = self.m.classes_

    def decision_function(self, x):
        return self.m.predict_proba(x)


def low_priority() -> None:
    """앱을 쓰는 동안에도 느려지지 않게 한다 (Windows, 자식 프로세스도 물려받음)."""
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass(ctypes.c_void_p(k.GetCurrentProcess()), 0x00000040)   # IDLE


def fold(data: list[dict], take: str) -> list[dict]:
    cv2.setNumThreads(1)
    tr = training_set(data, take)
    x, y = build_xy(tr)
    svm, rf = make_svm().fit(x, y), ForestScores(x, y)
    old_tr = [d for d in data if d["kind"] == "pos" and d["take"] != take and d["cond"] == "clean"
              and d["old"] is not None]
    xa, ya = T.augment(np.array([d["old"] for d in old_tr]), np.array([d["animal"] for d in old_tr]),
                       np.random.default_rng(0))
    old = T.make_model("svm").fit(T.featurize(xa), ya)
    out = []
    for d in data:
        if d["take"] != take or (d["cond"], d["seed"]) not in TEST:
            continue
        r = {k: d[k] for k in ("kind", "cond", "animal", "light", "take")}
        r["old"] = NONE
        if d["old"] is not None:
            p = old.predict_proba(T.shape_features(d["old"])[None])[0]
            if p.max() >= 0.7:
                r["old"] = str(old.classes_[p.argmax()])
        for name, m in (("new_svm", svm), ("new_rf", rf)):
            r[name] = classify(m, d["mask"])[0] if d["mask"] is not None else NONE
        out.append(r)
    print(take, len(out), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    n = ap.parse_args().jobs
    low_priority()
    data = sum(Parallel(n_jobs=4)(delayed(load)(light, a) for light in T.LIGHTS for a in T.ANIMALS), [])
    takes = sorted({d["take"] for d in data if d["kind"] == "pos"})
    rows = sum(Parallel(n_jobs=n)(delayed(fold)(data, t) for t in takes), [])
    OUT.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {OUT}  {len(rows)}행", flush=True)
    for m in ("old", "new_svm", "new_rf"):
        pos = [r for r in rows if r["kind"] == "pos"]
        line = [f"{c} {np.mean([r[m] == r['animal'] for r in pos if r['cond'] == c]):.1%}"
                for c in ("clean", "mild", "hard")]
        line.append(f"오답 {np.mean([r[m] not in (r['animal'], NONE) for r in pos]):.1%}")
        line += [f"{T.KO[a]} {np.mean([r[m] == a for r in pos if r['animal'] == a]):.1%}" for a in T.ANIMALS]
        z = [r for r in rows if r["kind"] == "empty"]
        line.append(f"빈 화면 오인식 {sum(r[m] != NONE for r in z)}/{len(z)}")
        z = [r for r in rows if r["kind"] == "trans"]
        line.append(f"전환 오답 {sum(r[m] not in (NONE, r['animal']) for r in z)}/{len(z)}")
        print(m, " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
