"""발표용 실험: 그림자 인식 모델의 검증 방식별 정확도와, 그림자 분리 방법(Otsu 등)별 비교.

결과는 presentation/data/ 에 저장하고, 그림은 tools/make_figures.py가 그린다.
1. 검증 방식 4가지 x (SVM, 랜덤포레스트), 증강 난수 3개(seed)
   - 무작위 70/30  : 이웃 프레임이 양쪽에 섞여 부풀려진다 (비교용)
   - 시간 70/30    : 영상 앞 70% → 뒤 30%
   - 동작 하나 빼기 : 그림자를 넣었다 뺀 동작을 통째로 하나씩 시험용으로 뺀다
   - 조명 교차     : 밝음 → 어두움, 어두움 → 밝음
2. 그림자 분리 방법 6가지로 같은 프레임을 다시 잘라 학습·시험 (SVM)
   분리에 실패해 조각을 얻지 못한 시험 프레임은 틀린 것으로 센다.

실행: .venv/Scripts/python.exe tools/shadow_experiments.py   (CPU에 따라 몇 분)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import train_shadow as T  # noqa: E402
from bookstory.recognizer import darkness, screen_region, shadow_crop, soft_map  # noqa: E402

OUT = ROOT / "presentation" / "data"
STEP = 2
SEEDS = (0, 1, 2)
MIN_TAKE = 10      # 이보다 짧은 동작 조각은 같은 영상의 앞 동작에 붙인다 (전환 중의 짧은 조각)

# ---------------------------------------------------------------- 그림자 분리 방법
METHODS = ["fixed", "otsu_full", "otsu_screen", "adaptive", "otsu_ratio", "ours"]


def segment(method: str, gray: np.ndarray, bg: np.ndarray, screen: np.ndarray) -> np.ndarray:
    """그림자 지도(0~1). ours만 부드러운 값이고 나머지는 0/1."""
    on = screen > 0
    g8 = np.clip(cv2.GaussianBlur(gray, (0, 0), 1.5), 0, 255).astype(np.uint8)
    if method == "fixed":
        m = (g8 < 128) & on
    elif method == "otsu_full":      # 화면 전체 (스크린 밖의 검은 테두리·방까지 포함)
        t, _ = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = g8 <= t
    elif method == "otsu_screen":    # 스크린 안쪽 픽셀만으로 Otsu
        t, _ = cv2.threshold(g8[on], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (g8 <= t) & on
    elif method == "adaptive":
        a = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 4)
        m = (a > 0) & on
    elif method == "otsu_ratio":     # 배경으로 나눈 비율에 Otsu
        d, peak = darkness(gray, bg, screen)
        d8 = np.clip(d * 255, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(d8[on], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = (d8 > t) & on
    else:
        d, peak = darkness(gray, bg, screen)
        return soft_map(d, peak)
    return m.astype(np.float32)


# ---------------------------------------------------------------- 데이터
def load_video(light: str, animal: str) -> dict:
    cv2.setNumThreads(1)
    path = T.ROOT / "assets" / "video" / light / f"{animal}.mp4"
    v = T.process_video(path, STEP)
    frames = T.read_gray(path, STEP)
    bg = T.estimate_background(frames)
    screen = screen_region(bg)
    by_method = {m: [shadow_crop(segment(m, frames[i], bg, screen)) for i in v["index"]] for m in METHODS}
    takes, last, names = v["take"], None, []
    for t in takes:
        if (takes == t).sum() >= MIN_TAKE or last is None:
            last = t
        names.append(f"{light}-{animal}-{last}")
    return {"crops": v["crops"], "methods": by_method, "take": names, "pos": v["pos"], "index": v["index"],
            "light": light, "animal": animal, "n_frames": v["n_frames"], "n_empty": v["n_empty"],
            "n_dropped": v["n_dropped"]}


def fit_predict(kind: str, crops, y, tr, te, seed: int):
    cv2.setNumThreads(1)
    xa, ya = T.augment(crops[tr], y[tr], np.random.default_rng(seed))
    model = T.make_model(kind).fit(T.featurize(xa), ya)
    proba = model.predict_proba(T.featurize(crops[te]))
    return model.classes_[proba.argmax(1)].astype(str), proba.max(1)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    vids = Parallel(n_jobs=4)(delayed(load_video)(l, a) for l in T.LIGHTS for a in T.ANIMALS)
    crops = np.array([c for v in vids for c in v["crops"]])
    y = np.array([v["animal"] for v in vids for _ in v["crops"]])
    light = np.array([v["light"] for v in vids for _ in v["crops"]])
    pos = np.concatenate([v["pos"] for v in vids])
    take = np.array([t for v in vids for t in v["take"]])
    idx = np.arange(len(y))
    print(f"데이터 {len(y)}장, 동작 {len(set(take))}개  ({time.time() - t0:.0f}s)", flush=True)

    perm = np.random.default_rng(0).permutation(idx)
    cut = int(len(idx) * 0.7)
    splits = {
        "random": [(perm[:cut], perm[cut:])],
        "time": [(idx[pos < 0.7], idx[pos >= 0.7])],
        "take": [(idx[take != t], idx[take == t]) for t in sorted(set(take))],
        "b2d": [(idx[light == "bright"], idx[light == "dark"])],
        "d2b": [(idx[light == "dark"], idx[light == "bright"])],
    }
    jobs = [(n, k, s, f) for n, folds in splits.items() for k in ("svm", "rf") for s in SEEDS
            for f in range(len(folds))]
    res = Parallel(n_jobs=-2, verbose=1)(
        delayed(fit_predict)(k, crops, y, *splits[n][f], s) for n, k, s, f in jobs)
    preds: dict = {}
    for (n, k, s, f), (p, c) in zip(jobs, res):
        e = preds.setdefault((n, k, s), ([], [], []))
        e[0].append(splits[n][f][1]), e[1].append(p), e[2].append(c)
    save = {}
    for (n, k, s), e in preds.items():
        te, p, c = map(np.concatenate, e)
        save[f"{n}|{k}|{s}"] = {"true": y[te].tolist(), "pred": p.tolist(), "conf": c.round(4).tolist(),
                                "take": take[te].tolist()}
        print(f"{n:7s} {k:3s} seed{s}  acc {(p == y[te]).mean():.4f}  n={len(te)}", flush=True)

    # 분리 방법 비교
    mjobs, mdata = [], {}
    for m in METHODS:
        cr = [c for v in vids for c in v["methods"][m]]
        ok = np.array([c is not None for c in cr])
        arr = np.array([c if c is not None else np.zeros((T.SHAPE, T.SHAPE), np.float32) for c in cr])
        mdata[m] = (arr, ok)
        mjobs += [(m, n) for n in ("time", "b2d", "d2b")]

    def one(m, n):
        arr, ok = mdata[m]
        tr, te = splits[n][0]
        tr, te_ok = tr[ok[tr]], te[ok[te]]
        if len(set(y[tr])) < 2 or len(te_ok) == 0:
            return 0.0
        p, _ = fit_predict("svm", arr, y, tr, te_ok, 0)
        return float((p == y[te_ok]).sum() / len(te))
    mres = Parallel(n_jobs=-2)(delayed(one)(m, n) for m, n in mjobs)
    methods = {m: {"found": {l: float(mdata[m][1][light == l].mean()) for l in T.LIGHTS}} for m in METHODS}
    for (m, n), a in zip(mjobs, mres):
        methods[m][n] = a
    for m in METHODS:
        print(m, methods[m], flush=True)

    meta = {f"{v['light']}|{v['animal']}": {k: v[k] for k in ("n_frames", "n_empty", "n_dropped")}
            | {"used": len(v["crops"]), "takes": len(set(v["take"]))} for v in vids}
    (OUT / "experiments.json").write_text(json.dumps(
        {"data": meta, "predictions": save, "methods": methods}, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(OUT / "dataset.npz", crops=(crops * 255).astype(np.uint8), y=y, light=light,
                        take=take, pos=pos)
    print(f"완료 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
