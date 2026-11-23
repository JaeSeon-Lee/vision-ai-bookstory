"""그림자 영상(assets/video/{bright,dark}/<동물>.mp4)으로 그림자 모양 분류기를 학습하고 평가한다.
(1차 방식. 앱이 쓰는 모델은 tools/train_silhouette.py)

순서
1. 영상마다 배경(빈 스크린)을 픽셀별로 가장 밝은 편(95번째 백분위) 값으로 추정한다.
   (그림자는 어둡게만 하므로, 그림자가 영상 대부분에 떠 있어도 오염되지 않는다)
2. 프레임을 배경에 맞춰 정렬하고(손떨림 보정, 확실할 때만), 그림자 지도 → 64x64 조각을 만든다.
3. 라벨: 그림자가 충분히 진하고 크게 보이는 프레임 = 그 영상의 동물.
   그림자가 들어오고 나가는 경계 프레임은 애매하므로 뺀다.
4. 평가 (같은 영상의 이웃 프레임은 거의 똑같아서, 무작위로 나누면 정확도가 부풀려진다)
   - 조명 교차: 밝은 곳으로 학습 → 어두운 곳으로 시험, 그리고 반대
   - 시간 분할: 각 영상의 앞 70%로 학습 → 뒤 30%로 시험
5. 전체 데이터로 최종 모델을 학습해 assets/models/shadow_svm.joblib 로 저장한다.

실행: .venv/Scripts/python.exe tools/train_shadow.py [--step 2] [--sheet 출력폴더]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bookstory.recognizer import (MODEL_PATH, PEAK_FLOOR, SHAPE, darkness, screen_region,  # noqa: E402
                                  shadow_crop, shape_features, soft_map, to_gray)

ANIMALS = ["bird", "deer", "fox", "moon", "rabbit"]
LIGHTS = ["bright", "dark"]
KO = {"bird": "새", "deer": "사슴", "fox": "여우", "moon": "달", "rabbit": "토끼"}
MAX_SHIFT = 12      # 손떨림 보정 최대 이동(픽셀, 480폭 기준)
EDGE_GAP = 4        # 그림자 있음/없음이 바뀌는 곳 앞뒤로 이만큼(샘플 수)은 버린다


def read_gray(path: Path, step: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            out.append(to_gray(f))
        i += 1
    cap.release()
    return out


def grad(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2)


def estimate_background(frames: list[np.ndarray]) -> np.ndarray:
    st = np.stack(frames)
    rough = np.percentile(st, 90, axis=0)
    _, s0 = cv2.threshold(rough.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s0 = cv2.erode(s0, np.ones((15, 15), np.uint8)) > 0
    # 그림자는 화면을 어둡게만 하므로, 노출을 맞춘 뒤 픽셀마다 가장 밝은 편(95번째 백분위)을 고르면
    # 빈 스크린이 된다. ("균일한 프레임"을 고르는 방식은 손이 있는 프레임이 섞여 들어갔다 — 밝은 곳 여우)
    gain = np.array([np.percentile(f[s0], 80) for f in st])
    norm = st * (np.median(gain) / np.maximum(gain, 1))[:, None, None]
    return cv2.GaussianBlur(np.percentile(norm, 95, axis=0).astype(np.float32), (0, 0), 2)


def process_video(path: Path, step: int) -> dict:
    frames = read_gray(path, step)
    bg = estimate_background(frames)
    screen = screen_region(bg)
    gbg = grad(bg)
    win = cv2.createHanningWindow(bg.shape[::-1], cv2.CV_32F)
    peaks, areas, shifts, crops = [], [], [], []
    for g in frames:
        (dx, dy), resp = cv2.phaseCorrelate(gbg, grad(g), win)
        shift = float(np.hypot(dx, dy))
        if resp >= 0.3 and 0.5 < shift <= MAX_SHIFT:   # 확실할 때만 보정 (그림자 윤곽에 속기 쉽다)
            g = cv2.warpAffine(g, np.float32([[1, 0, -dx], [0, 1, -dy]]), g.shape[::-1],
                               borderMode=cv2.BORDER_REPLICATE)
        d, peak = darkness(g, bg, screen)
        soft = soft_map(d, peak)
        areas.append(float((soft[screen > 0] > 0.35).mean()) if peak >= PEAK_FLOOR else 0.0)
        peaks.append(peak)
        shifts.append(shift)
        crops.append(shadow_crop(soft) if peak >= PEAK_FLOOR else None)
    peaks, areas, shifts = map(np.array, (peaks, areas, shifts))

    # 라벨: 0 = 그림자 없음, 1 = 그림자 있음, -1 = 애매(버림)
    ref = np.percentile(peaks, 90)
    state = np.full(len(peaks), -1)
    state[peaks < max(0.3 * ref, PEAK_FLOOR)] = 0
    present = peaks >= max(0.5 * ref, PEAK_FLOOR * 1.5)
    if present.any():
        med_area = np.median(areas[present])
        present &= areas >= 0.5 * med_area
    state[present] = 1
    keep = state.copy()
    for i in range(1, len(state)):
        if state[i] != state[i - 1]:
            keep[max(0, i - EDGE_GAP):i + EDGE_GAP] = -1
    use = [i for i in range(len(keep)) if keep[i] == 1 and crops[i] is not None]
    # 동작(take): 그림자를 넣었다 뺄 때까지의 한 덩어리. 사이가 0.5초 넘게 비면 다른 동작으로 본다
    takes, gap = [], max(1, round(15 / step))
    for k, i in enumerate(use):
        takes.append(0 if k == 0 else takes[-1] + (i - use[k - 1] > gap))
    return {"crops": [crops[i] for i in use], "pos": np.array(use) / max(1, len(frames) - 1),
            "index": np.array(use), "take": np.array(takes), "peaks": peaks, "label_state": keep,
            "n_frames": len(frames), "n_empty": int((keep == 0).sum()),
            "n_dropped": int(len(frames) - len(use) - (keep == 0).sum())}


def augment(crops: np.ndarray, labels: np.ndarray, rng: np.random.Generator):
    """좌우 반전(다른 손으로 만든 그림자) + 작은 회전·크기 변화."""
    xs, ys = [crops], [labels]
    flipped = crops[:, :, ::-1]
    xs.append(flipped)
    ys.append(labels)
    for base in (crops, flipped):
        out = []
        for c in base:
            ang = rng.uniform(-15, 15)
            sc = rng.uniform(0.9, 1.05)
            M = cv2.getRotationMatrix2D((SHAPE / 2, SHAPE / 2), ang, sc)
            out.append(cv2.warpAffine(c, M, (SHAPE, SHAPE)))
        xs.append(np.array(out))
        ys.append(labels)
    return np.concatenate(xs), np.concatenate(ys)


def featurize(crops: np.ndarray) -> np.ndarray:
    return np.stack([shape_features(c) for c in crops])


def make_model(kind: str = "svm"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if kind == "svm":
        from sklearn.calibration import CalibratedClassifierCV
        svc = SVC(C=10, gamma="scale", class_weight="balanced", random_state=0)
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(svc, ensemble=False))
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=0)


def evaluate(name: str, tr: np.ndarray, te: np.ndarray, data: dict, kind: str, rng) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    Xc, y = data["crops"], data["labels"]
    xa, ya = augment(Xc[tr], y[tr], rng)
    model = make_model(kind).fit(featurize(xa), ya)
    Xte = featurize(Xc[te])
    proba = model.predict_proba(Xte)
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)
    sure = conf >= 0.6
    return {
        "name": name, "model": kind, "n_train": int(len(tr)), "n_train_aug": int(len(ya)), "n_test": int(len(te)),
        "accuracy": float(accuracy_score(y[te], pred)),
        "macro_f1": float(f1_score(y[te], pred, average="macro")),
        "per_class_recall": {a: float((pred[y[te] == a] == a).mean()) if (y[te] == a).any() else None
                             for a in ANIMALS},
        "confusion": confusion_matrix(y[te], pred, labels=ANIMALS).tolist(),
        "coverage@0.6": float(sure.mean()),
        "accuracy@0.6": float((pred[sure] == y[te][sure]).mean()) if sure.any() else None,
    }


def contact_sheet(data: dict, out: Path, per: int = 12) -> None:
    rows = []
    for light in LIGHTS:
        for a in ANIMALS:
            idx = np.nonzero((data["labels"] == a) & (data["light"] == light))[0]
            pick = idx[np.linspace(0, len(idx) - 1, per).astype(int)] if len(idx) else []
            tiles = [(data["crops"][i] * 255).astype(np.uint8) for i in pick]
            tiles += [np.zeros((SHAPE, SHAPE), np.uint8)] * (per - len(tiles))
            row = np.hstack(tiles)
            cv2.putText(row, f"{light} {a}", (2, 10), 0, 0.35, 160, 1)
            rows.append(row)
    cv2.imwrite(str(out), cv2.resize(np.vstack(rows), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def fmt_report(results: list[dict], counts: dict) -> str:
    L = ["# 그림자 모양 인식 학습 결과", "",
         f"생성: {time.strftime('%Y-%m-%d %H:%M')} · tools/train_shadow.py", "",
         "## 데이터 (영상에서 뽑은 프레임 수)", "",
         "| 조명 | " + " | ".join(KO[a] for a in ANIMALS) + " |",
         "|---|" + "---:|" * len(ANIMALS)]
    for light in LIGHTS:
        L.append(f"| {light} | " + " | ".join(str(counts[light][a]) for a in ANIMALS) + " |")
    L += ["", "## 평가", "",
          "| 실험 | 모델 | 학습 | 시험 | 정확도 | macro F1 | 확신≥0.6 비율 | 그때 정확도 |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        a6 = f"{r['accuracy@0.6']:.1%}" if r["accuracy@0.6"] is not None else "-"
        L.append(f"| {r['name']} | {r['model']} | {r['n_train']} | {r['n_test']} | {r['accuracy']:.1%} | "
                 f"{r['macro_f1']:.3f} | {r['coverage@0.6']:.1%} | {a6} |")
    for r in results:
        if r["model"] != "svm":
            continue
        L += ["", f"### {r['name']} (SVM) — 혼동 행렬 (행 = 정답, 열 = 예측)", "",
              "| 정답＼예측 | " + " | ".join(KO[a] for a in ANIMALS) + " | 재현율 |",
              "|---|" + "---:|" * (len(ANIMALS) + 1)]
        for a, row in zip(ANIMALS, r["confusion"]):
            rec = r["per_class_recall"][a]
            L.append(f"| {KO[a]} | " + " | ".join(str(v) for v in row) +
                     (f" | {rec:.1%} |" if rec is not None else " | - |"))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=2, help="몇 프레임마다 하나씩 쓸지")
    ap.add_argument("--sheet", type=Path, help="라벨 확인용 그림자 조각 모음 이미지를 저장할 폴더")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    crops, labels, light_of, pos, counts = [], [], [], [], {l: {} for l in LIGHTS}
    for light in LIGHTS:
        for a in ANIMALS:
            t0 = time.time()
            v = process_video(ROOT / "assets" / "video" / light / f"{a}.mp4", args.step)
            crops += v["crops"]
            labels += [a] * len(v["crops"])
            light_of += [light] * len(v["crops"])
            pos += list(v["pos"])
            counts[light][a] = len(v["crops"])
            print(f"{light:6s} {a:6s} 프레임 {v['n_frames']:4d} → 그림자 {len(v['crops']):4d}, "
                  f"빈 화면 {v['n_empty']:4d}, 버림 {v['n_dropped']:4d}  ({time.time() - t0:.1f}s)")
    data = {"crops": np.array(crops), "labels": np.array(labels), "light": np.array(light_of),
            "pos": np.array(pos)}
    if args.sheet:
        args.sheet.mkdir(parents=True, exist_ok=True)
        contact_sheet(data, args.sheet / "shadow_dataset.png")

    idx = np.arange(len(labels))
    bright, dark = idx[data["light"] == "bright"], idx[data["light"] == "dark"]
    early, late = idx[data["pos"] < 0.7], idx[data["pos"] >= 0.7]
    splits = [("밝음→어두움", bright, dark), ("어두움→밝음", dark, bright), ("시간 분할 70/30", early, late)]
    results = []
    for name, tr, te in splits:
        for kind in ("svm", "rf"):
            t0 = time.time()
            r = evaluate(name, tr, te, data, kind, rng)
            results.append(r)
            print(f"[{name} / {kind}] 정확도 {r['accuracy']:.1%}  macroF1 {r['macro_f1']:.3f}  "
                  f"(확신≥0.6: {r['coverage@0.6']:.0%}, 정확도 {r['accuracy@0.6'] or 0:.1%})  "
                  f"{time.time() - t0:.0f}s")

    # 최종 모델: 전체 데이터
    import joblib
    xa, ya = augment(data["crops"], data["labels"], rng)
    final = make_model("svm").fit(featurize(xa), ya)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    report = fmt_report(results, counts)
    (MODEL_PATH.parent / "shadow_report.md").write_text(report, encoding="utf-8")
    (MODEL_PATH.parent / "shadow_report.json").write_text(
        json.dumps({"counts": counts, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n모델 저장: {MODEL_PATH}\n보고서: {MODEL_PATH.parent / 'shadow_report.md'}")


if __name__ == "__main__":
    main()
