"""기다림의 달 — 그림자 구연 동화 앱.

실행 (Windows, 프로젝트 루트에서):
    .venv\\Scripts\\python.exe main.py              # 메인 화면에서 시작 (Space로 이야기 시작)
    .venv\\Scripts\\python.exe main.py --cut 2      # 컷 2부터 바로 재생

스냅샷 (창 없이 특정 시점을 PNG로 저장):
    .venv\\Scripts\\python.exe main.py --snapshot 2 3.5 out.png   # 컷 2의 3.5초 시점
    (그림자 연출이 있는 컷은 대기 시점보다 뒤의 시간을 주면 그림자가 인식된 것으로 그린다)
"""
import argparse
import os
import sys
from pathlib import Path

from bookstory.cuts import CUTS, SHADOW_GUIDES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", type=int, default=0, help="시작할 컷 번호 (0 = 메인 화면)")
    parser.add_argument("--snapshot", nargs=3, metavar=("CUT", "SECONDS", "OUT"),
                        help="창을 띄우지 않고 해당 시점을 PNG로 저장")
    args = parser.parse_args()

    by_number = {c.number: i for i, c in enumerate(CUTS)}

    if args.snapshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QGuiApplication
        from bookstory.stage import Renderer
        app = QGuiApplication(sys.argv)  # noqa: F841 — 글꼴·이미지 처리에 필요
        cut_no, seconds, out = int(args.snapshot[0]), float(args.snapshot[1]), Path(args.snapshot[2])
        renderer = Renderer(SHADOW_GUIDES)
        renderer.preload(CUTS)
        renderer.snapshot(CUTS[by_number[cut_no]], seconds, out)
        print(f"저장: {out}")
        return

    from PySide6.QtWidgets import QApplication
    from bookstory.stage import Stage
    app = QApplication(sys.argv)
    stage = Stage(CUTS, start=by_number.get(args.cut, 0), guides=SHADOW_GUIDES)
    stage.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
