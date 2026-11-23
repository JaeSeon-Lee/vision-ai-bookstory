"""자막 읽어 주기(S 키)에 쓸 목소리를 미리 만든다. 이미 있는 것은 건너뛴다.

    .venv\\Scripts\\python.exe tools\\make_narration.py           # 없는 것만 만들기
    .venv\\Scripts\\python.exe tools\\make_narration.py --force   # 모두 다시 만들기

결과: assets/audio/narration/<해시>.wav (모델·목소리·문장이 같으면 같은 파일 이름)
자막 문장을 고치면 새 문장의 목소리만 새로 만들면 된다. 앱에서 S를 켜도 없는 것은 뒤에서 만든다.
"""
import argparse
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bookstory.cuts import CUTS  # noqa: E402
from bookstory.narrator import all_texts, cache_path, make_voice  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="있는 목소리도 다시 만든다")
    args = parser.parse_args()
    texts = all_texts(CUTS)
    for i, text in enumerate(texts, 1):
        path = cache_path(text)
        if args.force and path.exists():
            path.unlink()
        state = "있음" if path.exists() else "만듦"
        make_voice(text)
        with wave.open(str(path)) as w:
            seconds = w.getnframes() / w.getframerate()
        print(f"[{i:2}/{len(texts)}] {state} {seconds:4.1f}초  {text}")


if __name__ == "__main__":
    main()
