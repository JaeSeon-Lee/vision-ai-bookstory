"""Gemini API로 배경·구름 이미지를 생성한다.

사용법 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe tools\\generate_images.py bg_06 cloud_01
    .venv\\Scripts\\python.exe tools\\generate_images.py all

- 기준 이미지(Designer_1.png)를 함께 보내 화풍을 맞춘다.
- 결과는 assets/images/raw/<이름>.png 에 저장한다. 기존 파일은 <이름>_v2.png 처럼 번호를 붙여 보존한다.
- 저장 뒤 tools/prepare_images.py 로 1920x1080 맞추기와 구름 배경 제거를 한다.
"""
import os
import sys
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "assets" / "images" / "raw"
# 화풍 기준: 제미나이로 만든 연못 배경 (2026-11-16 제미나이 화풍으로 통일하기로 결정)
REFERENCE = ROOT / "assets" / "images" / "raw" / "bg_06.png"
# 이름별로 다른 기준 이미지 (구름은 구름 그림을 기준으로 해야 배경을 베껴 오지 않는다)
REFERENCE_FOR = {
    "cloud_01": ROOT / "assets" / "images" / "raw" / "bg_02.png",
    "cloud_01_clean": ROOT / "assets" / "images" / "raw" / "cloud_01.png",
    "animal_bird": ROOT / "assets" / "images" / "raw" / "bg_02.png",
    "animal_rabbit": ROOT / "assets" / "images" / "raw" / "animal_bird.png",
    "animal_rabbit_land": ROOT / "assets" / "images" / "raw" / "animal_rabbit.png",
    "animal_fox_front": ROOT / "assets" / "images" / "raw" / "animal_fox.png",
    "animal_deer_smile": ROOT / "assets" / "images" / "raw" / "animal_deer.png",
    "animal_deer_laugh": ROOT / "assets" / "images" / "raw" / "animal_deer.png",
    **{n: ROOT / "assets" / "images" / "raw" / "animal_fox.png"
       for n in ("animal_fox_right", "animal_fox_mid", "animal_fox_back")},
    **{f"animal_fox_turn{i}": [ROOT / "assets" / "images" / "raw" / "animal_fox.png",
                               ROOT / "assets" / "images" / "raw" / "animal_fox_front.png"]
       for i in (1,)},
    "animal_fox": ROOT / "assets" / "images" / "raw" / "animal_bird.png",
    "animal_deer": ROOT / "assets" / "images" / "raw" / "animal_bird.png",
    "bg_03_hidden": ROOT / "assets" / "images" / "raw" / "bg_03.png",
    "bg_title": [ROOT / "assets" / "images" / "raw" / f"{n}.png"
                 for n in ("bg_06", "animal_bird", "animal_rabbit", "animal_fox", "animal_deer")],
    "bg_06_cloudy": ROOT / "assets" / "images" / "raw" / "bg_06.png",
    "cloud_02": ROOT / "assets" / "images" / "raw" / "cloud_01.png",
    "bg_02": ROOT / "assets" / "images" / "raw" / "bg_01.png",
}
MODEL = "gemini-3-pro-image"

STYLE = (
    "Children's storybook illustration, Korean picture book style, cinematic composition, "
    "magical moonlit night, soft blue color palette, warm moonlight, dreamy atmosphere, "
    "watercolor painting, lush layered pine forest, mossy rocks, small white wildflowers, "
    "sparkling starry sky with glowing cross-shaped stars, fluffy lavender-lit clouds, "
    "high quality, consistent visual style, 16:9 wide landscape, "
    "no animals, no birds, no people, no characters, no text, no captions, no numbers, no speech bubbles."
)

STYLE_REF = (
    "Use the attached reference image ONLY as a guide for painting style, brushwork, color palette "
    "and lighting. Do not copy its composition. Create a new scene: "
)

PROMPTS = {
    "bg_01": STYLE_REF + (
        "A peaceful forest viewed from a grassy hillside at night, an old oak tree framing the "
        "left edge, mossy rocks and white wildflowers in the lower left foreground, rows of pine "
        "trees descending toward a distant calm lake and soft blue mountains, a bright full moon "
        "in the upper right of a wide open starry sky, a few soft clouds drifting in from the "
        "right edge, plenty of open sky in the upper center for something to fly through. "
    ) + STYLE,
    # bg_01과 같은 장소를 이어서 보여 주는 장면 (Designer_3, Designer_4 구성 참고)
    "bg_02": (
        "The attached image is the previous scene of the same picture book. Paint the NEXT "
        "scene at the SAME place, in exactly the same watercolor style, colors and level of "
        "detail: the same old oak tree on the left edge with its leafy branch reaching into "
        "the sky, the same pine forest, the same calm lake and the same soft low mountains far "
        "away. The camera is tilted slightly upward, so the sky takes about two thirds of the "
        "frame and the lake and forest sit lower in the frame. The full moon is now completely "
        "hidden behind a large mass of thick, billowing lavender and blue-gray clouds in the "
        "upper right; the moon itself is not visible, only a bright silver-gold glow leaking "
        "around the cloud edges. The sky is a slightly deeper night blue than before, with "
        "many sparkling stars in the open sky on the left and center. A wide open area of sky "
        "in the upper left and center, free of branches, for something to fly through. "
        "No sharp or snowy mountain peaks. "
    ) + STYLE,
    "bg_03": STYLE_REF + (
        "A gentle rolling hill covered with small white and pale blue wildflowers at night, "
        "the hill rises from the lower left to a soft rounded crest at the center right, "
        "open starry sky beyond the crest, distant forest and mountains far below the hill, "
        "moon hidden behind clouds with a soft glow, the hilltop is empty and open. "
    ) + STYLE,
    "bg_04": STYLE_REF + (
        "A quiet clearing in a deep pine and oak forest at night, soft blue moonlight filtering "
        "through the branches in gentle rays, mossy ground with ferns and white wildflowers at "
        "the edges, the center of the clearing is open and empty, tall trees framing both sides. "
    ) + STYLE,
    "bg_05": STYLE_REF + (
        "A narrow winding forest path leading deeper into a dense blue moonlit forest, tall "
        "trees on both sides forming a tunnel, the path curves away into misty darkness at the "
        "center of the frame, faint moonbeams through the canopy, mossy roots and ferns, "
        "the path is empty. The moon itself is NOT visible anywhere in the image: the sky "
        "between the branches is covered by soft clouds, with only a faint diffuse glow. "
    ) + STYLE,
    "bg_06": STYLE_REF + (
        "A calm round forest pond at night, a grassy bank with mossy rocks and white wildflowers "
        "in the lower left foreground, a brilliant full moon in the upper right sky with a "
        "perfect shimmering reflection on the still water, pine forest and soft mountains "
        "around the pond, the grassy bank in the foreground is open and empty. "
    ) + STYLE,
    # bg_02의 큰 구름 덩어리와 같은 화풍이어야 bg_02로 전환할 때 어색하지 않다
    "cloud_01": (
        "Look closely at the large billowing cloud mass in the upper right of the attached "
        "picture. Paint ONE cloud bank that looks exactly like it, as if cut out of the same "
        "painting: the same soft pastel lavender, periwinkle and white colors, the same light "
        "loose watercolor washes with soft edges, the same warm golden-cream moonlit glow on "
        "the upper rims. It is a wide, thick, fully opaque cumulus cloud bank, dense enough to "
        "completely hide a full moon behind it. Do not make it darker, grayer or more detailed "
        "than the clouds in the picture. The cloud is isolated in the center with generous "
        "empty margin around it, on a plain flat solid pure green (#00FF00) chroma key "
        "background. No sky, no stars, no moon, no trees, no other objects, no text."
    ),
    # cloud_01 위쪽에 기준 이미지(나무·하늘)의 잔상이 섞여 나와서, 구름은 그대로 두고 잔상만 지운다
    "cloud_01_clean": (
        "Edit the attached image. Keep the big lavender watercolor cloud exactly as it is: same "
        "shape, size, position, colors and brushwork. Remove everything else: the faded tree, "
        "leaves, glowing dots and hazy sky along the top of the image must disappear completely. "
        "The entire background, from edge to edge including the whole top area, must be one "
        "plain flat solid pure green (#00FF00) with no gradients, haze or shadows. "
        "No other changes, no text."
    ),
    # 동물 그림: 손그림자가 이 모양으로 바뀌며 완성된다. 초록 배경에서 오려 투명 PNG로 쓴다
    "animal_bird": (
        "Use the attached picture ONLY as a guide for painting style, colors and moonlit "
        "lighting. Paint ONE cute small bird, a Korean magpie-like songbird, flying upward "
        "with both wings spread wide and strong, seen from the side, body angled up toward the "
        "upper right, head lifted to look up at the sky, bright round eye, beak slightly open. "
        "Deep navy-blue and white feathers with soft lavender tints, clearly visible detailed "
        "feathers, warm golden moonlit rim light along the top edges of the wings. Same light "
        "watercolor children's picture book style as the attached picture, bright and clearly "
        "visible. The bird is isolated in the center with generous empty margin, on a plain "
        "flat solid pure green (#00FF00) chroma key background filling the whole frame. "
        "No sky, no trees, no branches, no moon, no stars, no other objects, no text."
    ),
    "animal_rabbit": (
        "The attached image is a watercolor bird illustration from the same picture book, on a "
        "green chroma key background. In exactly the same painting style, colors, moonlit rim "
        "light and level of detail, paint ONE cute fluffy rabbit (no bird) mid-hop, facing right, "
        "body stretched forward in a joyful leap, long ears up, bright round eye, soft white and "
        "pale gray fur with lavender shading, warm golden moonlit rim light along its back and "
        "ears. Bright and clearly visible. Isolated in the center with generous empty margin, on "
        "a plain flat solid pure green (#00FF00) background filling the whole frame. No ground, "
        "no grass, no shadow, no other objects, no text."
    ),
    # 토끼 점프 애니메이션의 착지 자세 (뛰는 자세 = animal_rabbit)
    "animal_rabbit_land": (
        "The attached image is a watercolor rabbit from a picture book, mid-leap, on a green "
        "chroma key background. Paint THE SAME rabbit again - identical fur colors, lavender "
        "shading, golden moonlit rim light, face, eye, ear shape, painting style and level of "
        "detail, same size and same side view facing right - but in a LANDING pose: legs pulled "
        "in under the body, front paws tucked together just under the chest touching down, big "
        "hind legs folded and gathered beneath the haunches, body compact and rounded like a "
        "crouch, back gently arched, ears up and slightly swept back, fluffy tail. Only the pose "
        "changes. Isolated in the center with generous empty margin, on a plain flat solid pure "
        "green (#00FF00) background filling the whole frame. No ground, no grass, no shadow, no "
        "other objects, no text."
    ),
    # 여우가 돌아설 때 중간에 끼우는 정면 모습 (좌우 대칭이라 뒤집어도 티가 나지 않는다)
    "animal_fox_front": (
        "The attached image is a watercolor fox from a picture book, on a green chroma key "
        "background. Paint THE SAME fox again - identical orange and cream fur colors, dark legs, "
        "face, eyes, ears, painting style, golden moonlit rim light and level of detail, same "
        "size - but standing on all four legs facing the viewer straight on (front view), body "
        "pointing toward the camera, head looking straight forward, perfectly left-right "
        "symmetrical pose, front legs side by side, the fluffy tail hanging straight down behind "
        "the legs so only its white tip shows between the hind legs. Isolated in the center with "
        "generous empty margin, on a plain flat solid pure green (#00FF00) background filling the "
        "whole frame. No ground, no grass, no shadow, no other objects, no text."
    ),
    # 여우가 옆모습(왼쪽) -> 정면으로 돌아서는 중간 모습. 생성은 turn1만 (turn2·3은 잘 안 나와서
    # tools/make_variants.py가 turn1의 몸 뒤쪽을 줄여 만든다). 반대쪽은 좌우를 뒤집어 쓴다
    **{f"animal_fox_turn{i}": (
        "The two attached images show the SAME watercolor fox from a picture book on a green "
        "chroma key background: first in side view with its body pointing to the LEFT (head "
        "turned toward the viewer), second facing the viewer straight on (front view). Paint "
        "THE SAME fox again - identical orange and cream fur, dark legs, face, eyes, ears, "
        "painting style, golden moonlit rim light and level of detail, same size - in an "
        f"in-between pose while it turns around: {desc} All four legs standing on the ground, "
        "head looking toward the viewer, tail hanging down behind. Isolated in the center with "
        "generous empty margin, on a plain flat solid pure green (#00FF00) background filling the "
        "whole frame. No ground, no grass, no shadow, no other objects, no text."
    ) for i, desc in (
        (1, "three-quarter view, its body pointing to the left and turned about 65 degrees away "
            "from the camera, so the left side of the body and part of the chest are visible, "
            "the tail still visible behind the body on the right."),
    )},
    # 여우 걷기·돌아서기 (tools/make_variants.py로 다듬고 tools/align_pose.py로 자리를 맞춘다)
    # 오른쪽으로 걷는 옆모습: 몸과 고개가 모두 오른쪽. 좌우를 뒤집으면 왼쪽으로 걷는 모습
    "animal_fox_right": (
        "The attached image is a watercolor fox from a picture book, on a green chroma key "
        "background. Paint THE SAME fox again - identical orange and cream fur, dark legs, face, "
        "eyes, ears, fluffy white-tipped tail, painting style, golden moonlit rim light and level "
        "of detail, same size - but in a clean SIDE PROFILE walking to the RIGHT: body pointing "
        "right, head facing forward to the right in profile (nose pointing right, looking where "
        "it is going, not at the viewer), mid-stride with one front leg lifted, tail flowing "
        "behind on the left. Isolated in the center with generous empty margin, on a plain flat "
        "solid pure green (#00FF00) background filling the whole frame. No ground, no grass, no "
        "shadow, no other objects, no text."
    ),
    # 돌아서는 중간: 몸은 관객 쪽으로 비스듬히, 고개는 가려는 오른쪽을 본다
    "animal_fox_mid": (
        "The attached image is a watercolor fox from a picture book, on a green chroma key "
        "background. Paint THE SAME fox again - identical orange and cream fur, dark legs, face, "
        "eyes, ears, painting style, golden moonlit rim light and level of detail, same size - "
        "caught in the middle of turning around: its body is angled three-quarters toward the "
        "viewer (chest and front legs toward the viewer, hindquarters behind on the left), while "
        "its HEAD is turned to the RIGHT in near profile, looking to the right where it is about "
        "to go (NOT looking at the viewer). Front legs stepping to the right, tail swinging "
        "behind. Isolated in the center with generous empty margin, on a plain flat solid pure "
        "green (#00FF00) background filling the whole frame. No ground, no grass, no shadow, no "
        "other objects, no text."
    ),
    # 숲 깊은 곳으로 걸어 들어가는 뒷모습 (컷 11)
    "animal_fox_back": (
        "The attached image is a watercolor fox from a picture book, on a green chroma key "
        "background. Paint THE SAME fox again - identical orange and cream fur, dark legs, ears, "
        "fluffy white-tipped tail, painting style, golden moonlit rim light and level of detail, "
        "same size - but seen FROM BEHIND, walking away from the viewer into the distance, "
        "heading slightly to the upper right: we see its back, hindquarters and the back of its "
        "head and ears, the big fluffy tail hanging down and swaying, mid-stride with one hind "
        "leg lifted. Face not visible. Isolated in the center with generous empty margin, on a "
        "plain flat solid pure green (#00FF00) background filling the whole frame. No ground, no "
        "grass, no shadow, no other objects, no text."
    ),
    # 사슴의 입을 벌리고 웃는 얼굴 (입 부분만 원본에 옮겨 붙인다: tools/make_variants.py)
    "animal_deer_laugh": (
        "The attached image is a watercolor deer from a picture book, on a green chroma key "
        "background. Paint EXACTLY the same deer again - identical pose, size, framing, fur "
        "colors, white spots, antlers, eye, painting style and lighting - with only ONE change: "
        "it is laughing happily with its MOUTH OPEN in a big joyful smile, the lower jaw dropped "
        "open, a little pink tongue visible inside, corners of the mouth curving upward. "
        "Everything else stays exactly the same. On a plain flat solid pure green (#00FF00) "
        "background filling the whole frame. No text."
    ),
    # 사슴의 웃는 얼굴 (입 부분만 원본에 옮겨 붙인다: tools/make_variants.py)
    "animal_deer_smile": (
        "The attached image is a watercolor deer from a picture book, on a green chroma key "
        "background. Paint EXACTLY the same deer again - identical pose, size, framing, fur "
        "colors, white spots, antlers, eye, painting style and lighting - with only ONE change: "
        "its mouth is a gentle happy smile, the corner of the mouth curving upward, lips closed, "
        "a warm content expression. Everything else stays exactly the same. On a plain flat solid "
        "pure green (#00FF00) background filling the whole frame. No text."
    ),
    "animal_fox": (
        "The attached image is a watercolor bird illustration from the same picture book, on a "
        "green chroma key background. In exactly the same painting style, colors, moonlit rim "
        "light and level of detail, paint ONE friendly clever fox (no bird) standing on all four "
        "legs, body facing left in three-quarter view, head turned slightly to look around "
        "curiously, big ears up, bushy tail, bright orange fur with cream chest and dark legs, "
        "warm golden moonlit rim light along its back. Bright and clearly visible. Isolated in "
        "the center with generous empty margin, on a plain flat solid pure green (#00FF00) "
        "background filling the whole frame. No ground, no grass, no shadow, no other objects, "
        "no text."
    ),
    "animal_deer": (
        "The attached image is a watercolor bird illustration from the same picture book, on a "
        "green chroma key background. In exactly the same painting style, colors, moonlit rim "
        "light and level of detail, paint ONE gentle young deer (no bird) lying down calmly with "
        "legs folded, body facing right, neck raised and head lifted to gaze up toward the upper "
        "right sky, small antlers, big gentle eyes, warm light brown fur with soft white spots "
        "and cream belly, warm golden moonlit rim light along its back and head. Bright and "
        "clearly visible. Isolated in the center with generous empty margin, on a plain flat "
        "solid pure green (#00FF00) background filling the whole frame. No ground, no grass, "
        "no shadow, no other objects, no text."
    ),
    # 메인 화면 배경. 제목 글자는 앱에서 얹으므로 위쪽 하늘을 비워 둔다
    "bg_title": (
        "Title screen illustration for a Korean children's picture book about four animals "
        "waiting for the moon. The first attached image shows the painting style, colors and "
        "the pond location to use. The other attached images are the four characters (a magpie, "
        "a rabbit, a fox and a deer); paint them with exactly these designs and colors. "
        "Scene: a calm moonlit night at the forest pond. A big glowing full moon sits low in the "
        "right part of the sky, softly veiled by thin lavender clouds, with its shimmering "
        "reflection on the water. On the grassy bank across the bottom of the picture, the four "
        "friends are small and together, all looking up at the moon: the deer lying calmly, the "
        "rabbit sitting beside it, the fox sitting, and the magpie perched on a mossy rock. "
        "The upper center of the sky is open, calm and uncluttered with only a few soft stars, "
        "leaving empty space for a title. Warm, hopeful, dreamy atmosphere, 16:9 wide. "
        "Children's storybook illustration, Korean picture book style, watercolor painting, "
        "soft blue color palette, warm moonlight. No text, no letters, no captions, no numbers."
    ),
    # 꽃 언덕: 처음 생성본은 달이 보여서, 같은 그림에서 달만 구름 뒤로 숨긴다
    "bg_03_hidden": (
        "Edit the attached picture. Keep everything exactly the same (the flowery hill, the "
        "forest, the mountains, the stars, the painting style and colors), but the moon must be "
        "completely hidden behind the soft lavender clouds: the moon disc is not visible at all, "
        "only a gentle silver-gold glow around the cloud edges where it hides. No other changes, "
        "no animals, no text."
    ),
    # 연못: 컷 12~14는 달이 가려진 상태. 같은 그림에서 달과 물에 비친 달을 모두 없앤다
    "bg_06_cloudy": (
        "Edit the attached picture. Keep everything exactly the same (the pond, the grassy bank, "
        "the rocks and flowers, the trees, the mountains, the stars, the painting style), but the "
        "full moon in the sky must be completely hidden behind a mass of soft lavender clouds, "
        "with only a faint silver glow at the cloud edges, and the moon's reflection on the pond "
        "must disappear: the water now reflects only the dark trees and a few faint stars. The "
        "night feels a little darker and quieter. No other changes, no animals, no text."
    ),
    "cloud_02": (
        "The attached image is a watercolor cloud on a green chroma key background. "
        "In exactly the same painting style and colors, paint three SMALLER, thinner, wispy, "
        "elongated clouds, isolated and spaced apart in a horizontal row with empty margin "
        "around each, on the same plain flat solid pure green (#00FF00) background. "
        "No sky, no stars, no moon, no landscape, no other objects, no text."
    ),
}


def next_path(name: str) -> Path:
    path = RAW / f"{name}.png"
    n = 2
    while path.exists():
        path = RAW / f"{name}_v{n}.png"
        n += 1
    return path


def generate(client: genai.Client, name: str, references: list[Image.Image]) -> Path | None:
    response = client.models.generate_content(
        model=MODEL,
        contents=[PROMPTS[name], *references],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9", image_size="2K"),
        ),
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.data:
            out = next_path(name)
            Image.open(BytesIO(part.inline_data.data)).save(out)
            return out
    print(f"[{name}] 이미지가 반환되지 않았습니다:", response.text)
    return None


def main() -> None:
    load_dotenv(ROOT / ".env")
    names = sys.argv[1:] or ["all"]
    if names == ["all"]:
        names = list(PROMPTS)
    unknown = [n for n in names if n not in PROMPTS]
    if unknown:
        sys.exit(f"알 수 없는 이름: {unknown}. 가능한 이름: {list(PROMPTS)}")

    RAW.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for name in names:
        print(f"[{name}] 생성 중...", flush=True)
        refs = REFERENCE_FOR.get(name, REFERENCE)
        refs = refs if isinstance(refs, list) else [refs]
        out = generate(client, name, [Image.open(r) for r in refs])
        if out:
            with Image.open(out) as img:
                print(f"[{name}] 저장: {out.relative_to(ROOT)} {img.size}")


if __name__ == "__main__":
    main()
