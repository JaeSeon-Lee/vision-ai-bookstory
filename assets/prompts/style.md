# 공통 화풍 (모든 씬에 동일 적용)

- 기준 이미지: `assets/images/sample/Designer_1.png`
- 출력 규격: 16:9, 1920x1080 PNG → `assets/images/scenes/scene_XX.png`
- 동물은 그리지 않는다. 동물 자리는 비워 두고, 웹캠으로 인식한 그림자를 앱(PySide6)에서 채운다.

## 공통 스타일 프롬프트 (모든 씬 끝에 붙임)

```
Children's storybook illustration, Korean picture book style, cinematic composition,
magical moonlit night, soft blue color palette, warm moonlight, dreamy atmosphere,
watercolor painting, lush layered pine forest, mossy rocks, small white wildflowers
in the foreground, sparkling starry sky with glowing cross-shaped stars,
fluffy lavender-lit clouds, high quality, consistent visual style,
16:9 wide landscape, no animals, no people, no text, no captions, no numbers, no speech bubbles.
```

> 원래 스타일 문구의 "bright and clearly visible animal character, detailed feathers and eyes"는
> 배경에 동물을 그리지 않기 위해 뺐다. 이 문구가 있으면 생성기가 동물을 그려 넣는다.

## scene_01 — 달이 사라진 밤

- `scene_01.png` (완료): 기준 이미지를 16:9로 잘라 1920x1080으로 키운 것. 달이 보이는 시작 화면.
- ~~`scene_01_b.png`~~: 필요 없음. 구름 레이어를 PySide6에서 움직여 처리한다. 전체 구성은 `scenes.md` 참고.

```
Same composition as the reference: an old oak tree framing the left side, mossy rocks
and white wildflowers in the foreground, a calm lake between pine forests, layered
blue mountains in the distance. A large fluffy cloud has drifted across the sky and
completely covers the full moon; only a soft silver glow leaks from the cloud's edges.
The lake no longer shows a moon reflection, just faint starlight. Quieter, slightly
darker mood. + [공통 스타일 프롬프트]
```
