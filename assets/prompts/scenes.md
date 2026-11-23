# 배경·레이어 프롬프트

> **2026-11-16 변경**: 제미나이 화풍으로 통일. 화풍 기준 이미지는 `assets/images/raw/bg_06.png`이고,
> 실제로 쓰는 최신 프롬프트는 `tools/generate_images.py`의 `PROMPTS`에 있다. 장면 1 배경은
> `scene_01.png`(Designer) 대신 `bg_01.png`(제미나이)를 쓴다.

## 원칙
- 동물은 그리지 않는다. 동물이 설 자리는 **비워 두고**, PySide6 앱이 그 위치에 그림자 실루엣을 표시한다. 웹캠 손그림자가 인식되면 그 실루엣이 동물로 바뀐다.
- 그림자 위치는 이미지에 박지 않고 앱의 좌표 설정으로 관리한다. 그래야 위치를 자유롭게 고칠 수 있다.
- 달과 구름의 움직임은 **레이어(구름 PNG)**로 처리한다. 배경은 "달이 보이는 상태" 하나만 만들고, 앱에서 구름 레이어를 겹치고 어둡게 덮어서 "달이 가려진 상태"를 연출한다.
- 모든 프롬프트 끝에 `[STYLE]`(아래 공통 스타일)을 붙인다. 출력은 16:9로 생성한다.

### [STYLE]
```
Children's storybook illustration, Korean picture book style, cinematic composition,
magical moonlit night, soft blue color palette, warm moonlight, dreamy atmosphere,
watercolor painting, lush layered pine forest, mossy rocks, small white wildflowers,
sparkling starry sky with glowing cross-shaped stars, fluffy lavender-lit clouds,
high quality, consistent visual style, 16:9 wide landscape,
no animals, no birds, no people, no characters, no text, no captions, no numbers, no speech bubbles.
```

---

## 배경 이미지 (6장)

### bg_01 — 달이 비추는 숲의 밤하늘 ✅ 완료
`assets/images/scenes/scene_01.png`. 컷 1~3에 쓴다.

### bg_02 — 구름 위 높은 하늘 (새)
```
High above the clouds at night, a sea of soft moonlit clouds below, sharp blue mountain
peaks rising through the clouds, the full moon hidden behind a thick cloud bank in the
upper right with only a silver glow at its edges, a vast open starry sky filling the
center and left of the frame, feeling of great height and distance, [STYLE]
```
빈 자리: 화면 중앙~왼쪽 위의 하늘 (새가 날아가는 경로)

### bg_03 — 꽃 언덕 (토끼)
```
A gentle rolling hill covered with small white and pale blue wildflowers at night,
the hill rises from the lower left to a soft rounded crest at the center right,
open starry sky beyond the crest, distant forest and mountains far below the hill,
moon hidden behind clouds with a soft glow, the hilltop is empty and open, [STYLE]
```
빈 자리: 언덕 비탈(왼쪽 아래) → 언덕 꼭대기(가운데 오른쪽)

### bg_04 — 달빛 숲속 빈터 (여우 등장)
```
A quiet clearing in a deep pine and oak forest at night, soft blue moonlight filtering
through the branches in gentle rays, mossy ground with ferns and white wildflowers at
the edges, the center of the clearing is open and empty, tall trees framing both sides,
[STYLE]
```
빈 자리: 빈터 가운데 바닥

### bg_05 — 깊은 숲길 (여우 퇴장)
```
A narrow winding forest path leading deeper into a dense blue moonlit forest, tall
trees on both sides forming a tunnel, the path curves away into misty darkness at the
center of the frame, faint moonbeams through the canopy, mossy roots and ferns,
the path is empty, [STYLE]
```
빈 자리: 앞쪽 길 → 화면 중앙 소실점

### bg_06 — 달이 뜬 연못 (사슴, 결말)
```
A calm round forest pond at night, a grassy bank with mossy rocks and white wildflowers
in the lower left foreground, a brilliant full moon in the upper right sky with a
perfect shimmering reflection on the still water, pine forest and soft mountains
around the pond, the grassy bank in the foreground is open and empty, [STYLE]
```
빈 자리: 왼쪽 아래 풀밭 (사슴이 앉는 자리)
연출: 앱에서 구름 레이어와 어두운 오버레이를 덮어 "달이 가려진 연못"으로 시작하고, 구름이 걷히면서 오버레이도 사라지게 한다.

---

## 구름 레이어 (2장)
생성기는 투명 배경을 만들지 못하므로 **단색 초록 배경**에서 만든 뒤, 초록색을 빼서 투명 PNG로 만든다.

### cloud_01 — 달을 가리는 큰 구름
```
A single large thick fluffy cumulus cloud, watercolor children's book style, soft
lavender and blue-gray shading with a silver moonlit rim on its top edge, isolated
in the center, on a plain flat solid pure green (#00FF00) background, no other
objects, no text
```

### cloud_02 — 옅은 구름 조각
```
Three small soft wispy watercolor clouds, pale blue-gray with faint lavender tint,
isolated and spaced apart, on a plain flat solid pure green (#00FF00) background,
no other objects, no text
```

---

## 컷 구성 (나레이션 16컷 → 배경 6장)

| 컷 | 내용 | 배경 | 레이어·연출 | 그림자 |
|---|---|---|---|---|
| 1 | 달이 비추는 밤 | bg_01 | 구름이 오른쪽에서 천천히 다가옴 | — |
| 2 | 달이 구름에 가려짐 | bg_01 | cloud_01이 달을 덮고 화면이 어두워짐 | — |
| 3 | 새가 달을 찾음 | bg_01 (어두운 상태) | — | 새: 아래 → 위 |
| 4 | 더 높이 날아가는 새 | bg_02 | cloud_02 아래로 흐름 | 새: 위쪽으로 이동 |
| 5 | 달을 찾지 못한 새 | bg_02 | — | 새: 화면 밖으로 |
| 6 | 토끼 등장 | bg_03 | — | 토끼: 비탈 |
| 7 | 언덕 위 토끼 | bg_03 | — | 토끼: 꼭대기 |
| 8 | 사라지는 토끼 | bg_03 | — | 토끼: 능선 너머로 |
| 9 | 여우 등장 | bg_04 | — | 여우: 빈터 |
| 10 | 숲을 찾는 여우 | bg_04 | 달빛이 흔들림 | 여우: 좌우로 이동 |
| 11 | 깊은 숲속 여우 | bg_05 | — | 여우: 소실점으로 작아짐 |
| 12 | 연못가 사슴 | bg_06 (어두운 상태) | cloud_01이 달을 덮음 | 사슴: 풀밭 |
| 13 | 기다리는 사슴 | bg_06 (어두운 상태) | 수면이 희미하게 일렁임 | 사슴 |
| 14 | 구름이 걷히기 시작 | bg_06 | cloud_01이 밀려나고 수면에 물결 | 사슴 |
| 15 | 달의 등장 | bg_06 | 오버레이가 사라지며 달과 반사가 드러남 | 사슴 |
| 16 | 엔딩 | bg_06 | 따뜻한 색 오버레이, 반짝이는 별 | 사슴 |
