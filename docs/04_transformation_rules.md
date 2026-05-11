# 04. Transformation Rules

## 1. 목적

본 프로젝트에서는 회사 측에서 제시한 이미지 표절 정의를 바탕으로 원본 이미지에서 다양한 표절 변형 이미지를 생성했습니다.

변형은 단일 변형과 복합 변형으로 나뉘며, 변형 강도는 `light`, `medium`, `heavy`로 구분했습니다.

## 2. 표절 정의 기준

| 변형 종류 | light | medium | heavy |
|---|---|---|---|
| 자르기 | 10% crop | 20% crop | 30% crop |
| 해상도 | 해상도 10% 줄이기 | 해상도 20% 줄이기 | 해상도 30% 줄이기 |
| 회전 | 90도 회전 | 270도 회전 | - |
| 반전 | 좌우 반전 | 상하 반전 | 좌우+상하 반전 |
| 캡처 | black border 10% | black border 20% | black border 30% |
| 흑백 | 흑백 변환 | - | - |
| 밝기 | 밝기 10% 높이기 | 밝기 20% 높이기 | 밝기 30% 높이기 |
| 대조 | 대조 10% 높이기 | 대조 20% 높이기 | 대조 30% 높이기 |
| 포맷 | jpg → jpeg | jpg → png | jpg → webp |
| 삽입 | 다른 이미지 위에 원본 이미지 삽입 | - | - |
| 붙이기 | 여러 이미지를 가로/세로로 이어 붙이기 | - | - |

## 3. 관련 코드

```text
src/transform/image_transform_single.py
src/transform/image_transform_score.py
configs/transform_config.yaml
```

## 4. 단일 변형

단일 변형은 하나의 변형만 적용한 이미지입니다.

예시:

```text
original.png
→ crop_light.png
→ rotation_medium.png
→ flip_heavy.png
```

단일 변형의 목적은 다음과 같습니다.

1. 변형 유형별 탐지 성능 분석
2. 각 알고리즘의 취약 변형 파악
3. FP/FN 오류 분석

## 5. 점수제 복합 변형

점수제 복합 변형은 2~4개의 변형을 랜덤으로 선택하여 조합하고, 각 변형 강도에 점수를 부여해 최종 등급을 결정하는 방식입니다.

```text
light = 1점
medium = 2점
heavy = 3점
```

등급 기준:

```text
1~3점: Light
4~6점: Medium
7점 이상: Heavy
```

예시:

```text
crop_medium = 2점
resolution_light = 1점
format_light = 1점
총합 = 4점 → Medium
```

## 6. 변형 조합 필터링

랜덤 조합은 완전히 무작위로만 만들지 않고, 학습에 방해될 수 있는 조합은 일부 배제했습니다.

예시:

```text
- 모든 변형이 light인 조합 제외
- format heavy만 있고 crop 등 주요 시각 변형이 없는 조합 제외
```

이 방식은 실제 표절 상황을 모사하되, 지나치게 의미 없는 변형 조합이 데이터셋에 포함되는 것을 줄이기 위한 목적입니다.

## 7. 산출물

단일 변형:

```text
Transformed_single/{field}/{paper}/{level}/{transform_type}/Figure/*.png
```

점수제 복합 변형:

```text
Transformed_score/{field}/{paper}/run_{timestamp}_{grade}_{score}_{combo}/Figure/*.png
```

metadata:

```text
metadata.txt
├── transforms: [('format', 'medium'), ('brightness', 'heavy')]
├── total_score: 5
├── grade: Medium
└── timestamp: 20250520T063444
```
