# 05. Score-based Transformation

## 1. 개요

`Transformed_score`는 실제 이미지 표절 상황을 더 현실적으로 모사하기 위해 만든 점수제 복합 변형 데이터셋입니다.

단일 변형이 `crop`, `rotation`, `brightness` 등 하나의 변형만 적용하는 방식이라면, 점수제 변형은 여러 변형을 동시에 적용합니다.

## 2. 관련 코드

```text
src/transform/image_transform_score.py
```

## 3. 처리 흐름

```text
Extracted/{field}/{paper}/Figure/*.png
    ↓
2~4개 변형 랜덤 선택
    ↓
각 변형 강도에 score 부여
    ↓
총점 계산
    ↓
Light / Medium / Heavy 등급 결정
    ↓
Transformed_score에 저장
```

## 4. 점수 체계

```python
score_map = {
    "light": 1,
    "medium": 2,
    "heavy": 3
}
```

등급 기준:

```python
if 1 <= total_score <= 3:
    grade = "Light"
elif 4 <= total_score <= 6:
    grade = "Medium"
else:
    grade = "Heavy"
```

## 5. 저장 폴더명 규칙

점수제 변형 결과는 적용 변형이 폴더명에 기록되도록 저장합니다.

```text
run_{timestamp}_{grade}_{total_score}_{transform_combo}
```

예시:

```text
run_20250520T063444_Medium_5_format_medium__brightness_heavy
```

이 폴더명만 봐도 다음 정보를 알 수 있습니다.

| 항목 | 값 |
|---|---|
| 생성 시각 | 20250520T063444 |
| 표절 강도 등급 | Medium |
| 총점 | 5 |
| 변형 조합 | format_medium + brightness_heavy |

## 6. metadata.txt

각 run 폴더에는 `metadata.txt`가 생성됩니다.

예시:

```text
transforms: [('format', 'medium'), ('brightness', 'heavy')]
total_score: 5
grade: Medium
timestamp: 20250520T063444
```

이 정보는 이후 CSV 생성 시 다음 컬럼으로 활용됩니다.

```text
transform_grade
total_score
transform_combo
```

## 7. 점수제 변형의 역할

점수제 변형은 다음 목적을 가집니다.

1. 실제 표절 상황처럼 여러 변형이 동시에 적용된 이미지 생성
2. Light/Medium/Heavy 표절 강도 등급 부여
3. 표절 여부 탐지 이후 강도 분류 분석 가능
4. 단순 단일 변형보다 현실적인 학습 데이터 확보

## 8. 단일 변형과의 차이

| 구분 | Transformed_single | Transformed_score |
|---|---|---|
| 변형 방식 | 하나의 변형만 적용 | 2~4개 변형 랜덤 조합 |
| 목적 | 오류 분석, 변형별 성능 확인 | 실제 표절 상황 모사 |
| 등급 | light/medium/heavy 직접 구분 | 총점 기반 Light/Medium/Heavy 자동 분류 |
| 저장 구조 | level/transform_type | run_timestamp_grade_score_combo |
