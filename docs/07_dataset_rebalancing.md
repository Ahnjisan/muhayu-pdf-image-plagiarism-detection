# 07. Dataset Rebalancing

## 1. 문제 발견

초기 데이터셋은 원본 이미지와 변형 이미지를 기반으로 표절쌍을 생성하고, 다른 이미지와의 조합을 비표절쌍으로 샘플링하는 방식이었습니다.

하지만 오류 분석 과정에서 다음 문제가 확인되었습니다.

```text
- 일부 transform_type에서 label 0 데이터가 거의 존재하지 않음
- 변형 종류별 positive/negative 분포가 불균형함
- FP/FN 분석이 transform_type별로 왜곡될 수 있음
```

즉, 전체 label 분포뿐 아니라 **변형 유형별 label 분포**도 중요했습니다.

## 2. 기존 방식의 한계

초기 방식:

```text
원본 이미지 기준 split
→ 원본과 변형 이미지를 label 1로 생성
→ 다른 이미지와 랜덤 매칭해 label 0 생성
```

문제:

```text
- transform_type별 negative pair가 균형 있게 생성되지 않음
- 일부 변형 유형은 positive만 존재
- 변형별 오탐/미탐 비율 계산이 어려움
```

## 3. 재구성 기준

최종 데이터셋은 다음 조건으로 재구성했습니다.

```text
총 이미지 쌍: 30,000 pairs
train : valid : test = 80 : 10 : 10
positive : negative = 1 : 1
변형 종류 분포 균형화
```

split별 구성:

| Split | Total | Positive | Negative |
|---|---:|---:|---:|
| Train | 24,000 | 12,000 | 12,000 |
| Valid | 3,000 | 1,500 | 1,500 |
| Test | 3,000 | 1,500 | 1,500 |

## 4. 관련 코드

```text
src/data/generate_dataset_csv.py
src/data/dataset_stats.py
scripts/04_generate_dataset_csv.py
scripts/05_check_dataset_stats.py
```

## 5. 개선된 pair 생성 흐름

```text
1. positive pair 후보 생성
   original_path ↔ transformed_path, label = 1

2. transform_type 추출
   crop, rotation, flip, score_based 등

3. negative pair 후보 생성
   original_path ↔ unrelated_transformed_path, label = 0

4. transform_type별 positive/negative 균형 샘플링

5. train/valid/test 분할

6. metadata 및 통계 저장
```

## 6. CSV 컬럼

```csv
original_path,transformed_path,label,field,paper,transform_type,transform_grade,total_score,transform_combo
```

이 컬럼 구조를 통해 단순 전체 성능뿐 아니라 다음 분석이 가능해졌습니다.

```text
- transform_type별 성능
- Light/Medium/Heavy별 성능
- 단일 변형 vs 복합 변형 성능
- FP/FN 변형 유형 분석
```

## 7. 재구성의 효과

데이터셋 재구성 후 baseline의 일부 성능 지표는 낮아졌지만, 평가 신뢰도는 높아졌습니다.

특히 Siamese 모델의 경우 label 균형 데이터셋에서 더 엄격한 조건으로 평가할 수 있었고, Contrastive Loss 기반 모델이 최종적으로 더 좋은 성능을 보였습니다.
