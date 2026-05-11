# 06. Baseline Similarity Methods

## 1. 개요

본 프로젝트에서는 딥러닝 모델과 비교하기 위한 전통적 유사도 기반 baseline을 구현했습니다.

사용한 baseline은 다음 3가지입니다.

```text
1. Cosine Similarity
2. SSIM
3. pHash
```

## 2. 관련 코드

```text
src/baseline/similarity_metrics.py
src/baseline/evaluate_trad.py
src/baseline/save_threshold.py
src/baseline/generate_results.py
```

## 3. Cosine Similarity

Cosine Similarity는 두 이미지를 벡터로 펼친 뒤, 두 벡터의 방향 유사도를 계산합니다.

특징:

```text
- 색상과 밝기 변화에 민감
- 이미지 구조 변형에는 약함
- threshold가 낮으면 대부분을 표절로 판단할 수 있음
```

전처리:

```text
Resize(224x224)
RGB 유지
Tensor 변환
```

## 4. SSIM

SSIM은 이미지의 구조적 유사도를 비교하는 방식입니다.

특징:

```text
- 구조와 패턴 비교에 적합
- 회전, 반전, crop 등 물리적 변형에는 취약
- 보수적으로 작동하는 경향
```

전처리:

```text
Resize(224x224)
Grayscale 변환
```

## 5. pHash

pHash는 이미지를 해시값으로 변환한 뒤, 해시 간 거리를 비교합니다.

특징:

```text
- 계산이 빠름
- 포맷 변경, 약한 밝기 변화에는 비교적 안정적
- 세밀한 구조 변화에는 약함
```

## 6. threshold 문제

초기 실험에서는 모든 baseline에 동일한 threshold `0.75`를 적용했습니다.

하지만 방식별 score 분포가 달라 같은 threshold를 적용하면 성능이 왜곡되었습니다.

예시:

```text
Cosine: threshold가 낮으면 FP 증가
SSIM/pHash: threshold가 높으면 FN 증가
```

## 7. ROC Curve 기반 threshold 최적화

이를 해결하기 위해 validation set에서 ROC Curve를 만들고, Youden's J가 최대가 되는 threshold를 선택했습니다.

```text
Youden's J = TPR - FPR
```

관련 함수:

```python
find_best_threshold_by_roc(y_true, scores)
```

## 8. 최종 baseline 성능

| Method | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Cosine Similarity | 0.7573 | 0.9923 | 0.5187 | 0.6813 | 0.7689 |
| SSIM | 0.7760 | 0.9165 | 0.6073 | 0.7306 | 0.7903 |
| pHash | 0.7790 | 0.9729 | 0.5740 | 0.7220 | 0.8249 |

## 9. 해석

전통적 유사도 기반 방법은 Precision은 높지만 Recall이 낮은 경향을 보였습니다. 즉, 표절이라고 예측한 경우 실제 표절일 가능성은 높지만, 실제 표절 이미지를 많이 놓치는 문제가 있었습니다.

특히 flip, rotation, heavy crop과 같은 물리적 변형에서 미탐이 많이 발생했습니다.
