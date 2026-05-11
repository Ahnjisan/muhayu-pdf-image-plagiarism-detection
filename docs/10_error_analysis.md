# 10. Error Analysis

## 1. 목적

모델의 전체 Accuracy나 F1-score만으로는 어떤 변형에 강하고 약한지 알기 어렵습니다.

따라서 본 프로젝트에서는 변형 유형별 False Positive와 False Negative를 분석했습니다.

## 2. 관련 코드

```text
src/evaluation/fnfp_count.py
src/evaluation/analyze_transform_type.py
src/evaluation/analyze_by_grade.py
scripts/11_count_fnfp_by_transform.py
scripts/12_analyze_by_grade.py
```

## 3. 분석 기준

분석 대상:

```text
- Cosine Similarity
- SSIM
- pHash
- Siamese Contrastive Model
```

분석 단위:

```text
- transform_type별 FP/FN
- Light/Medium/Heavy별 FP/FN
- 단일 변형 vs 복합 변형
```

## 4. FP/FN 정의

| 구분 | 의미 |
|---|---|
| False Positive | 실제 비표절인데 표절로 예측 |
| False Negative | 실제 표절인데 비표절로 예측 |

표절탐지에서는 특히 False Negative가 중요합니다. 실제 표절 이미지를 놓치는 것이기 때문입니다.

## 5. Baseline 분석

전통적 유사도 기반 방법은 전체적으로 오탐률은 낮았지만 미탐률이 높았습니다.

주요 경향:

```text
- Cosine, SSIM, pHash 모두 flip, rotation, crop 계열에 취약
- 이미지의 밝기, 대비, 해상도 변화는 비교적 잘 탐지
- 물리적 구조 변형에는 baseline 방식의 한계가 큼
```

세부 차이:

```text
- SSIM은 다른 baseline보다 flip에 상대적으로 강함
- pHash는 border와 light crop에 상대적으로 강함
- Cosine은 색상/밝기 변화에 민감함
```

## 6. Siamese 모델 분석

Contrastive Loss 기반 Siamese Network는 대부분의 표절 유형에서 baseline보다 우수했습니다.

다만 다음 유형에서는 상대적으로 취약했습니다.

```text
- heavy crop
```

heavy crop은 원본 이미지의 상당 부분이 제거되기 때문에, 전체 이미지 embedding만으로는 원본과의 관계를 충분히 잡기 어려웠던 것으로 해석됩니다.

## 7. 개선 방향

### 1. Heavy crop 대응

```text
- 부분 이미지 매칭
- patch-level similarity
- local feature matching
- object/region-level embedding 비교
```

### 2. Baseline score 활용 방식 개선

현재 baseline은 독립적인 판단 방식으로 사용했지만, 추후에는 각 baseline score를 feature로 활용할 수 있습니다.

```text
features = [cosine_score, ssim_score, phash_score, siamese_score]
label = plagiarism or non-plagiarism
```

### 3. Meta-classifier

PPT에서 제안한 개선 방향처럼 RandomForestClassifier 등을 사용하여 각 score의 가중치를 자동 학습할 수 있습니다.

```text
Cosine score
SSIM score
pHash score
Siamese score
    ↓
Meta-classifier
    ↓
Final prediction
```

### 4. 후보군 추출 고도화

회사 측 도전 과제와 연결하여, 실제 서비스 환경에서는 전체 DB와 모든 이미지를 비교하기보다 먼저 후보군을 줄이는 과정이 필요합니다.

```text
- embedding vector search
- FAISS
- clustering
- approximate nearest neighbor search
```
