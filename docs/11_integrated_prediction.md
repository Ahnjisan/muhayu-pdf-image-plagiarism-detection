# 11. Integrated Prediction

## 1. 개요

본 프로젝트의 최종 예측기는 전통적 유사도 기반 방법과 딥러닝 기반 Siamese 모델을 함께 사용합니다.

입력 이미지 쌍에 대해 다음 결과를 동시에 제공합니다.

```text
- Cosine Similarity score / prediction
- SSIM score / prediction
- pHash score / prediction
- Siamese Model score / prediction
```

## 2. 관련 코드

```text
src/detector/predictor.py
scripts/13_predict_integrated.py
```

## 3. 입력

```text
original_path
transformed_path
```

예시:

```bash
python scripts/13_predict_integrated.py \
  --original_path test_data_set/Extracted/Biology/paper/Figure/image.png \
  --transformed_path test_data_set/Transformed_score/Biology/paper/run_xxx/Figure/image.png
```

## 4. 출력 예시

```json
{
  "model": {
    "score": -0.6439,
    "is_plagiarized": true,
    "threshold": -0.6500,
    "model_type": "contrastive"
  },
  "Cosine": {
    "score": 0.9543,
    "is_plagiarized": false,
    "threshold": 0.9903
  },
  "SSIM": {
    "score": 0.2936,
    "is_plagiarized": false,
    "threshold": 0.5669
  },
  "pHash": {
    "score": 0.5312,
    "is_plagiarized": false,
    "threshold": 0.6562
  }
}
```

## 5. threshold 로드

예측기는 다음 threshold 파일을 사용합니다.

```text
outputs/thresholds/baseline_thresholds.json
outputs/thresholds/best_threshold.json
```

baseline threshold 예시:

```json
{
  "Cosine": 0.9933,
  "SSIM": 0.5050,
  "pHash": 0.6250
}
```

model threshold 예시:

```json
{
  "model_type": "contrastive",
  "score_type": "negative_distance",
  "threshold": -0.6439,
  "accuracy": 0.9660,
  "f1_score": 0.9667
}
```

## 6. BCE 모델과 Contrastive 모델의 차이

예측기는 모델 타입에 따라 score 계산 방식을 다르게 적용합니다.

### BCE

```python
logit = model(img1, img2)
score = sigmoid(logit)
pred = score > threshold
```

### Contrastive

```python
emb1, emb2 = model(img1, img2, return_embeddings=True)
distance = L2(emb1, emb2)
score = -distance
pred = score > threshold
```

## 7. 통합 예측의 의미

현재 최종 모델은 Contrastive Loss 기반 Siamese Network이지만, baseline 결과를 함께 출력하는 이유는 다음과 같습니다.

1. 모델 판단 근거를 비교할 수 있음
2. baseline과 model이 불일치하는 사례를 분석할 수 있음
3. 추후 meta-classifier의 feature로 활용 가능
4. 서비스 환경에서 threshold 기반 fallback 전략을 설계할 수 있음

## 8. 향후 개선

PPT에서 제안한 개선 방향은 각 score를 feature로 사용하는 meta-classifier입니다.

```text
[Cosine, SSIM, pHash, Siamese]
    ↓
RandomForestClassifier / LogisticRegression
    ↓
Final plagiarism prediction
```

이 방식은 사람이 직접 threshold나 가중치를 조정하지 않아도, 데이터 기반으로 최종 판단을 자동화할 수 있습니다.
