# 08. Siamese Network with BCE Loss

## 1. 개요

초기 딥러닝 모델은 **BCE 기반 Siamese Network**로 구현했습니다.

두 이미지가 입력되면 동일한 CNN backbone을 통과해 각각 임베딩을 만들고, 두 임베딩 차이를 MLP에 입력하여 표절 확률을 예측합니다.

## 2. 관련 코드

```text
src/model/model.py
src/model/train.py
src/model/evaluate.py
```

## 3. 모델 구조

```text
Image A ─┐
         ├─ Shared ResNet18 Backbone ─ f1 ─┐
Image B ─┘                                  │
                                            ├─ |f1 - f2| ─ MLP ─ logit ─ sigmoid ─ score
Image B ─┐                                  │
         ├─ Shared ResNet18 Backbone ─ f2 ─┘
Image A ─┘
```

## 4. BCE 학습 방식

손실 함수:

```python
nn.BCEWithLogitsLoss()
```

label 정의:

```text
1 = 표절쌍
0 = 비표절쌍
```

예측 방식:

```python
prob = sigmoid(logit)
pred = prob > threshold
```

## 5. threshold 최적화

일반적으로 binary classification에서는 threshold 0.5를 사용하지만, 데이터 분포에 따라 0.5가 최적이 아닐 수 있습니다.

따라서 validation set에서 ROC Curve를 구하고, Youden's J가 최대가 되는 threshold를 사용했습니다.

```text
Youden's J = TPR - FPR
```

## 6. 기존 데이터셋에서의 BCE 결과

| Metric | Value |
|---|---:|
| Accuracy | 0.9662 |
| Precision | 0.9504 |
| Recall | 0.9628 |
| F1-score | 0.9566 |
| ROC AUC | 0.9924 |

## 7. 재구성 데이터셋에서의 BCE 결과

| Metric | Value |
|---|---:|
| Accuracy | 0.9370 |
| Precision | 0.9079 |
| Recall | 0.9727 |
| F1-score | 0.9392 |
| ROC AUC | 0.9919 |

## 8. 해석

BCE 기반 Siamese Network는 baseline 대비 높은 성능을 보였지만, 데이터셋을 더 균형 있게 재구성한 후에는 Precision과 Accuracy가 일부 낮아졌습니다.

이는 더 어려운 negative pair와 균형 잡힌 transform_type 분포에서 모델이 더 엄격하게 평가되었기 때문으로 해석됩니다.

이후 이미지 표절탐지 문제의 특성을 반영하기 위해 거리 기반 학습 방식인 Contrastive Loss를 추가 실험했습니다.
