# 09. Siamese Network with Contrastive Loss

## 1. 개요

최종 딥러닝 모델은 **Contrastive Loss 기반 Siamese Network**입니다.

BCE 기반 모델은 두 이미지의 임베딩 차이를 MLP에 넣어 표절 확률을 예측하는 방식이지만, Contrastive Loss는 두 이미지 임베딩 간 거리를 직접 학습합니다.

## 2. 관련 코드

```text
src/model/model.py
src/model/losses.py
src/model/train.py
src/model/evaluate.py
src/model/threshold.py
```

## 3. 학습 목표

```text
표절쌍(label=1): 임베딩 거리 ↓
비표절쌍(label=0): 임베딩 거리 ↑
```

즉, 같은 원본에서 파생된 이미지는 embedding space에서 가깝게, 서로 관련 없는 이미지는 멀게 학습합니다.

## 4. 모델 구조

```text
Image A ─┐
         ├─ Shared ResNet18 Backbone ─ Embedding A
Image B ─┘

Image B ─┐
         ├─ Shared ResNet18 Backbone ─ Embedding B
Image A ─┘

Distance = L2(Embedding A, Embedding B)
```

## 5. Contrastive Loss

```python
positive_loss = label * distance.pow(2)
negative_loss = (1 - label) * clamp(margin - distance, min=0).pow(2)
loss = mean(positive_loss + negative_loss)
```

## 6. threshold 계산 시 주의점

Contrastive Loss에서는 거리가 작을수록 표절입니다.

하지만 ROC Curve는 score가 클수록 positive라고 해석합니다.

따라서 validation score 계산 시 distance를 그대로 쓰지 않고, `-distance`를 사용했습니다.

```python
score = -distance
```

예측:

```python
pred = score > threshold
```

## 7. 최종 성능

재구성 데이터셋 기준 최종 Contrastive Siamese 결과입니다.

| Metric | Value |
|---|---:|
| Accuracy | 0.9660 |
| Precision | 0.9464 |
| Recall | 0.9880 |
| F1-score | 0.9667 |
| ROC AUC | 0.9963 |

## 8. BCE와 비교

| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Siamese BCE | 0.9370 | 0.9079 | 0.9727 | 0.9392 | 0.9919 |
| Siamese Contrastive | **0.9660** | **0.9464** | **0.9880** | **0.9667** | **0.9963** |

## 9. 해석

Contrastive Loss는 이미지 표절탐지 문제와 잘 맞았습니다. 표절탐지는 본질적으로 “두 이미지가 같은 원본에서 파생되었는가”를 판단하는 문제이기 때문에, 단순 확률 분류보다 embedding distance를 직접 학습하는 방식이 더 적합했습니다.

최종적으로 Contrastive Loss 기반 Siamese Network를 핵심 딥러닝 모델로 선정했습니다.
