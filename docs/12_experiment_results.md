# 12. Experiment Results

## 1. 실험 목적

본 실험의 목적은 전통적 유사도 기반 방법과 딥러닝 기반 Siamese Network의 이미지 표절탐지 성능을 비교하는 것입니다.

비교 대상:

```text
A 경로: Cosine Similarity, SSIM, pHash
B 경로: Siamese Network
```

## 2. 최종 데이터셋 설정

```text
총 이미지 쌍: 30,000 pairs
train : valid : test = 8 : 1 : 1
positive : negative = 1 : 1
변형 종류 분포 균형화
```

## 3. 전처리 방식

### A 경로: SSIM / pHash

```text
Resize(224x224)
Grayscale(1 channel)
ToTensor()
```

### Cosine Similarity

```text
Resize(224x224)
RGB 유지
ToTensor()
```

### B 경로: Siamese Network

```text
Resize(224x224)
RGB 3채널 보장
ImageNet mean/std normalization
```

## 4. 최종 성능 비교

| Method | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Cosine Similarity | 0.7573 | **0.9923** | 0.5187 | 0.6813 | 0.7689 |
| SSIM | 0.7760 | 0.9165 | 0.6073 | 0.7306 | 0.7903 |
| pHash | 0.7790 | 0.9729 | 0.5740 | 0.7220 | 0.8249 |
| Siamese Contrastive | **0.9660** | 0.9464 | **0.9880** | **0.9667** | **0.9963** |

## 5. Confusion Matrix

최종 Contrastive Siamese 모델의 confusion matrix는 다음과 같습니다.

```text
[[1416,  84],
 [  18, 1482]]
```

해석:

```text
TN = 1416
FP = 84
FN = 18
TP = 1482
```

Recall이 0.988로 매우 높아 실제 표절 이미지를 놓치는 비율이 낮았습니다.

## 6. Baseline 결과 해석

전통적 유사도 기반 방법은 Precision은 높았지만 Recall이 낮았습니다.

즉, 표절이라고 판단한 경우 실제 표절일 가능성은 높았지만, 실제 표절 이미지를 비표절로 놓치는 FN이 많았습니다.

특히 다음 변형에 취약했습니다.

```text
- flip
- rotation
- crop
- heavy crop
```

## 7. Siamese Contrastive 결과 해석

Contrastive Loss 기반 Siamese Network는 이미지 쌍의 embedding distance를 직접 학습했기 때문에 대부분의 변형에 강했습니다.

특히 baseline이 놓치는 물리적 변형에서도 높은 Recall을 보였습니다.

다만 heavy crop은 원본 정보가 크게 손실되기 때문에 상대적으로 어려운 변형으로 확인되었습니다.

## 8. 최종 결론

최종적으로 **Contrastive Loss 기반 Siamese Network**를 핵심 탐지 모델로 선정했습니다.

선정 이유:

```text
1. 가장 높은 F1-score
2. 가장 높은 ROC AUC
3. 낮은 False Negative
4. 다양한 변형 유형에서 baseline 대비 우수한 성능
```

## 9. 개선 방향

1. heavy crop 대응을 위한 local feature matching 추가
2. baseline score와 model score를 feature로 사용하는 meta-classifier 적용
3. FAISS 기반 후보군 검색 구조 추가
4. Table/List 영역까지 확장
5. 실제 논문 표절 사례 기반 검증 데이터 확보
