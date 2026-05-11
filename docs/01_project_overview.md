# 01. Project Overview

## 1. 프로젝트 개요

본 프로젝트는 **논문 PDF 이미지 기반 표절탐지 시스템** 구축을 목표로 진행되었습니다.

최근 논문, 학술 리포트, 연구자료에서 이미지 표절 문제가 중요해지고 있지만, 실제 표절 판별은 여전히 사람의 수작업 검토에 의존하는 경우가 많습니다. 특히 이미지가 자르기, 회전, 반전, 밝기/대조 조절, 포맷 변경 등 단순 변형을 거치면 사람이 아닌 알고리즘으로는 원본과의 관계를 탐지하기 어려워집니다.

본 프로젝트에서는 논문 PDF에서 이미지를 추출하고, 회사 측 표절 정의에 맞는 변형 이미지를 생성한 뒤, 전통적 유사도 기반 방법과 딥러닝 기반 방법을 비교했습니다.

## 2. 회사 측 핵심 과제 및 도전 과제

### 핵심 과제

1. 이미지 표절 정의 구체화
2. 원본 이미지 500개 이상 수집 및 DB 구축
3. 원본 이미지에서 파생된 표절 정의 기반 표절 이미지 데이터셋 구축
4. 이미지 표절 탐지 알고리즘 개발
   - 휴리스틱 기반 접근 가능
   - 딥러닝 모델 기반 접근 가능

### 도전 과제

1. 표와 차트를 포함한 이미지 표절 여부 탐지
2. DB에서 후보군 추출
   - 클러스터링
   - 벡터 서치
   - 임베딩 기반 검색

## 3. 프로젝트 목표

본 프로젝트의 목표는 다음과 같습니다.

1. PubLayNet 기반으로 논문 PDF에서 Figure 이미지를 자동 추출한다.
2. 회사 측 표절 정의에 맞춰 단일 변형 및 점수제 복합 변형 이미지를 생성한다.
3. 원본 이미지와 변형 이미지를 쌍으로 구성하여 표절/비표절 이진 분류 데이터셋을 구축한다.
4. Cosine Similarity, SSIM, pHash 기반 baseline을 구현한다.
5. Siamese Network 기반 딥러닝 모델을 구현한다.
6. BCE Loss와 Contrastive Loss를 비교한다.
7. 변형 유형별 오탐/미탐을 분석해 각 방식의 한계를 파악한다.
8. 최종적으로 baseline 결과와 딥러닝 모델 결과를 통합해 표절 여부를 예측하는 구조를 설계한다.

## 4. 전체 시스템 파이프라인

```text
논문 PDF 수집
    ↓
PubLayNet 기반 Figure 영역 추출
    ↓
Extracted 디렉토리에 원본 이미지 저장
    ↓
회사 측 표절 정의 기반 이미지 변형
    ├─ Transformed_single: 단일 변형 데이터
    └─ Transformed_score: 점수제 복합 변형 데이터
    ↓
원본-변형 이미지 쌍 CSV 생성
    ↓
A 경로: Cosine / SSIM / pHash baseline
B 경로: Siamese Network 모델
    ↓
ROC Curve + Youden's J 기반 threshold 최적화
    ↓
변형 유형별 FP/FN 분석
    ↓
통합 예측 및 최종 판단
```

## 5. 주요 코드 위치

| 영역 | 경로 | 설명 |
|---|---|---|
| PDF 이미지 추출 | `src/extraction/extract_pdf_images.py` | PubLayNet 기반 Figure 추출 |
| 단일 변형 생성 | `src/transform/image_transform_single.py` | crop, rotation 등 개별 변형 생성 |
| 점수제 변형 생성 | `src/transform/image_transform_score.py` | 2~4개 변형 랜덤 조합 및 등급 부여 |
| CSV 생성 | `src/data/generate_dataset_csv.py` | train/valid/test 이미지 쌍 생성 |
| 전처리 | `src/data/preprocess.py` | A/B 경로별 전처리 분리 |
| baseline | `src/baseline/` | Cosine, SSIM, pHash 평가 |
| Siamese 모델 | `src/model/` | BCE/Contrastive 기반 모델 학습 |
| 통합 예측 | `src/detector/predictor.py` | baseline + model 통합 예측 |
| 오류 분석 | `src/evaluation/` | 변형 유형별 FP/FN 분석 |
