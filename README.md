# PDF Image Plagiarism Detection System

논문 PDF 이미지 기반 표절탐지 시스템입니다. 본 프로젝트는 무하유 측에서 제시한 이미지 표절탐지 과제를 바탕으로, 논문 PDF에서 Figure 영역을 자동 추출하고 표절 정의에 맞는 변형 이미지 데이터셋을 구축한 뒤, 전통적 유사도 기반 방법과 Siamese Network 기반 딥러닝 모델을 비교 실험했습니다.

## 📋 프로젝트 개요

본 프로젝트의 목표는 논문 PDF 내부의 이미지, 차트, 표 형태 자료 중 이미지 표절 가능성이 있는 Figure를 추출하고, 원본 이미지와 변형 이미지를 비교하여 표절 여부를 판정하는 것입니다.

전체 흐름은 다음과 같습니다.

```text
논문 PDF 수집
→ PubLayNet 기반 Figure 영역 추출
→ 표절 정의에 따른 이미지 변형 데이터셋 생성
→ 원본-변형 이미지 쌍 CSV 구축
→ 전통적 유사도 기반 탐지 실험
→ Siamese Network 기반 딥러닝 탐지 실험
→ Threshold 최적화 및 오류 분석
→ 통합 예측기 구현
```

## 🎯 핵심 과제 및 도전 과제

### 핵심 과제

1. 이미지 표절 정의 구체화
2. 원본 이미지 500개 이상 수집 및 DB 구축
3. 원본 이미지에서 파생된 표절 정의 기반 표절 이미지 데이터셋 구축
4. 이미지 표절 탐지 알고리즘 개발
   - 휴리스틱 기반 유사도 알고리즘
   - 딥러닝 기반 이미지 쌍 비교 모델

### 도전 과제

1. 표와 차트를 포함한 표절 여부 탐지
2. DB에서 후보군 추출
   - 클러스터링
   - 벡터 서치
   - 임베딩 기반 검색

## 🔍 표절 정의

회사 측 표절 정의와 회의 내용을 바탕으로 이미지 변형 유형을 다음과 같이 구성했습니다.

| 변형 종류 | light | medium | heavy |
|---|---|---|---|
| 자르기 | 10% crop | 20% crop | 30% crop |
| 해상도 | 해상도 10% 줄이기 | 해상도 20% 줄이기 | 해상도 30% 줄이기 |
| 회전 | 90도 회전 | 270도 회전 | - |
| 반전 | 좌우 반전 | 상하 반전 | 좌우+상하 반전 |
| 캡쳐 | black border 10% | black border 20% | black border 30% |
| 흑백 | 흑백 변환 | - | - |
| 밝기 | 밝기 10% 높이기 | 밝기 20% 높이기 | 밝기 30% 높이기 |
| 대조 | 대조 10% 높이기 | 대조 20% 높이기 | 대조 30% 높이기 |
| 포맷 | jpg → jpeg | jpg → png | jpg → webp |
| 삽입 | 다른 이미지 위에 원본 이미지 삽입 | - | - |
| 붙이기 | 여러 이미지를 가로/세로로 이어 붙이기 | - | - |

표절 강도는 각 변형의 난이도와 강도를 점수화하여 Light / Medium / Heavy로 분류했습니다.

```text
light = 1점
medium = 2점
heavy = 3점

총점 1~3점  → Light
총점 4~6점  → Medium
총점 7점 이상 → Heavy
```

예시:

```text
crop medium = 2점
resolution light = 1점
format light = 1점
총합 4점 → Medium
```

## 🏗️ 프로젝트 구조

```text
pdf-image-plagiarism-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── configs/                         # 실행 설정 파일
│   ├── extraction_config.yaml
│   ├── transform_config.yaml
│   ├── dataset_config.yaml
│   ├── baseline_config.yaml
│   └── model_config.yaml
│
├── test_data_set/                   # 실제 데이터 배치 구조, 대용량 파일은 Git 제외
│   ├── README.md
│   ├── Origin/                      # 추출 대상 논문 PDF
│   ├── Extracted/                   # PubLayNet으로 추출한 Figure 이미지
│   ├── Transformed_single/          # 단일 변형 데이터, 변형별 오류 분석용
│   ├── Transformed_score/           # 점수제 랜덤 조합 변형 데이터
│   └── dataset_split/               # train/valid/test CSV
│
├── src/
│   ├── extraction/                  # PDF → Figure 추출
│   ├── transform/                   # 단일 변형 / 점수제 변형 생성
│   ├── data/                        # CSV 생성, Dataset, DataLoader, 전처리
│   ├── baseline/                    # Cosine, SSIM, pHash 기반 탐지
│   ├── model/                       # Siamese Network 학습/평가
│   ├── detector/                    # 통합 예측기
│   ├── evaluation/                  # FP/FN, 변형별, 등급별 분석
│   └── utils/                       # 공통 유틸리티
│
├── scripts/                         # 실행 진입점
├── outputs/                         # checkpoint, threshold, prediction, analysis 결과
└── docs/                            # 프로젝트 문서
```

## 📊 데이터셋 구성

### 1. Origin

논문 PDF 원본을 저장하는 공간입니다.

```text
test_data_set/Origin/<분야>/<논문 PDF 파일>
```

### 2. Extracted

PubLayNet 기반 레이아웃 검출 모델을 사용하여 PDF에서 Figure 영역을 추출한 결과입니다.
본 실험에서는 PubLayNet의 5개 레이블 중 Figure를 중심으로 사용했습니다.

```text
test_data_set/Extracted/<분야>/<논문명>/Figure/<이미지 파일>
```

### 3. Transformed_single

정밀도, 재현율, 오차 분석을 위해 단일 변형을 적용한 데이터입니다.

```text
test_data_set/Transformed_single/<분야>/<논문명>/<강도>/<변형종류>/Figure/<이미지 파일>
```

### 4. Transformed_score

실제 표절 상황을 모사하기 위해 2~4개의 변형을 랜덤 조합하고, 점수 기준으로 Light / Medium / Heavy 등급을 자동 부여한 데이터입니다.

```text
test_data_set/Transformed_score/<분야>/<논문명>/run_타임스탬프_등급_총점_변형조합/Figure/<이미지 파일>
```

예시:

```text
test_data_set/Transformed_score/Biology/2025.04.30.651282v1.full/
run_20250520T063444_Medium_5_format_medium__brightness_heavy/
Figure/2025.04.30.651282v1.full_page9_Figure_1.png
```

### 5. dataset_split

원본 이미지와 비교 이미지를 쌍으로 구성한 CSV입니다.

```csv
original_path,transformed_path,label,field,paper,transform_type,transform_level,transform_grade,total_score,transform_combo
```

- `label=1`: 원본에서 파생된 표절 이미지
- `label=0`: 서로 다른 이미지 간 비표절 쌍
- `transform_type`: crop, rotation, flip 등 단일 변형 또는 score 기반 조합
- `transform_grade`: Light / Medium / Heavy

## 🔬 연구 진행 방식

### 초반

- 논문 PDF 오픈 사이트 및 수집 키워드 조사
- 데이터셋 구축 방법 및 모델 알고리즘 조사
- PDF 레이아웃 검출 모델로 PubLayNet 결정
- 치타 서버에 PDF 및 코드 업로드
- `Origin`, `Extracted`, `Transformed_single`, `Transformed_score` 구조 설계

### 중반

- PubLayNet 기반 논문 Figure 자동 추출 구현
- 회사 측 표절 정의에 맞는 변환 스크립트 작성
- 단일 변형 데이터와 점수 기반 랜덤 조합 변형 데이터 생성
- Cosine Similarity, SSIM, pHash baseline 구현
- validation set 기반 ROC Curve와 Youden's J로 method별 threshold 최적화

### 후반

- BCE 기반 Siamese Network 학습
- 데이터셋 분포 불균형 문제 발견
- transform_type별 label 0/1 균형을 맞춘 데이터셋으로 재구성
- Contrastive Loss 기반 Siamese Network 재학습
- 변형 유형별 FP/FN 분석
- 최종 통합 예측기 구현

## 🧠 탐지 방법

### A. 전통적 유사도 기반 방법

1. Cosine Similarity
   - 이미지 벡터 방향 차이를 기반으로 유사도 계산
   - 색상과 밝기 변화에 민감
2. SSIM
   - 구조적 유사도 기반 비교
   - 회전, crop, 왜곡에 민감
3. pHash
   - 이미지 해시값 차이를 기반으로 빠르게 비교
   - 속도는 빠르지만 세밀한 변형에는 한계

각 방법은 동일한 threshold를 쓰면 성능이 왜곡될 수 있어, validation set에서 ROC Curve와 Youden's J를 이용해 개별 threshold를 산출했습니다.

### B. Siamese Network

두 이미지를 같은 backbone에 통과시켜 임베딩을 추출하고, 두 이미지의 차이 또는 거리를 기반으로 표절 여부를 판단했습니다.

- BCE 기반 Siamese Network
- Contrastive Loss 기반 Siamese Network

최종 모델은 데이터셋 재구성 이후 Contrastive Loss 기반 Siamese Network를 중심으로 정리했습니다.

## 📈 실험 결과 요약

| 실험 | 모델 | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| 기존 데이터셋 | BCE Siamese | 0.9662 | 0.9504 | 0.9628 | 0.9566 | 0.9924 |
| 재구성 데이터셋 | BCE Siamese | 0.9370 | 0.9079 | 0.9727 | 0.9392 | 0.9919 |
| 재구성 데이터셋 | Contrastive Siamese | 0.9660 | 0.9464 | 0.9880 | 0.9667 | 0.9963 |

전통적 유사도 기반 방법은 오탐률은 낮은 편이었지만, 회전, crop, border, 복합 변형에서 미탐이 증가했습니다. Siamese 모델은 대부분의 변형에 강했으며, 최종적으로 Contrastive Loss 기반 모델이 가장 안정적인 성능을 보였습니다.

## 🚀 실행 순서

```bash
# 1. PDF에서 Figure 추출
python scripts/01_extract_pdf_images.py

# 2. 단일 변형 데이터 생성
python scripts/02_transform_single.py

# 3. 점수 기반 랜덤 조합 변형 데이터 생성
python scripts/03_transform_score.py

# 4. train/valid/test CSV 생성
python scripts/04_generate_dataset_csv.py

# 5. 데이터셋 분포 확인
python scripts/05_check_dataset_stats.py

# 6. baseline 기본 평가
python scripts/06_evaluate_baseline_default.py

# 7. baseline threshold 저장
python scripts/07_save_baseline_thresholds.py

# 8. threshold 적용 baseline 결과 생성
python scripts/08_generate_baseline_results.py

# 9. Siamese Network 학습
python scripts/09_train_siamese.py --loss_type contrastive

# 10. Siamese Network 평가
python scripts/10_evaluate_siamese.py

# 11. 변형 유형별 FP/FN 분석
python scripts/11_count_fnfp_by_transform.py

# 12. 등급별 성능 분석
python scripts/12_analyze_by_grade.py

# 13. 통합 예측 실행
python scripts/13_predict_integrated.py --original_path <원본 이미지> --transformed_path <비교 이미지>
```

## ⚠️ 주의사항

- 논문 PDF, 추출 이미지, 변형 이미지, 학습 데이터 CSV는 저작권 및 용량 문제로 GitHub에 포함하지 않습니다.
- 모델 가중치(`.pth`, `.pt`, `.ckpt`)도 GitHub에 포함하지 않습니다.
- 레포지토리에는 데이터 구축, 변형 생성, baseline 탐지, 모델 학습, 평가 분석에 필요한 코드와 문서만 포함합니다.
- 빈 폴더 구조는 `.gitkeep`으로 유지합니다.

## 📁 Git 관리

`.gitignore`를 통해 다음 항목을 제외합니다.

- PDF, 이미지 등 원본/생성 데이터
- 학습된 모델 가중치
- prediction, analysis, figure 결과 파일
- Python cache, pytest cache, 가상환경
- IDE 설정 및 로그 파일

## 👥 팀원

- **안지산**
- **변개령**
- **임영재**
