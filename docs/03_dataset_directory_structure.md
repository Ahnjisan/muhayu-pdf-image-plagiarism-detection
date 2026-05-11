# 03. Dataset Directory Structure

## 1. 개요

프로젝트 데이터셋은 치타 서버 기준으로 다음 루트 경로에 구성되었습니다.

```text
/home/jovyan/Capstone/test_data_set
```

GitHub 레포지토리에서는 저작권 및 용량 문제로 실제 PDF, 추출 이미지, 변형 이미지는 포함하지 않고, 동일한 폴더 구조만 `.gitkeep`으로 유지합니다.

## 2. 전체 구조

```text
test_data_set/
├── README.md
├── Origin/
├── Extracted/
├── Transformed_single/
├── Transformed_score/
└── dataset_split/
```

## 3. 폴더별 역할

| 폴더 | 역할 | GitHub 업로드 여부 |
|---|---|---|
| `Origin/` | 원본 논문 PDF 저장 | 실제 PDF 제외 |
| `Extracted/` | PDF에서 추출한 원본 Figure 이미지 저장 | 실제 이미지 제외 |
| `Transformed_single/` | 단일 변형 이미지 저장 | 실제 이미지 제외 |
| `Transformed_score/` | 점수제 복합 변형 이미지 저장 | 실제 이미지 제외 |
| `dataset_split/` | train/valid/test CSV 및 metadata 저장 | 필요 시 일부 결과만 업로드 |

## 4. Origin 구조

```text
Origin/
└── Biology/
    ├── 2025.04.30.651282v1.full.pdf
    └── ...
```

논문은 분야별로 분류하여 저장했습니다. 회의 기준으로 분야별 5~10개, 총 약 40개 논문 PDF를 수집 대상으로 두었습니다.

## 5. Extracted 구조

```text
Extracted/
└── Biology/
    └── 2025.04.30.651282v1.full/
        └── Figure/
            ├── 2025.04.30.651282v1.full_page9_Figure_1.png
            └── ...
```

`Extracted`는 PubLayNet 기반으로 PDF에서 crop된 Figure 원본 이미지가 저장되는 위치입니다.

## 6. Transformed_single 구조

```text
Transformed_single/
└── Biology/
    └── 2025.04.30.651282v1.full/
        ├── light/
        │   ├── crop/
        │   ├── resolution/
        │   ├── rotation/
        │   └── ...
        ├── medium/
        └── heavy/
```

`Transformed_single`은 변형 유형별 성능 분석과 오탐/미탐 분석을 위해 생성한 단일 변형 데이터입니다.

## 7. Transformed_score 구조

```text
Transformed_score/
└── Biology/
    └── 2025.04.30.651282v1.full/
        └── run_20250520T063444_Medium_5_format_medium__brightness_heavy/
            ├── metadata.txt
            └── Figure/
                └── 2025.04.30.651282v1.full_page9_Figure_1.png
```

`Transformed_score`는 실제 표절 상황을 모사하기 위해 여러 변형을 랜덤 조합하고, 총점 기준으로 Light/Medium/Heavy 등급을 부여한 결과입니다.

## 8. dataset_split 구조

```text
dataset_split/
├── train.csv
├── valid.csv
├── test.csv
├── train_metadata.txt
├── valid_metadata.txt
└── test_metadata.txt
```

최종 재구성 데이터셋은 다음 조건을 목표로 구성했습니다.

```text
총 이미지 쌍: 30,000 pairs
train : valid : test = 8 : 1 : 1
positive : negative = 1 : 1
변형 종류 분포 균형화
```

## 9. CSV 기본 컬럼

```csv
original_path,transformed_path,label,field,paper,transform_type,transform_grade,total_score,transform_combo
```

| 컬럼 | 의미 |
|---|---|
| `original_path` | 원본 이미지 경로 |
| `transformed_path` | 비교 대상 이미지 경로 |
| `label` | 1 = 표절쌍, 0 = 비표절쌍 |
| `field` | 논문 분야 |
| `paper` | 논문명 |
| `transform_type` | crop, rotation, score 등 |
| `transform_grade` | Light, Medium, Heavy |
| `total_score` | 점수제 변형 총점 |
| `transform_combo` | 적용된 변형 조합 |
