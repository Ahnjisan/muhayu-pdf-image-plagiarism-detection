# 02. PubLayNet 기반 PDF Figure 추출

## 1. 목적

본 프로젝트에서는 논문 PDF에서 표절탐지 대상이 되는 이미지를 자동으로 추출하기 위해 **PubLayNet 기반 레이아웃 검출 모델**을 사용했습니다.

PubLayNet은 논문 페이지의 레이아웃을 다음과 같은 클래스로 검출할 수 있습니다.

```text
Text, Title, List, Table, Figure
```

회의 초기에는 Table, Figure, List를 추출 대상으로 고려했지만, 최종 이미지 표절탐지 실험에서는 **Figure 중심**으로 데이터셋을 구축했습니다. Table/List는 향후 도전 과제로 확장 가능한 영역입니다.

## 2. 관련 코드

```text
src/extraction/extract_pdf_images.py
```

이 코드는 다음 작업을 수행합니다.

1. PDF 파일을 페이지 이미지로 변환
2. PubLayNet/Detectron2 Layout Model을 사용해 레이아웃 객체 탐지
3. 검출된 객체 중 필요한 label만 crop
4. 분야명/논문명/Figure 구조로 저장

## 3. 처리 흐름

```text
Origin/{field}/{paper}.pdf
    ↓
pdf2image.convert_from_path()
    ↓
PubLayNet Layout Detection
    ↓
Figure 영역 crop
    ↓
Extracted/{field}/{paper}/Figure/*.png
```

## 4. 입력 구조

```text
test_data_set/
└── Origin/
    └── Biology/
        ├── paper_001.pdf
        ├── paper_002.pdf
        └── ...
```

## 5. 출력 구조

```text
test_data_set/
└── Extracted/
    └── Biology/
        └── 2025.04.30.651282v1.full/
            └── Figure/
                ├── 2025.04.30.651282v1.full_page9_Figure_1.png
                └── 2025.04.30.651282v1.full_page10_Figure_1.png
```

## 6. 핵심 설정

```yaml
# configs/extraction_config.yaml
pdf_dpi: 300
score_threshold: 0.8
target_labels:
  - Figure
```

## 7. 구현 시 고려사항

- PDF마다 레이아웃 구조가 다르므로 검출 threshold가 너무 낮으면 불필요한 영역이 추출될 수 있습니다.
- 너무 높은 threshold를 사용하면 실제 Figure를 놓칠 수 있습니다.
- 실험에서는 PubLayNet이 논문 구조 분석에 적합하다고 판단해 사용했습니다.
- 추출된 이미지는 이후 표절 변형 데이터셋의 원본 이미지로 사용됩니다.

## 8. 향후 확장

본 실험에서는 Figure 중심으로 진행했지만, 회사 측 도전 과제인 **표와 차트를 포함한 표절 여부 탐지**를 위해서는 아래 label까지 확장할 수 있습니다.

```text
Table, List, Figure
```

이 경우 `configs/extraction_config.yaml`의 `target_labels`만 수정하여 확장할 수 있습니다.
