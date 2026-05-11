# Documentation Index

이 폴더는 **PDF 논문 이미지 기반 표절탐지 프로젝트**의 설계, 데이터셋 구축, 모델 실험, 결과 분석을 정리한 문서 모음입니다.

본 프로젝트는 논문 PDF에서 Figure 영역을 자동 추출하고, 회사 측에서 제시한 이미지 표절 정의를 기반으로 표절 변형 데이터셋을 구축한 뒤, 전통적 유사도 기반 방법과 딥러닝 기반 Siamese Network를 비교한 프로젝트입니다.

## 문서 구성

| 문서 | 내용 |
|---|---|
| `01_project_overview.md` | 프로젝트 목표, 핵심 과제, 전체 파이프라인 |
| `02_pubLayNet_extraction.md` | PubLayNet 기반 PDF Figure 추출 과정 |
| `03_dataset_directory_structure.md` | 치타 서버 기준 데이터셋 디렉토리 구조 |
| `04_transformation_rules.md` | 회사 측 표절 정의와 변형 기준 |
| `05_score_based_transformation.md` | 단일 변형과 점수제 복합 변형 생성 방식 |
| `06_baseline_similarity.md` | Cosine, SSIM, pHash baseline 방식 |
| `07_dataset_rebalancing.md` | 데이터셋 불균형 문제와 30,000쌍 재구성 |
| `08_siamese_bce.md` | BCE 기반 Siamese Network 실험 |
| `09_siamese_contrastive.md` | Contrastive Loss 기반 Siamese Network 최종 실험 |
| `10_error_analysis.md` | 변형 유형별 FP/FN 분석 및 취약점 |
| `11_integrated_prediction.md` | baseline + model 통합 예측기 구조 |
| `12_experiment_results.md` | 최종 성능 결과 요약 |

## 핵심 결과 요약

| Method | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Cosine Similarity | 0.7573 | 0.9923 | 0.5187 | 0.6813 | 0.7689 |
| SSIM | 0.7760 | 0.9165 | 0.6073 | 0.7306 | 0.7903 |
| pHash | 0.7790 | 0.9729 | 0.5740 | 0.7220 | 0.8249 |
| Siamese Network + Contrastive Loss | **0.9660** | 0.9464 | **0.9880** | **0.9667** | **0.9963** |

최종적으로 **Contrastive Loss 기반 Siamese Network**가 전통적 유사도 기반 방법과 BCE 기반 Siamese 모델 대비 가장 높은 F1-score와 ROC AUC를 보였습니다.
