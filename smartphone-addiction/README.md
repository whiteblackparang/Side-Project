# Smartphone Addiction Prediction

## 문제 정의

스마트폰 사용 행동 데이터 기반 addicted_label 예측. Kaggle Tabular
Playground Series 데이터셋(합성) 기반, train 691,369행 / test 296,302행.

## 데이터

[Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8/data)
(Kaggle Playground Series S6E8). data/ 폴더에 train.csv, test.csv 배치 필요.

### 데이터 소개

Kaggle Playground Series S6E8, Predicting Smartphone Addiction 데이터셋
(합성). train 691,369행, test 296,302행. id 중복 0건, 완전 중복 행 0건.

## 가설 검증 및 실험

| 단계 | 내용 | 파일 |
|---|---|---|
| 1 | 데이터 구조·결측치·분포 확인, 초기 가설 검증 | notebooks/01_eda.ipynb |
| 2 | 파생변수 생성, Target Encoding | notebooks/02_feature_engineering.ipynb |
| 3 | Baseline(LR) → LightGBM, CV 기반 ROC-AUC 비교 | notebooks/03_modeling.ipynb |
| 4 | SHAP 기반 모델 해석 | notebooks/04_shap.ipynb |

## 결과

- Stratified 5-Fold 기준 LightGBM 평균 ROC-AUC 0.963 (LR baseline 0.917)
- 핵심 변수: daily_screen_time_hours, social_media_hours,
  weekend_screen_time
- 상세 해석 및 한계: report/report.md 참고

## 폴더 구성

- src/preprocessing.py — 결측치 처리, 데이터 품질 체크
- src/features.py — 파생변수, Target Encoding
- src/model.py — 모델 학습 함수
- output/submission.csv — 제출 파일
- report/report.md — 분석 보고서