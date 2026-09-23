# Side Projects (Kaggle 데이터 기반)

Kaggle 데이터셋을 활용해 진행한 개인 사이드 프로젝트 모음.

| 프로젝트 | 문제 유형 | 핵심 기법 | 결과 |
|---|---|---|---|
| [Smartphone Addiction Prediction](./smartphone-addiction) | 이진 분류 | LightGBM, SHAP | ROC-AUC 0.963 |

## 프로젝트별 요약


### Smartphone Addiction Prediction

**문제 상황**
스마트폰 사용 행동과 중독 위험의 관계, 알려진 통념과 실제 데이터 간 간극 확인 필요. 화면 사용시간·SNS·게임·알림 빈도 등 후보 지표 중 실제 예측에 기여하는 변수 파악이 목적.

**도구 및 분석 방법**
Python(pandas, LightGBM, SHAP), Stratified 5-Fold CV. EDA → Feature Engineering → Baseline(LR)·LightGBM 비교 → SHAP 해석 순으로 진행.

**핵심 인사이트**
- daily_screen_time_hours 상관계수 0.61, 시간 계열 변수가 빈도 지표보다 압도적 우위
- gaming_ratio 상관계수 -0.19, "게임 비중 높음 = 위험"이라는 통념과 반대 방향
- screen_time_per_sleep 상관계수 0.48이지만 원본 변수와 상관 0.86 → 다중공선성으로 gain importance 최하위권 (단변량·다변량 지표 차이 직접 확인)

**결과**
Stratified 5-Fold 기준 LightGBM ROC-AUC 0.963 (LR baseline 0.917 대비 +0.046). 화면·SNS 사용시간 중심 모니터링 지표 우선순위화 가능성 제시.

**배운 점 & 개선하고 싶은 부분**
단변량 상관계수와 트리 모델의 gain importance가 다른 것을 측정한다는 점을 다중공선성 사례로 체감. 다음에는 SHAP을 전체 데이터 기준으로 재계산하고, CatBoost 등 모델을 추가해 OOF 기반 앙상블까지 검증 예정.

---
