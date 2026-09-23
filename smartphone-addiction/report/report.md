# 스마트폰 사용 행동 기반 중독 위험 예측

## 1. 문제 정의

### 1.1 프로젝트 배경

스마트폰 사용시간 증가에 따른 사용 빈도, 앱 사용 패턴, SNS·게임 이용시간,
수면시간, 스트레스 수준 등 행동 데이터와 스마트폰 의존도 간의 관계 확인
필요성. 어떤 행동이 addicted_label과 함께 움직이는지, 이 패턴을 예측
모델로 재현 가능한지가 이 프로젝트의 출발점.

### 1.2 데이터 소개

Kaggle Tabular Playground Series 제공 합성 데이터셋. train 691,369행,
test 296,302행. id 중복 0건, 완전 중복 행 0건. 실제 사용자 전체를 대표하는
자료가 아니므로, "특정 행동이 중독의 원인"이 아닌 "주어진 데이터에서 특정
행동과 addicted_label 사이 나타나는 패턴"으로 해석.

### 1.3 분석 목표 및 핵심 질문

목표 세 가지 — (1) 사용자 스마트폰 이용 행태 파악, (2) addicted_label과
관련된 행동 특성 탐색, (3) 예측 모델 구축. 이 중 두 번째가 핵심. ROC-AUC
수치 자체보다, 어떤 행동이 왜 중요한 신호로 작동하는지에 대한 설명력이
개입 전략 설계의 근거가 됨.

---

## 2. 가설 검증 및 실험

### 2.1 데이터 품질 분석

id 중복 0건, 완전 중복 행 0건 — 데이터 무결성 문제 없음. 컬럼별 결측률
4~19% 수준, social_media_hours 최고.

결측치 대체 전 결측 여부와 addicted_label의 관련성 확인. Target 그룹별
결측률 차이 전 컬럼 0.2%p 미만. 동시 결측 개수별 addicted_label 평균도
0.70~0.72 구간에 밀집 — 결측의 무작위성(MCAR에 가까움) 확인.

초기 가설("결측 자체가 신호") 기각. 결측치는 중앙값/최빈값으로 대체하되,
결측 indicator는 핵심이 아닌 보조 검증용으로 한정.

### 2.2 EDA

**Target 분포**: addicted_label=1 약 71%, 0 약 29% — 불균형 존재. CV는
Stratified K-Fold로 고정, 평가 지표는 Accuracy 대신 ROC-AUC로 확정.

**수치형 변수 vs Target 상관계수**

| 변수 | 상관계수 |
|---|---|
| daily_screen_time_hours | 0.611 |
| weekend_screen_time | 0.590 |
| social_media_hours | 0.532 |
| work_study_hours | 0.251 |
| gaming_hours | 0.205 |
| app_opens_per_day | 0.063 |
| sleep_hours | 0.043 |
| notifications_per_day | -0.012 |

화면 사용시간·SNS 사용시간 계열의 확실한 신호, 알림·수면시간의 낮은
단독 관련성. "빈도(알림·앱실행)가 시간보다 중요할 것"이라는 초기 가설
기각 — 시간 계열의 압도적 우위.

boxplot 확인 결과, daily_screen_time_hours·weekend_screen_time의 뚜렷한
중앙값 차이 대비 gaming_hours의 상대적으로 좁은 분포 차이. 상관계수 단독
판단의 위험성 — 분포 확인의 필요성.

**범주형 변수 vs Target**: gender, stress_level, academic_work_impact
모두 그룹별 addicted_label 비율 0.70 안팎 — 단독 구분력 약함.
stress_level별 daily_screen_time_hours 재확인 결과, addicted_label 0/1
간 차이 폭의 stress_level에 따른 유의미한 변화 없음 — 기대했던 상호작용
효과 미미.

### 2.3 가설 및 검증

| 초기 가설 | 검증 결과 |
|---|---|
| 결측 자체가 신호 | 기각 (Target별 결측률 차이 0.2%p 미만) |
| 빈도가 시간보다 중요 | 기각 (시간 계열 상관계수 압도적 우위) |
| 범주형 × 시간 변수 상호작용 | 뚜렷한 확인 실패 |

### 2.4 Feature Engineering

01_eda 결론 기반 파생변수 5종 생성.

- social_gaming_hours = social_media_hours + gaming_hours
- social_media_ratio = social_media_hours / daily_screen_time_hours
- gaming_ratio = gaming_hours / daily_screen_time_hours
- weekend_weekday_gap = weekend_screen_time - daily_screen_time_hours
- screen_time_per_sleep = daily_screen_time_hours / sleep_hours

| 파생변수 | 상관계수 |
|---|---|
| social_gaming_hours | 0.48 |
| screen_time_per_sleep | 0.48 |
| social_media_ratio | 0.10 |
| weekend_weekday_gap | 0.00 |
| gaming_ratio | -0.19 |

예상 밖 결과 두 가지.

- weekend_weekday_gap 상관계수 0에 근접 — 주말 사용량 증가폭과
  addicted_label의 무관성. 핵심 변수에서 제외.
- gaming_ratio 음의 상관(-0.19) — 게임 비중 높을수록 addicted_label=0에
  근접하는 역방향. "게임 많을수록 중독 위험 높음"이라는 직관과 반대.
  SHAP 단계에서 추가 검증 대상.

범주형 변수는 KFold 기반 Target Encoding으로 수치화. boxplot 상 barh
대비 소폭의 차이 확대 확인, 다만 시간 계열 변수 대비 낮은 구분력.

### 2.5 모델링

Logistic Regression baseline 후 LightGBM 진행. LR을 먼저 둔 이유 —
이후 모델이 잡아내는 관계의 비선형성 여부에 대한 비교 기준 확보.

| 모델 | Fold 평균 ROC-AUC | Fold 편차 |
|---|---|---|
| Logistic Regression | 0.9166 | 0.9157~0.9172 |
| LightGBM | 0.9629 | 0.9622~0.9638 |

LightGBM의 LR 대비 0.046p 상승. 두 모델 모두 fold 간 편차 0.001~0.002
수준 — 특정 fold 이상치나 분할 불안정성 없음.

### 2.6 모델 평가

LR baseline만으로 0.92 근접 AUC — 핵심 변수와 addicted_label의 상당한
선형적 관계. LightGBM의 추가 0.046p 상승 — 비선형 관계·변수 간 상호작용의
실질적 기여(예: gaming_ratio 방향 반전).

**Feature Importance (gain) 상위 5**

1. daily_screen_time_hours (압도적 1위)
2. social_media_hours
3. weekend_screen_time
4. notifications_per_day
5. app_opens_per_day

2.4 결과와의 두 가지 불일치.

- screen_time_per_sleep, social_gaming_hours: 단독 상관계수 0.48 대비
  gain importance 최하위권. daily_screen_time_hours와의 상관관계 각각
  0.86, 0.62 확인 — 예측력 부재가 아닌 정보 중복에 의한 gain 배분 감소.
  단변량 상관계수와 다변량 gain importance의 측정 대상 차이 확인.
- notifications_per_day: 단독 상관계수 -0.01 대비 gain importance 4위 —
  조건부 신호 가능성. SHAP dependence plot으로 추가 확인.

SHAP summary plot 상 상위 변수 방향성과 gain importance 순위 일치.
개별 사용자 waterfall plot 사례(idx=0) — social_media_hours(0.78, 낮음),
daily_screen_time_hours(7.77, 낮음)의 각각 -1.38, -1.36 기여, 유일하게
app_opens_per_day(149, 높음)의 +0.22 반대 방향 기여. 시간 계열 우위 결론과
개별 사례 단위 일치 확인.

---

## 3. 결과

### 3.1 모델 해석

addicted_label에 대한 최상위 설명 변수 — 절대적 사용 시간
(daily_screen_time_hours, social_media_hours, weekend_screen_time).
사용 빈도(알림, 앱 실행)는 단독 약신호, 결합 조건 하 트리 모델의 보조
활용 신호. 게임 비중(gaming_ratio)의 낮은 addicted_label 연관성 — SNS
중심군과 게임 중심군의 이질적 성격 가능성.

### 3.2 사용자 행동 패턴

- 화면·SNS 사용시간 상위군의 addicted_label=1 비율 뚜렷한 상승
- 앱 실행·알림 빈도 단독 약신호, 화면 사용시간 고위험군 내 추가 위험
  신호 가능성
- 게임 비중 상위군의 저위험군 편향 — SNS·전반적 화면 사용 패턴과의
  이질성
- gender, stress_level, academic_work_impact 단독 구분력 약함

### 3.3 비즈니스 활용 방안

화면 사용시간·SNS 사용시간 중심 모니터링 지표 우선순위화 및 조기 위험군
식별 가능성. 게임 사용 비중 기반 위험군 분류 방식의 이 데이터 결과와의
불일치 — 활동 유형별 차등 기준 적용 필요성. 실제 서비스 적용이나 의료적
판단이 아닌 데이터 분석 관점의 활용 가능성으로 한정.

### 3.4 한계점

- 합성 데이터 특성상 실제 사용자 행동과 다른 인위적 패턴 혼입 가능성.
  Kaggle 측 synthetic 데이터 아티팩트 가능성 공식 언급.
- gaming_ratio 음의 상관, notifications_per_day 조건부 신호 등 직관과
  반대되는 결과 다수 — 합성 데이터 생성 과정의 영향 가능성, 해석의 신중함
  필요.
- Target Encoding된 범주형 변수의 낮은 예측력 — 스트레스 수준·성별에
  따른 차별화된 결론 도출의 한계.

### 3.5 결론

daily_screen_time_hours, social_media_hours, weekend_screen_time 등
절대적 사용 시간의 addicted_label 최상위 설명력. LightGBM 기준 Stratified
5-Fold 평균 ROC-AUC 0.963. 결측치의 낮은 정보력, 초기 가설(빈도 중요성,
주말 사용 증가폭, 게임 비중 방향) 다수의 기각 또는 역방향 확인. 가설의
반증 지점과 그 원인(다중공선성, 조건부 신호) 추적 과정 자체가 단일 성능
지표 이상의 프로젝트 핵심 성과.