# 검출 알고리즘 개선 아이디어

**현재 성능**: 76.6% (Baseline Z-score, 3 features)
**목표**: 80-85% 달성

---

## 현재 시스템 분석

### 강점
- ✅ 단순하고 해석 가능 (Z-score threshold)
- ✅ 학습 불필요 (baseline만 계산)
- ✅ 3개 features로 효율적

### 약점
- ❌ 선형 분리만 가능 (Z > threshold)
- ❌ Feature 간 상호작용 무시
- ❌ 개별 병리 특성 고려 안 함
- ❌ Sensitivity 낮음 (65.9%) - 34%를 놓침

### 오분류 패턴 분석
```
False Negatives (31명): 병적 보행을 정상으로 오판
  → 보상된 병적 보행 (compensated pathology)
  → Mild severity cases

False Positives (15명): 정상을 병적으로 오판
  → 매우 빠르거나 느린 정상 보행
  → 운동 선수, 노인
```

---

## 카테고리 1: 알고리즘 개선 (현재 features 유지)

### 1.1 Adaptive Thresholding (적응형 임계값)

**아이디어**: 나이, 성별, BMI에 따라 다른 baseline 사용

**현재**:
```python
# Single baseline for all
if Z > 1.5: pathological
```

**개선**:
```python
# Age-stratified baselines
if age < 30:
    baseline = baseline_young
    threshold = 1.5
elif age < 60:
    baseline = baseline_middle
    threshold = 1.7  # More lenient
else:
    baseline = baseline_elderly
    threshold = 2.0  # Much more lenient
```

**예상 개선**: +2-3% (특히 elderly에서 false positive 감소)

**장점**: 여전히 해석 가능, 단순
**단점**: Age/gender metadata 필요

---

### 1.2 Feature-Weighted Z-score (가중 Z-score)

**아이디어**: Cohen's d에 비례하여 features에 가중치 부여

**현재**:
```python
# Equal weighting
Z = (z_cadence + z_variability + z_irregularity) / 3
```

**개선**:
```python
# Weight by discriminative power (Cohen's d)
w_cadence = 0.85       # d = 0.85 (LARGE)
w_variability = 0.35   # d = 0.35 (SMALL)
w_irregularity = 0.51  # d = 0.51 (MEDIUM)

# Normalize weights
total = w_cadence + w_variability + w_irregularity  # = 1.71
w_cadence_norm = 0.85/1.71 = 0.50
w_variability_norm = 0.35/1.71 = 0.20
w_irregularity_norm = 0.51/1.71 = 0.30

# Weighted Z-score
Z_weighted = 0.50*z_cadence + 0.20*z_variability + 0.30*z_irregularity
```

**예상 개선**: +3-5% (cadence signal 더 강조)

**장점**: 여전히 해석 가능, 쉽게 구현
**단점**: Overfitting 위험 (작은 데이터셋)

---

### 1.3 Mahalanobis Distance (다변량 거리)

**아이디어**: Feature 간 covariance 고려한 거리 계산

**현재**: 각 feature 독립적으로 Z-score 계산
**개선**: Feature 간 상관관계 고려

```python
import numpy as np
from scipy.spatial.distance import mahalanobis

# Build baseline covariance matrix
normal_features = np.array([[cadence, var, irreg] for pattern in normal_patterns])
mean = np.mean(normal_features, axis=0)
cov = np.cov(normal_features.T)

# Compute Mahalanobis distance for test pattern
test_features = [test_cadence, test_var, test_irreg]
dist = mahalanobis(test_features, mean, np.linalg.inv(cov))

# Classify
if dist > threshold:
    prediction = 'pathological'
```

**예상 개선**: +2-4% (feature correlation 고려)

**장점**: 통계적으로 더 rigorous, 여전히 해석 가능
**단점**: Covariance matrix 추정 (작은 n에서 불안정)

---

### 1.4 Confidence-Based Classification (신뢰도 기반)

**아이디어**: Borderline cases를 "uncertain"으로 분류

**현재**: Binary classification (normal or pathological)

**개선**: 3-class classification
```python
if Z < 1.2:
    prediction = 'normal' (high confidence)
elif 1.2 <= Z < 1.8:
    prediction = 'uncertain' (refer for assessment)
else:
    prediction = 'pathological' (high confidence)
```

**예상 개선**: Accuracy는 비슷하지만 clinical utility 증가
- High confidence normal: 95% specificity
- High confidence pathological: 85% sensitivity
- Uncertain: Human review

**장점**: 임상적으로 매우 유용
**단점**: 3-class evaluation metrics 필요

---

## 카테고리 2: Feature Engineering (새로운 features)

### 2.1 Full-Body Kinematics Features

**아이디어**: Heel 외에 다른 body parts 추가

**새로운 features (Cohen's d 예상)**:

**1. Stride Length (보폭)**:
```python
# Hip-to-ankle distance at heel strike
stride_length = distance(hip_position, heel_position_at_peak)
```
- Normal: 0.8-1.2m
- Pathological: 0.4-0.8m (shorter)
- **Expected Cohen's d: 0.9-1.1** (LARGE)

**2. Trunk Sway (몸통 흔들림)**:
```python
# Shoulder lateral movement
shoulder_left = landmark[11]
shoulder_right = landmark[12]
shoulder_center_x = (shoulder_left.x + shoulder_right.x) / 2

trunk_sway = np.std(shoulder_center_x_trajectory)
```
- Normal: Low sway (< 0.05)
- Pathological: High sway (> 0.10)
- **Expected Cohen's d: 0.7-0.9** (MEDIUM-LARGE)

**3. Arm Swing Asymmetry (팔 흔들림 비대칭)**:
```python
# Wrist vertical movement amplitude
left_wrist_amplitude = max(wrist_left_y) - min(wrist_left_y)
right_wrist_amplitude = max(wrist_right_y) - min(wrist_right_y)

arm_asymmetry = abs(left_wrist_amplitude - right_wrist_amplitude)
```
- Normal: Low asymmetry (< 0.1)
- Pathological: High asymmetry (> 0.2, e.g., Parkinson's, stroke)
- **Expected Cohen's d: 0.6-0.8** (MEDIUM)

**4. Step Width Variability (보폭 폭 변동성)**:
```python
# Lateral distance between heel strikes
step_widths = [distance_x(left_heel_peak, right_heel_peak) for peaks]
step_width_var = np.std(step_widths) / np.mean(step_widths)
```
- Normal: Low variability (< 0.15)
- Pathological: High variability (> 0.25, fall risk)
- **Expected Cohen's d: 0.7-0.9** (MEDIUM-LARGE)

**예상 개선**: +4-8% (3 features → 6-7 features with d>0.7)

**주의사항**:
- Feature selection 필수 (Cohen's d > 0.7만 사용)
- Correlation check (|r| < 0.7)
- Stride length와 cadence 상관관계 확인 필요

---

### 2.2 Temporal Pattern Features (시계열 패턴)

**아이디어**: 시간에 따른 pattern shape 분석

**1. Gait Cycle Symmetry (보행 주기 대칭성)**:
```python
# Correlation between left and right heel trajectories
from scipy.stats import pearsonr

# Time-align trajectories
r, p = pearsonr(heel_left_aligned, heel_right_aligned)

symmetry = r  # Range: -1 to 1
```
- Normal: High symmetry (r > 0.85)
- Pathological: Low symmetry (r < 0.70, e.g., hemiplegia)
- **Expected Cohen's d: 0.8-1.0** (LARGE for asymmetric pathologies)

**2. Peak Sharpness (피크 날카로움)**:
```python
# Kurtosis of heel height trajectory
from scipy.stats import kurtosis

peak_sharpness = kurtosis(heel_height)
```
- Normal: Sharp peaks (kurtosis > 0)
- Pathological: Flat peaks (kurtosis < 0, shuffling gait)
- **Expected Cohen's d: 0.5-0.7** (MEDIUM)

**3. Harmonic Ratio (조화비)**:
```python
# FFT of heel height trajectory
from scipy.fft import fft

fft_vals = fft(heel_height)
power = np.abs(fft_vals)**2

# Ratio of first harmonic to higher harmonics
harmonic_ratio = power[1] / np.sum(power[2:6])
```
- Normal: High ratio (smooth gait)
- Pathological: Low ratio (irregular gait)
- **Expected Cohen's d: 0.6-0.8** (MEDIUM)

**예상 개선**: +2-4% (temporal patterns 추가)

---

### 2.3 Multi-View Fusion (다중 시점)

**아이디어**: 정면(frontal) + 측면(sagittal) 동시 분석

**Frontal view에서만 보이는 features**:
```python
# 1. Lateral trunk lean
trunk_lean = angle(shoulder_center, hip_center, vertical)

# 2. Step width
step_width = distance_x(left_heel, right_heel)

# 3. Knee valgus/varus
knee_angle_frontal = angle(hip, knee, ankle) - 180
```

**Fusion 방법**:
```python
# Late fusion (combine scores)
Z_sagittal = compute_z_score(sagittal_features)
Z_frontal = compute_z_score(frontal_features)

Z_combined = (Z_sagittal + Z_frontal) / 2

if Z_combined > threshold:
    prediction = 'pathological'
```

**예상 개선**: +3-6% (complementary information)

**단점**: 2개 영상 필요 (deployment 복잡도 증가)

---

## 카테고리 3: Machine Learning Approaches

### 3.1 Logistic Regression (해석 가능한 ML)

**아이디어**: Feature 간 비선형 조합 학습

```python
from sklearn.linear_model import LogisticRegression

# Features
X = [[cadence, variability, irregularity] for pattern in patterns]
y = [0 if normal else 1 for pattern in patterns]

# Train
model = LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y)

# Coefficients (interpretable!)
print(f"Cadence weight: {model.coef_[0][0]}")
print(f"Variability weight: {model.coef_[0][1]}")
print(f"Irregularity weight: {model.coef_[0][2]}")

# Predict
prob = model.predict_proba(test_features)[0][1]
if prob > 0.5:
    prediction = 'pathological'
```

**예상 개선**: +3-5% (non-linear decision boundary)

**장점**:
- 여전히 해석 가능 (coefficients)
- Feature importance 자동 학습
- Probability output (confidence)

**단점**:
- Training 필요
- Overfitting 위험 (작은 n=187)
- Cross-validation 필수

---

### 3.2 Random Forest (Feature Importance)

**아이디어**: Decision tree ensemble로 feature interaction 학습

```python
from sklearn.ensemble import RandomForestClassifier

# Train
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Prevent overfitting
    min_samples_leaf=10,
    random_state=42
)
model.fit(X, y)

# Feature importance
importances = model.feature_importances_
print(f"Cadence: {importances[0]:.3f}")
print(f"Variability: {importances[1]:.3f}")
print(f"Irregularity: {importances[2]:.3f}")

# Predict
prob = model.predict_proba(test_features)[0][1]
```

**예상 개선**: +5-8% (non-linear, feature interactions)

**장점**:
- Feature interactions 자동 학습
- Feature importance 출력
- Robust to outliers

**단점**:
- "Black box" (해석 어려움)
- Overfitting 위험 (small n)
- Deployment 복잡 (100 trees)

---

### 3.3 Support Vector Machine (Non-linear Boundary)

**아이디어**: Kernel trick으로 non-linear decision boundary

```python
from sklearn.svm import SVC

# Train with RBF kernel
model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True
)
model.fit(X, y)

# Predict
prob = model.predict_proba(test_features)[0][1]
```

**예상 개선**: +4-7% (complex boundaries)

**장점**: Non-linear separation
**단점**:
- Hyperparameter tuning 필요
- Not interpretable
- Small n 위험

---

### 3.4 Gradient Boosting (XGBoost/LightGBM)

**아이디어**: 최고 성능의 classical ML

```python
from xgboost import XGBClassifier

# Train
model = XGBClassifier(
    n_estimators=50,
    max_depth=3,  # Shallow to prevent overfitting
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X, y)

# Feature importance
importances = model.feature_importances_

# Predict
prob = model.predict_proba(test_features)[0][1]
```

**예상 개선**: +6-10% (best classical ML)

**장점**:
- State-of-the-art performance
- Feature importance
- Handles complex interactions

**단점**:
- Overfitting 위험 (작은 n)
- Hyperparameter tuning 필수
- Less interpretable
- Deployment 복잡

---

### 3.5 Deep Learning (LSTM for Time Series)

**아이디어**: 시계열 전체를 input으로 end-to-end 학습

```python
import tensorflow as tf
from tensorflow.keras import layers

# Model
model = tf.keras.Sequential([
    layers.LSTM(64, input_shape=(n_frames, 2)),  # 2 = left/right heel
    layers.Dropout(0.5),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Input: raw heel height trajectories (no manual features!)
X = [pattern['heel_height_left'] + pattern['heel_height_right'] for pattern in patterns]
y = [0 if normal else 1 for pattern in patterns]

# Train
model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2)

# Predict
prob = model.predict(test_trajectory)[0][0]
```

**예상 개선**: +8-15% (learns optimal features)

**장점**:
- End-to-end learning
- Learns optimal features automatically
- Captures temporal dependencies

**단점**:
- **작은 데이터셋 (n=187)에서 overfitting 심각**
- Black box (완전히 해석 불가)
- Deployment 복잡 (model weights)
- Training 느림

**권장**: 데이터 500+ 확보 후 시도

---

## 카테고리 4: Ensemble Methods (앙상블)

### 4.1 Stacking (여러 모델 조합)

**아이디어**: 여러 알고리즘의 예측을 meta-learner로 조합

```python
from sklearn.ensemble import StackingClassifier

# Base models
estimators = [
    ('z_score', ZScoreClassifier()),  # Custom
    ('logistic', LogisticRegression()),
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=5)),
]

# Meta-learner
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_model.fit(X, y)
```

**예상 개선**: +5-8% (각 모델의 강점 결합)

**장점**: Best of all models
**단점**: 복잡, 해석 어려움

---

### 4.2 Voting Classifier (다수결)

**아이디어**: 여러 모델의 vote로 결정

```python
from sklearn.ensemble import VotingClassifier

# Models
model1 = ZScoreClassifier()  # 76.6%
model2 = LogisticRegression()  # ~80%
model3 = RandomForestClassifier()  # ~82%

# Voting
voting_model = VotingClassifier(
    estimators=[('z', model1), ('lr', model2), ('rf', model3)],
    voting='soft',  # Probability averaging
    weights=[1, 2, 2]  # Trust ML more than baseline
)

voting_model.fit(X, y)
```

**예상 개선**: +4-7% (robust to individual model errors)

---

## 카테고리 5: Pathology-Specific Detection

### 5.1 Hierarchical Classification (계층적 분류)

**아이디어**: Stage 1 (normal vs pathological) → Stage 2 (specific pathology)

```python
# Stage 1: Binary classification
if Z < 1.5:
    return 'normal'

# Stage 2: Multi-class classification (pathological type)
pathology_features = extract_pathology_specific_features(pattern)

if asymmetry > 0.3 and cadence < 150:
    return 'hemiplegia (stroke)'
elif trunk_sway > 0.15 and step_width_var > 0.3:
    return 'ataxia (cerebellar)'
elif cadence < 100 and stride_length < 0.5:
    return 'parkinsonian gait'
else:
    return 'unspecified pathological'
```

**예상 개선**: Stage 1은 비슷, but clinical utility 증가 (specific diagnosis)

---

### 5.2 One-Class SVM (Anomaly Detection)

**아이디어**: Normal만 학습, pathological을 outlier로 검출

```python
from sklearn.svm import OneClassSVM

# Train on normal only!
normal_features = [extract_features(p) for p in normal_patterns]

model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
model.fit(normal_features)

# Predict
score = model.decision_function(test_features)
if score < 0:
    prediction = 'pathological' (outlier)
else:
    prediction = 'normal'
```

**예상 개선**: +2-4% (특히 rare pathologies)

**장점**: Normal만 학습 (unbalanced data에 강함)
**단점**: Hyperparameter tuning 어려움

---

## 추천 단계별 로드맵

### Phase 1: Quick Wins (1-2주, +5-8% 예상)

**우선순위 1**: Feature-Weighted Z-score (가중 Z-score)
- 구현: 1일
- Expected: +3-5%
- Reason: 단순, 해석 가능, Cohen's d 활용

**우선순위 2**: Confidence-Based Classification (3-class)
- 구현: 2일
- Expected: Clinical utility 증가
- Reason: Borderline cases 처리

**우선순위 3**: Age-Stratified Baseline (나이별 baseline)
- 구현: 3일 (나이 metadata 수집 필요)
- Expected: +2-3%
- Reason: False positive 감소 (elderly)

**예상 결과**: 76.6% → 81-84%

---

### Phase 2: Feature Engineering (2-4주, +4-8% 추가)

**우선순위 1**: Stride Length (보폭)
- 구현: 5일
- Expected Cohen's d: 0.9-1.1 (LARGE)
- Expected: +3-5%

**우선순위 2**: Trunk Sway (몸통 흔들림)
- 구현: 3일
- Expected Cohen's d: 0.7-0.9
- Expected: +2-3%

**우선순위 3**: Gait Cycle Symmetry (대칭성)
- 구현: 2일
- Expected Cohen's d: 0.8-1.0
- Expected: +2-3%

**Feature selection**:
- 각 feature의 Cohen's d 계산
- d > 0.7만 추가
- Correlation < 0.7 확인

**예상 결과**: 81-84% → 85-89%

---

### Phase 3: Machine Learning (4-6주, +3-5% 추가)

**우선순위 1**: Logistic Regression
- 구현: 1주 (cross-validation 포함)
- Expected: +3-5%
- Reason: 해석 가능, 적은 overfitting

**우선순위 2**: Random Forest (if data > 300)
- 구현: 1주
- Expected: +4-6%
- Reason: Feature interactions

**Cross-validation**:
- Stratified 5-fold CV
- Test on unseen 20%
- Report mean ± std

**예상 결과**: 85-89% → 88-92%

---

### Phase 4: Advanced (장기, data 확보 후)

**데이터 확보 목표**: 500+ patterns
- 현재: 187 patterns
- 목표: 500+ (clinical trial 통해 수집)

**이후 시도**:
- XGBoost/LightGBM
- LSTM/Transformer
- Multi-view fusion
- Ensemble methods

**예상 결과**: 88-92% → 90-95%

---

## 즉시 시작 가능한 Top 3 추천

### 🥇 #1: Feature-Weighted Z-score

**Why**:
- 구현 매우 간단 (30분)
- 해석 가능 유지
- Cohen's d 직접 활용
- Overfitting 위험 낮음

**Code**:
```python
def weighted_z_score(features, baseline):
    # Weights from Cohen's d
    w_cadence = 0.50
    w_variability = 0.20
    w_irregularity = 0.30

    z_cadence = abs(features.cadence - baseline['cadence_mean']) / baseline['cadence_std']
    z_var = abs(features.variability - baseline['variability_mean']) / baseline['variability_std']
    z_irreg = abs(features.irregularity - baseline['irregularity_mean']) / baseline['irregularity_std']

    Z = w_cadence*z_cadence + w_variability*z_var + w_irregularity*z_irreg
    return Z
```

**Expected**: 76.6% → 79-81%

---

### 🥈 #2: Stride Length Feature 추가

**Why**:
- Cohen's d 예상: 0.9-1.1 (LARGE!)
- MediaPipe에서 계산 가능 (hip, ankle landmarks)
- 임상적으로 의미 있음
- 단일 feature로 큰 개선 가능

**Code**:
```python
def compute_stride_length(pattern):
    hip_y = pattern['hip_center_y']
    ankle_y = pattern['ankle_y']

    # Distance at heel strike (peak)
    peaks, _ = find_peaks(pattern['heel_height_left'])

    stride_lengths = []
    for peak in peaks:
        hip_pos = [pattern['hip_x'][peak], hip_y[peak]]
        ankle_pos = [pattern['ankle_x'][peak], ankle_y[peak]]
        stride_lengths.append(np.linalg.norm(np.array(hip_pos) - np.array(ankle_pos)))

    return np.mean(stride_lengths)
```

**Expected**: 79-81% → 82-85%

---

### 🥉 #3: Logistic Regression (4 features)

**Why**:
- Feature 간 interaction 학습
- 해석 가능 (coefficients)
- Probability output
- 작은 데이터셋에서도 안정적

**Code**:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Features: cadence, variability, irregularity, stride_length
X = [[p.cadence, p.variability, p.irregularity, p.stride_length] for p in patterns]
y = [0 if p.label=='normal' else 1 for p in patterns]

# Train with L2 regularization
model = LogisticRegression(penalty='l2', C=1.0, max_iter=1000)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Train on all data
model.fit(X, y)

# Coefficients (interpretable!)
print("Feature importances:")
for i, name in enumerate(['cadence', 'variability', 'irregularity', 'stride_length']):
    print(f"{name}: {model.coef_[0][i]:.3f}")
```

**Expected**: 82-85% → 85-88%

---

## 구현 순서 요약

```
Week 1: Feature-Weighted Z-score
         ↓
       79-81%

Week 2-3: Stride Length 추가
         ↓
       82-85%

Week 4-5: Logistic Regression
         ↓
       85-88%

Week 6+: Trunk Sway, Symmetry 추가
         ↓
       88-92% (if data sufficient)
```

---

## 최종 권장사항

**즉시 시작** (이번 주):
1. ✅ Feature-Weighted Z-score (30분 구현, +3-5%)
2. ✅ Stride Length 추가 (2일 구현, +3-5%)

**다음 단계** (2-4주):
3. ✅ Logistic Regression (1주 구현, +3-5%)
4. ✅ Trunk Sway or Symmetry (1-2주 구현, +2-4%)

**목표 달성 예상**:
- 현재: 76.6%
- 단기 (4주): 85-88%
- 중기 (8주): 88-92% (데이터 충분 시)

**핵심 원칙 유지**:
- ✅ Cohen's d > 0.7 features만 추가
- ✅ Correlation < 0.7 확인
- ✅ 해석 가능성 유지 (clinicians trust)
- ✅ Cross-validation 필수 (overfitting 방지)

---

**파일**: ALGORITHM_IMPROVEMENT_IDEAS.md
**작성일**: 2025-10-30
**목표**: 76.6% → 85-92% 달성
**추천 시작**: Feature-Weighted Z-score + Stride Length
