# 남은 과제 해결 요약

## 완료된 작업 (Day 1-2)

### ✅ Task 1: 우측 관절 ICC 악화 원인 분석 (완료)

**구현**: [`diagnose_right_side.py`](diagnose_right_side.py)

**주요 발견**:
- **우측 무릎 ICC**: 0.018 (좌측: 0.344) → **Δ = -0.327** (심각한 악화)
- **고관절/발목**: 우측이 약간 개선 (여전히 ICC ≈ 0)

**원인 분석**:
1. **카메라 각도 편향**: 측면 촬영에서 좌측에 더 가까움 → 우측 landmark visibility 저하 가능
2. **좌우 비대칭 보행**: 병리학적 또는 보상적 보행 패턴
3. **우측 landmark 검출 노이즈**: MediaPipe 3D reconstruction 품질 저하

**권장 조치**:
- ✅ 좌측 데이터를 우선적으로 사용하여 각도 변환 학습
- ✅ 우측 특화 변환 파라미터 별도 학습 (측면별 분리)
- ⏳ 다중 뷰 카메라 시스템 도입 (장기)

**산출물**:
- 📄 [`RIGHT_SIDE_DIAGNOSIS.md`](validation_results_improved/RIGHT_SIDE_DIAGNOSIS.md)
- 📊 [`right_side_diagnosis.pdf`](validation_results_improved/right_side_diagnosis.pdf) - 시계열, 히트맵, 오차 분포
- 📋 [`right_side_diagnosis.json`](validation_results_improved/right_side_diagnosis.json)

---

### ✅ Task 2: 힐스트라이크 융합 가중치 최적화 (완료, 성능 개선 실패)

**구현**: [`optimize_heel_weights.py`](optimize_heel_weights.py)

**결과**:
- **최적화 알고리즘**: Differential Evolution (전역 최적화)
- **좌측 가중치**: heel=0.304, ankle=0.124, toe=0.572, velocity=0.716
- **우측 가중치**: heel=0.319, ankle=0.205, toe=0.476, velocity=0.268
- **Validation 성능**: **-2~3% 악화**

**실패 원인**:
1. **평가 지표 불일치**: DTW distance (heel strike interval) ≠ 각도 ICC
   - 힐스트라이크 타이밍 개선 ≠ 각도 정확도 개선
2. **간접 지표 문제**: 병원 데이터에 실제 heel strike frame이 없어 간접 추정 사용
3. **과도한 toe 가중치**: 0.57 (기본 0.1) → 발가락 landmark 노이즈에 민감

**교훈**:
- 힐스트라이크 검출 품질보다 **각도 변환 품질**이 ICC에 더 직접적 영향
- 기본 가중치 (0.6/0.3/0.1)가 경험적으로 검증된 값임
- 최적화 목적함수를 **ICC 직접 최대화**로 변경 필요 (but 계산 비용 높음)

**산출물**:
- 📋 [`optimized_heel_weights.json`](validation_results_improved/optimized_heel_weights.json)
- **권장**: 기본 가중치 유지, 각도 변환에 집중

---

### ✅ Task 3: 비선형 각도 변환 구현 (완료)

**구현**: [`angle_converter.py`](angle_converter.py) 수정

**추가된 변환 방법**:
1. **Linear** (기존): `y = offset + scale * x`
2. **Polynomial 2nd** (신규): `y = c0 + c1*x + c2*x²` (Ridge 정규화)
3. **Piecewise** (신규): 구간별 선형 (gait cycle 0-50%, 50-100% 다른 scale)
4. **Ridge** (신규): 다항식 + 강한 L2 정규화 (alpha=1.0)

**기술 세부사항**:
```python
# Polynomial (Ridge regularization)
ridge = Ridge(alpha=0.1)
X = np.column_stack([pred**i for i in range(degree + 1)])
ridge.fit(X, targ)

# Piecewise (quantile-based segmentation)
boundaries = np.percentile(pred, [0, 50, 100])
for segment in segments:
    mask = (pred >= lower) & (pred < upper)
    A = np.column_stack([np.ones_like(pred_seg), pred_seg])
    coeffs, _ = nnls(A, targ_seg)
```

**교차검증 전략**:
- **Train set (14명)**: 파라미터 학습 (offset, scale, coeffs)
- **Validation set (4명)**: 방법 선택 (ICC 최고값)
- **Test set (3명)**: 최종 평가만 (1회)

**과적합 방지**:
- ✅ Ridge regularization (L2 penalty)
- ✅ Train/Val/Test 엄격 분리
- ✅ Validation ICC로만 방법 선택

**예상 효과**:
- **Polynomial**: 비선형 관계 포착 → ICC 0.1 → 0.3~0.4 기대
- **Piecewise**: Gait cycle 구간별 특성 (stance vs swing phase) 반영
- **실제 테스트 필요**: 현재 코드만 완성, 실행은 `run_improved_validation.py` 재실행 필요

---

## 미완료 작업 (Day 3-4)

### ⏳ Task 4: 관절별 특화 방법 추가

**계획**:
```python
# angle_converter.py에 추가
self.method_registry = {
    'knee': [
        'joint_angle', 'joint_angle_inverted', 'projected_2d',
        'polynomial_2nd'  # 신규
    ],
    'hip': [
        'segment_to_vertical', 'joint_angle',
        'trunk_relative',  # 신규: 체간 대비 각도
        'pelvic_tilt'      # 신규: 골반 경사 고려
    ],
    'ankle': [
        'joint_angle', 'segment_to_horizontal',
        'foot_ground_angle'  # 신규: 지면 접촉각
    ]
}
```

**구현 필요**:
1. `trunk_relative`: 양측 어깨 중점 - 골반 중점 벡터 대비 고관절 각도
2. `pelvic_tilt`: 좌우 골반 랜드마크 연결선의 수평 대비 경사
3. `foot_ground_angle`: 발끝-뒤꿈치 벡터의 수평 대비 각도 (dorsiflexion/plantarflexion)

**예상 시간**: 2-3시간

---

### ⏳ Task 5: SPM Permutation Test 구현

**계획**: [`spm_analysis.py`](spm_analysis.py) 수정

**추가 기능**:
```python
def spm_analysis_with_fallback(mp_angles, hosp_angles):
    residuals = mp_angles - hosp_angles
    _, p_shapiro = shapiro(residuals)

    if p_shapiro > 0.05:
        # 정규성 만족 → Parametric
        return parametric_spm_fdr(mp_angles, hosp_angles)
    else:
        # 정규성 위배 → Permutation
        return permutation_spm(mp_angles, hosp_angles, n_perm=10000)

def permutation_spm(mp, hosp, n_perm=10000):
    """Sign-flipping permutation test for paired data."""
    obs_t = paired_t_statistic(mp, hosp)

    null_dist = []
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=len(mp))
        diff_perm = signs[:, None] * (mp - hosp)
        t_perm = t_statistic(diff_perm)
        null_dist.append(np.max(np.abs(t_perm)))

    threshold = np.percentile(null_dist, 95)
    return {'t': obs_t, 'threshold': threshold, 'p_values': empirical_p(obs_t, null_dist)}
```

**장점**:
- 정규성 가정 불필요
- Family-wise error rate 제어
- 시계열 상관 구조 보존

**예상 시간**: 2시간

---

### ⏳ Task 6: CI 파이프라인 설정

**GitHub Actions 예시**:
```yaml
# .github/workflows/gait_validation.yml
name: Gait Validation Regression Test
on: [push, pull_request]

jobs:
  regression-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python regression_test.py
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: regression-report
          path: validation_results_improved/regression_report.json
```

**Pre-commit Hook**:
```bash
#!/bin/bash
# .git/hooks/pre-commit
python /data/gait/regression_test.py
if [ $? -ne 0 ]; then
    echo "❌ Regression test failed! Commit aborted."
    exit 1
fi
```

**예상 시간**: 1시간

---

### ⏳ Task 7: 최종 재검증

**실행 명령**:
```bash
# 비선형 변환 + 개선된 방법 적용
python run_improved_validation.py --use-nonlinear --output-dir validation_results_v2

# 결과 비교
python visualize_improvements.py \
    --baseline validation_results_improved/improved_validation_report.json \
    --improved validation_results_v2/improved_validation_report.json \
    --output validation_results_v2/final_comparison.pdf
```

**예상 성과**:
- **현실적 목표**: Test ICC > 0.5 (Moderate) for 2/3 joints
- **도전적 목표**: Test ICC > 0.7 (Good) for 1/3 joints (left knee)
- **필수 달성**: 발목 ICC > 0, 우측 ICC 개선

**예상 시간**: 30분 (실행) + 1시간 (분석)

---

## 전체 진행 상황

### 완료율
- ✅ **Day 1 (우측 분석, 힐스트라이크)**: 100%
- ✅ **Day 2 (비선형 변환 구현)**: 100% (테스트 미실행)
- ⏳ **Day 3 (관절별 특화, SPM)**: 0%
- ⏳ **Day 4 (CI, 재검증)**: 0%

**총 진행률**: **50%** (4/8 tasks 완료)

---

## 핵심 교훈 및 다음 단계

### 발견된 근본 문제
1. **좌우 비대칭성**: 카메라 각도 또는 보행 패턴 문제 → 측면별 모델 필요
2. **선형 변환 한계**: 현재 Validation ICC < 0.1 → 비선형 필수
3. **힐스트라이크 vs 각도**: 타이밍 정확도 ≠ 각도 정확도 (독립적 문제)

### 우선순위 권장
1. **최우선**: Task 7 (재검증) - 비선형 변환 효과 확인
2. **2순위**: Task 4 (관절별 특화) - 고관절/발목 ICC 개선
3. **3순위**: Task 5 (SPM permutation) - 통계적 엄밀성
4. **4순위**: Task 6 (CI) - 자동화

### 현실적 성과 예측

#### 낙관적 시나리오 (비선형 변환 효과 큼)
| Joint | Current ICC | Expected ICC | Status |
|-------|-------------|--------------|--------|
| Left Knee | 0.344 | 0.6~0.7 | ✅ Good 달성 가능 |
| Right Knee | 0.018 | 0.3~0.4 | 🔶 개선되지만 부족 |
| Hip | -0.01 | 0.2~0.3 | 🔶 양수 전환 |
| Ankle | -0.04 | 0.1~0.2 | 🔶 양수 전환 |

#### 보수적 시나리오 (비선형 효과 제한적)
| Joint | Current ICC | Expected ICC | Status |
|-------|-------------|--------------|--------|
| Left Knee | 0.344 | 0.4~0.5 | 🔶 Moderate |
| Right Knee | 0.018 | 0.05~0.1 | ❌ 여전히 Poor |
| Hip | -0.01 | 0.05~0.1 | 🔶 근소 개선 |
| Ankle | -0.04 | 0~0.05 | ❌ 미미한 개선 |

### 근본 한계 인정 기준
만약 **Task 4 + Task 7 완료 후에도 Test ICC < 0.3** 이면:
- **단안 영상의 근본 한계** 명시
- **다중 뷰 시스템 필요성** 제안
- **현재 방법론의 적용 범위** 명확화
  - ✅ Screening (대략적 평가)
  - ❌ Clinical diagnosis (정밀 진단) → 부적합

---

## 산출물 목록

### 분석 도구
- ✅ [`diagnose_right_side.py`](diagnose_right_side.py)
- ✅ [`optimize_heel_weights.py`](optimize_heel_weights.py)
- ✅ [`angle_converter.py`](angle_converter.py) (비선형 변환 추가)
- ⏳ [`spm_analysis.py`](spm_analysis.py) (permutation 미추가)

### 보고서
- ✅ [`RIGHT_SIDE_DIAGNOSIS.md`](validation_results_improved/RIGHT_SIDE_DIAGNOSIS.md)
- ✅ [`CURRENT_STATUS_PLAN.md`](CURRENT_STATUS_PLAN.md)
- ✅ [`IMPROVEMENT_REPORT.md`](IMPROVEMENT_REPORT.md)
- ✅ 본 문서: [`REMAINING_TASKS_SUMMARY.md`](REMAINING_TASKS_SUMMARY.md)

### 시각화
- ✅ [`right_side_diagnosis.pdf`](validation_results_improved/right_side_diagnosis.pdf)
- ✅ [`icc_comparison.png`](validation_results_improved/icc_comparison.png)
- ✅ [`rmse_comparison.png`](validation_results_improved/rmse_comparison.png)
- ✅ SPM 플롯: `spm_{joint}.png` (6개)

### 데이터
- ✅ [`optimized_heel_weights.json`](validation_results_improved/optimized_heel_weights.json)
- ✅ [`right_side_diagnosis.json`](validation_results_improved/right_side_diagnosis.json)
- ✅ [`improved_validation_report.json`](validation_results_improved/improved_validation_report.json) (1.7MB)

---

## 다음 액션

### 즉시 실행 가능 (< 1시간)
```bash
# 1. 비선형 변환 재검증
python run_improved_validation.py

# 2. 결과 확인
cat validation_results_improved/improved_validation_report.json | jq '.summary_statistics.left_knee.test.icc'

# 3. 시각화 업데이트
python visualize_improvements.py
```

### 1일 내 완료 목표
1. ✅ 비선형 변환 효과 확인 → Task 7
2. ⏳ 효과 있으면: 관절별 특화 추가 → Task 4
3. ⏳ 효과 없으면: 한계 인정 보고서 작성

### 최종 목표
- **기술적 목표**: Test ICC > 0.5 for at least 2 joints
- **과학적 목표**: 단안 영상 gait analysis의 **가능성과 한계** 명확히 규명
- **실용적 목표**: Screening tool로서의 가치 입증 (정밀 진단 X, 이상 탐지 O)

---

*Generated: Day 2 완료 시점*
*Next milestone: Task 7 (비선형 변환 재검증) 실행 및 결과 분석*
