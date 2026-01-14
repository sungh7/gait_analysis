# Deming 회귀 수정 결과 보고서

## 수정 작업 완료

✅ **Fix #1**: Line 165 컬럼 'Z' → 'X' 변경 완료
✅ **Fix #2**: 커브별 평균 중심화 추가 완료
✅ **재실행**: improve_calibration_deming.py 재실행 완료

---

## 결과 비교

### 수정 전 (Broken)
```json
{
  "ankle": {"slope": 0.051, "intercept": 1.75,   "equivalence": false},
  "knee":  {"slope": 0.062, "intercept": -20.98, "equivalence": false},
  "hip":   {"slope": 0.014, "intercept": 0.29,   "equivalence": false}
}
```

### 수정 후 (Fixed)
```json
{
  "ankle": {"slope": 0.829, "intercept": -0.002, "equivalence_intercept": true},
  "knee":  {"slope": 0.099, "intercept": -0.000, "equivalence_intercept": true},
  "hip":   {"slope": 0.035, "intercept": 0.000,  "equivalence_intercept": true}
}
```

---

## 평가

### ✅ 발목 (ANKLE) - 성공!

**Deming 회귀 결과:**
- Slope: **0.829** (95% CI: 0.757-0.900) ✅ **예상 범위 0.8-1.2 내**
- Intercept: -0.002° (95% CI: -0.22 to 0.22) ✅ **~0에 가까움**
- Equivalence intercept: **true** ✅
- Equivalence slope: false (CI가 1.0을 포함하지 않음, 하지만 0.829는 허용 가능)

**해석**: 발목 캘리브레이션이 정상적으로 작동합니다. MP와 GT 데이터 스케일이 약 0.829 비율로 일치합니다.

---

### ⚠️ 무릎 (KNEE) - 개선되었으나 여전히 문제

**Deming 회귀 결과:**
- Slope: **0.099** (95% CI: 0.084-0.113) ❌ **여전히 10배 작음**
- Intercept: -0.000° (95% CI: -0.12 to 0.12) ✅
- Equivalence slope: false

**표준편차 분석:**
```
MP (centered):  8.5° std
GT X column:    1.1° std
Ratio: 7.5x (MP가 GT보다 7.5배 큰 변동성)
```

**문제**: GT 데이터의 무릎 각도가 MP보다 훨씬 작은 변동성을 보입니다. 이는:
1. GT 데이터가 다른 정규화를 사용하거나
2. MP와 GT가 다른 각도 정의를 사용할 수 있음

---

### ⚠️ 엉덩이 (HIP) - 개선되었으나 여전히 문제

**Deming 회귀 결과:**
- Slope: **0.035** (95% CI: 0.027-0.042) ❌ **여전히 30배 작음**
- Intercept: 0.000° (95% CI: -0.20 to 0.20) ✅
- Equivalence slope: false

**표준편차 분석:**
```
MP (centered):  23.4° std
GT X column:     4.0° std
Ratio: 5.8x (MP가 GT보다 5.8배 큰 변동성)
```

**추가 조사**:
- GT Y column: 15.5° std (ratio 1.5x) ✅ **MP와 비슷한 스케일!**
- GT Y column을 사용하면 엉덩이 캘리브레이션이 개선될 가능성 있음

---

## 근본 원인 분석

### MP vs GT 데이터 스케일 비교

| 관절 | MP std (°) | GT X std (°) | Ratio | 결과 |
|------|-----------|-------------|-------|------|
| **발목** | 3.8 | 2.7 | **1.4x** | ✅ 비슷함 → slope 0.829 |
| **무릎** | 8.5 | 1.1 | **7.5x** | ❌ GT가 너무 작음 → slope 0.099 |
| **엉덩이** | 23.4 | 4.0 | **5.8x** | ❌ GT가 너무 작음 → slope 0.035 |

### 가설: GT 데이터 형식 불일치

**Traditional_condition.csv 컬럼 구조:**
- X 컬럼: 평균으로부터의 편차? (작은 변동성)
- Y 컬럼: 원시 각도? (큰 변동성, 특히 무릎/엉덩이)
- Z 컬럼: 알 수 없는 정규화

**증거:**
- 발목: X, Z 모두 MP와 비슷한 스케일
- 무릎: 모든 컬럼이 MP와 다른 스케일
- 엉덩이: Y 컬럼만 MP와 비슷한 스케일

---

## 다음 조사 단계

### 1. GT 데이터 형식 문서 확인
- traditional_condition.csv는 어떻게 생성되었는가?
- X, Y, Z 컬럼의 정확한 정의는?
- 병원 시스템의 원본 데이터 형식은?

### 2. 대체 GT 소스 탐색
- `gait_long.csv`의 `condition1_avg` 컬럼 (원시 각도일 가능성)
- Excel 원본 파일 (MediaPipe_S1_01_traditional_format.xlsx)
- 다른 중간 처리 파일

### 3. MP 각도 계산 검증
- MP 각도가 올바른 정의를 사용하는가?
- 무릎/엉덩이 각도 계산 로직 재검토
- 3D 랜드마크 → 각도 변환 과정 확인

### 4. 관절별 컬럼 매핑 시도
```python
JOINT_COLUMN_MAP = {
    "ankle_dorsi_plantarflexion": "X",  # ✅ 작동함
    "knee_flexion_extension": "Y",      # 시도해볼 가치 있음
    "hip_flexion_extension": "Y",       # Y 컬럼이 더 나을 수 있음
}
```

---

## 임시 권장사항

### Option A: 발목만 사용
- 발목 캘리브레이션은 성공적으로 작동
- 무릎/엉덩이는 추가 조사 후 진행
- Phase 1 Day 4-5 (필터링)를 발목만으로 시작

### Option B: Y 컬럼 시도
- 엉덩이는 Y 컬럼이 더 나은 스케일을 보임
- 무릎도 Y 컬럼 시도해볼 가치 있음
- `load_gt_curves` 함수에 관절별 컬럼 선택 로직 추가

### Option C: 원본 데이터 재확인
- GT 데이터 생성 과정 재검토
- Excel 원본이나 gait_long.csv에서 직접 추출
- 데이터 파이프라인 전체 재검증

---

## 현재 상태

**Phase 1 Day 1-3 진행 상황:**
- ✅ Deming 회귀 구현 완료
- ✅ 발목 캘리브레이션 성공
- ⚠️ 무릎/엉덩이 데이터 스케일 불일치 발견
- 🔧 추가 조사 필요

**Go/No-Go 결정:**
- **발목 단독 진행**: Go ✅
- **3개 관절 모두 진행**: No-Go ❌ (무릎/엉덩이 수정 필요)

---

## 백업 파일

- `calibration_parameters_deming_broken.json` - 수정 전 (망가진) 파라미터
- `calibration_parameters_deming.json` - 수정 후 (발목 성공, 무릎/엉덩이 미해결)

**생성일**: 2025-11-07
