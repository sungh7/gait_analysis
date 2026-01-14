# Phase 1 Day 1-5 완료 보고서

## 요약

**현재 상태**: ⚠️ **부분 성공 - 엉덩이 관절만**

4번의 수정을 거쳐 다음 결과 달성:
- ✅ **엉덩이 ICC = +0.030** (양수이지만 목표치 미달)
- ❌ **무릎 ICC = -0.385** (음수)
- ❌ **발목 ICC = -0.790** (음수)
- 전체 ICC = -0.383 (목표 0.40 미달)

---

## 최종 결과 (V4)

| 관절 | ICC | RMSE | RMSE (중심화) | 상관계수 | Bias |
|------|-----|------|--------------|---------|------|
| **발목** | -0.790 | 69.0° | 27.2° | 0.273 | 0.00° |
| **무릎** | -0.385 | 42.6° | 33.8° | 0.494 | 0.00° |
| **엉덩이** | **+0.030** | **15.3°** | **7.3°** | **0.792** | **0.00°** |

### 성과

1. ✅ **Bias 제거 완료**: 모든 관절 ~0° (평균 보정 성공)
2. ✅ **엉덩이 ICC 양수 달성**: 0.030 (낮지만 양수)
3. ✅ **엉덩이 강한 상관관계**: 0.792
4. ✅ **Deming 회귀 구현 완료**: slope, intercept 계산
5. ✅ **Butterworth 필터링 적용**: 4Hz(발목), 6Hz(무릎/엉덩이)

### 문제점

1. ❌ **발목/무릎 ICC 음수**: 절대 일치도 매우 낮음
2. ❌ **MP와 GT 스케일 불일치**: Deming slope 0.47-7.4배 (ideal 1.0)
3. ❌ **생체역학적 범위 불일치**: MP 각도가 GT와 다른 정의 사용

---

## 근본 원인

### MP 각도 계산 문제 (`core_modules/main_pipeline.py`)

#### 발목
```python
ankle_angle = angle - 90  # 90도 빼기
```
- MP 결과: mean=23.0°, std=5.0°, range=[11.8°, 29.6°]
- GT 범위: mean=1.5°, std=8.5°, range=[-20.6°, 13.8°] ✅ 생체역학 정상
- **문제**: MP가 ~22° 오프셋, GT보다 작은 변동성

#### 무릎
```python
knee_flexion = 180 - angle  # 180도에서 빼기
```
- MP 결과: mean=30.5°, std=10.1°, range=[17.8°, 48.6°]
- GT 범위: mean=19.0°, std=20.6°, range=[-0.2°, 61.7°] ✅ 완전 신전~굴곡
- **문제**: MP가 ROM 과소 추정 (std 10.1° vs GT 20.6°)

#### 엉덩이
```python
flexion = np.degrees(np.arctan2(local[2], local[1]))  # pelvis frame
```
- MP 결과: mean=-91.0°, std=32.3°, range=[-138.7°, -48.8°]
- GT 범위: mean=-0.2°, std=15.5°, range=[-23.2°, 19.0°] ✅ 생체역학 정상
- **문제**: MP가 ~-91° 거대 오프셋, GT보다 2배 큰 변동성

---

## 왜 엉덩이만 작동하는가?

1. **가장 강한 상관관계** (0.792) - 파형 모양 일치
2. **가장 작은 RMSE** (15.3°)
3. **가장 작은 중심화 RMSE** (7.3° - 임상적으로 허용 가능)
4. **가장 낮은 slope 편차** (0.474 vs ideal 1.0)

**가설**: `arctan2` 기반 엉덩이 각도 계산이 우연히 GT convention과 더 잘 맞음

---

## 다음 단계 옵션

### 옵션 A: 엉덩이만 진행 (단기 권장)

**접근**:
- Phase 2 (DTW + Bland-Altman)를 엉덩이 관절만 진행
- 엉덩이 ICC ≥ 0.50 목표

**장점**:
- ✅ 즉시 진행 가능
- ✅ 캘리브레이션 파이프라인 proof-of-concept 완료
- ✅ 엉덩이 각도는 임상적 가치 있음

**단점**:
- ❌ 전체 보행 분석 불가 (발목/무릎 필요)
- ❌ 발목/무릎 수정 미해결

### 옵션 B: MP 각도 계산 수정 (중기 권장)

**접근**:
1. 병원 GT 각도 계산 방법 조사
2. `core_modules/main_pipeline.py` 각도 계산 재작성
3. 해부학적 zero reference 맞춤
4. 전체 Phase 1 재실행

**예상 결과**:
- Deming slope 1.0에 가까워짐
- 3개 관절 모두 ICC ≥ 0.40 달성

**소요 시간**: 2-4시간 (코딩 + 검증)

**위험**: 기존 MP 처리 파이프라인 영향 가능

### 옵션 C: 다른 방법으로 ICC 개선

**방법**:
1. Subject-specific offset 학습
2. DTW temporal alignment
3. Quality filtering
4. Multi-cycle weighted averaging

**예상 개선**: 방법당 +0.1~+0.3 ICC

**소요 시간**: Phase 2-3 (이미 계획됨)

---

## 권장 조치

### 즉시 결정 필요

**질문**: 어떤 옵션으로 진행할까요?

1. **옵션 A - 엉덩이만 진행**:
   - Phase 2로 넘어가서 엉덩이 ICC 개선 (DTW + Bland-Altman)
   - 발목/무릎은 나중에 해결

2. **옵션 B - MP 각도 수정**:
   - MP 각도 계산 코드 재작성
   - 3개 관절 모두 작동하도록 수정
   - Phase 1 재실행

3. **옵션 C - 추가 조사**:
   - 병원에 GT 각도 계산 방법 문의
   - 원본 데이터 형식 재확인

---

## 달성한 것

1. ✅ **Deming 회귀 구현 완료**
   - 올바른 GT 소스 식별 (traditional_condition.csv Y 컬럼)
   - Sign flip 적용 (발목/엉덩이)
   - 평균 중심화 구현
   - 평균 보정 적용

2. ✅ **Butterworth 필터링 적용**
   - 연구 검증된 cutoff frequency 사용
   - 4차 low-pass filter

3. ✅ **검증 파이프라인 구축**
   - ICC, RMSE, correlation 계산
   - 개별 환자 GT와 비교

4. ✅ **문제 진단 완료**
   - MP 각도 계산 issue 식별
   - 생체역학적 범위 분석
   - 근본 원인 파악

---

## 해결 못한 것

1. ❌ **발목/무릎 ICC < 0.40**
   - 근본 원인: MP 각도 정의가 GT와 다름
   - Deming 만으로는 해결 불가

2. ❌ **전체 ICC 목표 미달**
   - 목표: ICC ≥ 0.40
   - 달성: ICC = -0.383

---

## 생성 파일

### 스크립트
- `improve_calibration_deming.py` - V1 (traditional_normals 사용)
- `improve_calibration_deming_v2.py` - V2 (traditional_condition Y 사용) ✅
- `apply_phase1_filtering_v4.py` - 최종 필터링 파이프라인 ✅

### 캘리브레이션 파라미터
- `calibration_parameters_deming.json` - V1 (normal range 기준)
- `calibration_parameters_deming_v2.json` - V2 (개별 환자 기준) ✅

### 결과 데이터
- `processed/S1_mediapipe_phase1_filtered_v4.json` - 필터링된 각도
- `processed/phase1_validation_summary_v4.csv` - 검증 요약
- `processed/phase1_validation_metrics_v4.csv` - 상세 메트릭

### 보고서
- `DEMING_DATA_MISMATCH_REPORT.md` - 초기 진단
- `DEMING_FIX_RESULTS.md` - 첫 번째 수정
- `DEMING_OPTIONB_RESULTS.md` - Y 컬럼 시도
- `ANGLE_DIRECTION_MISMATCH_REPORT.md` - 방향 불일치 분석
- `PHASE1_DIAGNOSTIC_REPORT.md` - 종합 진단 ✅
- `PHASE1_상태보고.md` - 한글 요약 (이 파일) ✅

---

**보고서 생성일**: 2025-11-08
**상태**: Phase 1 Day 1-5 완료, 부분 성공
**결정 필요**: 다음 단계 옵션 선택
