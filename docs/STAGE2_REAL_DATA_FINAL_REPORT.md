# STAGE 2: Pattern-Based Detection - Real Data Evaluation

**Date**: 2025-10-30
**Status**: COMPLETE
**Dataset**: Real GAVD MediaPipe patterns (230 videos, 62 minutes extraction)

---

## Executive Summary

실제 GAVD 데이터로 STAGE 2 Pattern-Based Detector를 평가한 결과, **DTW 기반 패턴 매칭만으로는 병리 보행 검출이 어렵다**는 것을 확인했습니다.

### Key Findings

1. **✅ 실제 패턴 추출 성공**
   - 264개 비디오 처리 (62분 소요, 순차 처리)
   - 230개 성공 추출 (87.1% 성공률)
   - 정상: 106개, 비정상: 124개

2. **❌ DTW 패턴 매칭의 한계**
   - Binary 정확도: **51.6%** (랜덤 수준)
   - Multi-class 정확도: **4.1%**
   - 템플릿 간 분리도: **0.06** (필요: >2.0)

3. **💡 핵심 발견**
   - 정상/비정상 보행 패턴의 **시간적 형태가 거의 동일**
   - DTW는 형태 매칭이므로 구별 불가능
   - Scalar features (진폭, 비대칭)도 차이 미미

---

## 1. 실제 데이터 추출

### 1.1 추출 과정

**데이터 소스**: GAVD Dataset
- 총 510개 비디오, 264개 사이드뷰 필터링
- MediaPipe Pose로 heel height 추출
- 각 패턴을 101 포인트로 정규화

**기술적 도전**:
- ❌ **멀티프로세싱 deadlock** (8 워커, 54분 후 정지)
  - MediaPipe + TensorFlow 스레드 풀 충돌
  - 각 워커가 1,700+ 스레드 생성
  - `futex_wait_queue_me` 상태로 멈춤

- ✅ **순차 처리로 해결**
  - 62분 만에 완료
  - 안정적 처리
  - 7.5초/비디오

### 1.2 추출 결과

```
총 처리: 264개 비디오
성공: 230개 (87.1%)
실패: 34개 (손상된 파일)

클래스 분포:
  - normal: 106개
  - abnormal: 73개
  - exercise: 28개
  - cerebral palsy: 8개
  - stroke: 7개
  - prosthetic: 5개
  - antalgic: 2개
  - inebriated: 1개

패턴 품질:
  - Valid left: 219/230 (95.2%)
  - Valid right: 219/230 (95.2%)
  - 평균 진폭: 4.2
```

---

## 2. Population-Based 템플릿 생성

### 2.1 Binary 템플릿

```
Normal Template:
  - Samples: 98
  - Mean amplitude (L/R): 4.11 / 4.14
  - Pattern variability: 0.98

Abnormal Template:
  - Samples: 121
  - Mean amplitude (L/R): 4.33 / 4.31
  - Pattern variability: 0.99
  - Includes: abnormal, stroke, cerebral palsy, prosthetic, antalgic, exercise
```

### 2.2 Multi-class 템플릿

7개 클래스 템플릿 생성:
- normal (98 samples)
- abnormal (71 samples)
- exercise (28 samples)
- cerebral palsy (8 samples)
- stroke (7 samples)
- prosthetic (5 samples)
- antalgic (2 samples)

---

## 3. STAGE 2 평가 결과

### 3.1 Binary Classification

```
Dataset: 219 samples

Accuracy: 51.6%
Sensitivity: 47.1%
Specificity: 57.1%

Confusion Matrix:
  TP (abnormal detected): 57
  TN (normal detected): 56
  FP (normal → abnormal): 42
  FN (abnormal → normal): 64
```

**⚠️ 랜덤 수준 성능 (50%)**

### 3.2 Multi-class Classification

```
Dataset: 219 samples, 7 classes

Overall Accuracy: 4.1%

Per-class Accuracy:
  - prosthetic: 80.0% (5 samples)
  - antalgic: 50.0% (2 samples)
  - stroke: 42.9% (7 samples)
  - cerebral palsy: 12.5% (8 samples)
  - exercise: 0.0% (28 samples)
  - abnormal: 0.0% (71 samples)
  - normal: 0.0% (98 samples)
```

**❌ 완전 실패** (랜덤: 14.3%)

---

## 4. 실패 원인 분석

### 4.1 DTW 거리 분석

```
정상 패턴 샘플:
  → normal 템플릿: 67.73
  → abnormal 템플릿: 66.08
  ❌ 잘못 분류 (abnormal이 더 가까움)

비정상 패턴 샘플:
  → normal 템플릿: 74.17
  → abnormal 템플릿: 73.61
  ✅ 올바르게 분류 (근소한 차이)
```

### 4.2 템플릿 분리도 분석

```
클래스 간 거리: 4.95
클래스 내 변동성:
  - Normal: 76.00
  - Abnormal: 77.72

분리 비율: 0.06

⚠️  필요 분리 비율: > 2.0
❌  실제 분리 비율: 0.06 (30배 부족!)
```

**결론**: 클래스 내 변동성이 클래스 간 차이보다 **15배 이상 크다**

### 4.3 Scalar Feature 분석

```
진폭 (mean):
  - Normal: 4.13
  - Abnormal: 4.35
  - 차이: 0.22 (5.3%)

비대칭 (|L-R|):
  - Normal: 0.50
  - Abnormal: 0.51
  - 차이: 0.01 (2.0%)
```

**결론**: Scalar features도 구별력 미미

---

## 5. 핵심 발견

### 5.1 왜 DTW가 실패했나?

**DTW (Dynamic Time Warping)**는:
- ✅ **시간적 형태(temporal shape)** 매칭에 강점
- ❌ **진폭 차이**에는 둔감
- ❌ **변동성 차이**에는 둔감

**정상 vs 비정상 보행**:
- 시간적 형태: **거의 동일** (양발이 교대로 움직이는 기본 패턴)
- 진폭/변동성: **약간 다름** (하지만 DTW가 포착 못함)

### 5.2 GAVD "abnormal" 클래스의 문제

GAVD의 "abnormal" 클래스는:
- **너무 일반적** (다양한 비정상 포함)
- **병리별 특성 없음** (stroke, cerebral palsy 등 혼합)
- **정상과 유사한 패턴도 포함** (경미한 이상)

### 5.3 Multi-class에서 일부 성공

```
Prosthetic: 80.0% (5 samples)
  → 의족 보행은 독특한 패턴 (DTW 적용 가능)

Stroke: 42.9% (7 samples)
  → 편마비 패턴이 어느 정도 구별 가능
```

**시사점**: **명확한 패턴 차이가 있는 병리**에는 DTW 유효

---

## 6. 결론 및 권장 사항

### 6.1 STAGE 2의 한계

**Pattern-Based Detection (DTW 템플릿 매칭)**:
- ❌ 일반 병리 검출에는 부적합
- ✅ 특정 병리 (의족, 편마비) 구별에는 유용
- ⚠️ STAGE 1 (scalar Z-score)보다 성능 낮음

### 6.2 STAGE 1 vs STAGE 2 비교

| Metric | STAGE 1 (Scalar) | STAGE 2 (DTW) |
|--------|-----------------|--------------|
| Binary Accuracy | 85-93% | 51.6% |
| Multi-class Accuracy | 51-56% (simulated) | 4.1% (real) |
| Computational Cost | Low | High |
| Interpretability | High | Low |

**결론**: **STAGE 1이 STAGE 2보다 우수**

### 6.3 권장 사항

#### Option A: STAGE 1 사용 (권장)
- Binary 검출에 집중 (normal vs abnormal)
- Scalar Z-score 기반 이상 탐지
- 85-93% 정확도 달성
- 실시간 처리 가능

#### Option B: STAGE 2 개선 (연구용)
- **특정 병리 전문 검출기** 개발
  - Prosthetic gait detector
  - Hemiplegic gait detector
- DTW + Scalar features 결합
- 병리별 별도 모델

#### Option C: 머신러닝 기반 접근
- GAVD 데이터로 CNN/LSTM 훈련
- End-to-end 학습
- Feature engineering 불필요
- 하지만 해석 가능성 낮음

---

## 7. 프로젝트 현황

### 7.1 완료된 작업

✅ **STAGE 1**: Baseline Detector
- Binary accuracy: 85-93%
- Z-score 기반 이상 탐지
- 실시간 처리 가능

✅ **STAGE 2**: Pattern-Based Detector
- 실제 GAVD 데이터 추출 (230 patterns)
- Population-based 템플릿 생성
- DTW 템플릿 매칭 평가
- 한계 분석 완료

✅ **Option B**: Right ICC 0.903 달성
- 프레임 제외 전략 성공
- 논문 기준 충족

### 7.2 다음 단계

**추천**: STAGE 1 기반 시스템 배포
1. STAGE 1 detector를 production으로
2. Real-time 웹/모바일 앱 개발
3. Clinical validation study

**선택적**: STAGE 2 연구 지속
1. 특정 병리 검출기 개발 (prosthetic, stroke)
2. DTW + scalar features 하이브리드
3. 학술 논문 발표

---

## 8. 파일 목록

### 생성된 파일

1. **extract_gavd_patterns.py** (420 lines)
   - 실제 GAVD 비디오에서 패턴 추출
   - MediaPipe 기반 heel height 추출
   - 순차 처리 (deadlock 방지)

2. **gavd_real_patterns.json** (5.2 MB)
   - 230개 추출된 실제 패턴
   - 101 포인트로 정규화
   - 진폭, 피크 타이밍 등 메타데이터

3. **evaluate_stage2_real_data.py** (400 lines)
   - Population-based 템플릿 생성
   - Binary/Multi-class 평가
   - DTW 거리 계산

4. **stage2_real_data_results.json**
   - 평가 결과 저장
   - Binary: 51.6%
   - Multi-class: 4.1%

5. **STAGE2_REAL_DATA_FINAL_REPORT.md** (this file)
   - 전체 분석 및 결론

### 로그 파일

- `gavd_extraction_sequential.log` - 순차 추출 로그
- `stage2_real_evaluation.log` - 평가 로그

---

## 9. 교훈

### 9.1 기술적 교훈

1. **멀티프로세싱 + MediaPipe = Deadlock**
   - TensorFlow Lite 스레드 풀 충돌
   - 순차 처리가 더 안정적

2. **DTW의 한계**
   - 형태 매칭에만 유용
   - 진폭/변동성 차이 포착 못함
   - 일반 병리 검출에는 부적합

3. **실제 데이터의 복잡성**
   - "abnormal" 클래스가 너무 이질적
   - 병리별 구분 필요
   - 시뮬레이션 ≠ 실제

### 9.2 연구 교훈

1. **Simpler is Better**
   - STAGE 1 (scalar) > STAGE 2 (DTW)
   - 해석 가능성 중요
   - 계산 비용 고려

2. **Problem Definition 중요**
   - Binary vs Multi-class
   - 일반 검출 vs 병리별 검출
   - 목적에 맞는 방법 선택

3. **Data Quality > Algorithm**
   - GAVD 데이터의 레이블 품질
   - 병리별 충분한 샘플 필요
   - 클래스 불균형 문제

---

## 10. 최종 권장사항

### For Clinical Deployment

**STAGE 1 Baseline Detector 사용**
- ✅ 85-93% binary accuracy
- ✅ 실시간 처리 가능
- ✅ 해석 가능
- ✅ 검증 완료

### For Research

**특정 병리 검출기 개발**
- Prosthetic gait: DTW 유효 (80%)
- Hemiplegic gait: DTW 부분 유효 (43%)
- 병리별 전문화된 접근

### For Future Work

**머신러닝 접근 검토**
- CNN/LSTM for gait classification
- End-to-end learning
- Larger dataset 필요

---

**Report Complete**: 2025-10-30
**Total Processing Time**: 62 minutes (pattern extraction) + 5 minutes (evaluation)
**Success Rate**: 87.1% extraction, 51.6% binary classification
**Conclusion**: STAGE 1 > STAGE 2 for general pathological gait detection
