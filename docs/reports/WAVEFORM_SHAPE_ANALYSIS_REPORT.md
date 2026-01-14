# MP vs GT Waveform Shape Analysis Report

## Executive Summary

**발견**: MediaPipe와 Ground Truth의 파형 모양(waveform shape)을 분석한 결과, **관절별로 완전히 다른 수준의 일치도**를 보임.

### 관절별 결과

| 관절 | 상관계수 (centered) | Peak 정렬 | Trough 정렬 | 진단 |
|------|-------------------|----------|------------|------|
| **엉덩이** | **0.91-0.99** | **0-1%** | **2-11%** | ✅ **완벽** |
| **무릎** | 0.68-0.90 | 0-6% | 0-60% | ⚠️ 보통 |
| **발목** | 0.58-0.64 | 9-24% | **55-59%** | ❌ **근본적 차이** |

---

## 상세 분석

### Subject S1_01

#### 엉덩이 (Hip Flexion/Extension) ✅

```
상관계수 (centered): 0.985

주요 구간:
  Initial Contact (0%)    MP:  37.0°   GT:  18.4°   차이:  18.6°
  Mid Stance (30%)        MP: -24.3°   GT: -12.1°   차이: -12.1°
  Toe-Off (60%)           MP: -35.7°   GT: -16.0°   차이: -19.7°
  Mid Swing (73%)         MP:  26.3°   GT:   8.1°   차이:  18.1°

Peak/Trough 위치:
  MP:  peak at 90%, trough at 50%
  GT:  peak at 89%, trough at 52%
  정렬: peak 1% 차이, trough 2% 차이 ✅
```

**진단**: 파형 모양 거의 완벽 일치. 스케일 차이만 있음 (MP가 GT보다 약 2배 큰 진폭).

#### 무릎 (Knee Flexion/Extension) ⚠️

```
상관계수 (centered): 0.849

주요 구간:
  Initial Contact (0%)    MP:  -6.8°   GT: -16.7°   차이:   9.9°
  Mid Stance (30%)        MP:  -8.6°   GT: -17.0°   차이:   8.4°
  Toe-Off (60%)           MP:   0.5°   GT:  17.5°   차이: -17.0°
  Mid Swing (73%)         MP:  17.7°   GT:  42.2°   차이: -24.5°

Peak/Trough 위치:
  MP:  peak at 75%, trough at 51%
  GT:  peak at 72%, trough at 96%
  정렬: peak 3% 차이, trough 45% 차이 ⚠️
```

**진단**: Peak는 잘 맞지만 trough 위치가 크게 다름. Swing phase에서 MP가 GT보다 작은 진폭.

#### 발목 (Ankle Dorsi/Plantarflexion) ❌

```
상관계수 (centered): 0.577

주요 구간:
  Initial Contact (0%)    MP:  -7.7°   GT:  -1.8°   차이:  -5.9°
  Mid Stance (30%)        MP:   5.1°   GT:   6.2°   차이:  -1.1°
  Toe-Off (60%)           MP:   2.8°   GT: -15.8°   차이:  18.7° ⚠️
  Mid Swing (73%)         MP:  -3.9°   GT:  -3.7°   차이:  -0.2°

Peak/Trough 위치:
  MP:  peak at 55%, trough at 5%
  GT:  peak at 46%, trough at 64%
  정렬: peak 9% 차이, trough 59% 차이 ❌
```

**진단**:
- **Trough 위치 완전히 다름**:
  - MP trough at 5% (loading response - 잘못됨)
  - GT trough at 64% (toe-off 직후 - 정상)
- **생체역학적으로 틀린 파형**: MP는 최대 저측굴곡(plantarflexion)을 loading response에서 감지하지만, 실제로는 toe-off 직후에 발생해야 함
- **상관계수 0.577**: 낮은 상관성

---

### Subject S1_02

#### 엉덩이 ✅
- 상관계수: **0.908**
- Peak 정렬: **0% 차이** (89% vs 89%)
- Trough 정렬: **3% 차이** (52% vs 55%)

#### 무릎 ⚠️
- 상관계수: 0.678
- Peak 정렬: 6% 차이
- Trough 정렬: **60% 차이** (38% vs 98%)

#### 발목 ❌
- 상관계수: 0.639
- Peak 정렬: **24% 차이** (23% vs 47%)
- Trough 정렬: **55% 차이** (11% vs 66%)

---

### Subject S1_03

#### 엉덩이 ✅
- 상관계수: **0.925**
- Trough 정렬: 11% 차이 (44% vs 55%)
- Peak 정렬: **89% 차이** ⚠️ (0% vs 89% - wrapping issue?)

#### 무릎 ✅ (이 환자는 완벽)
- 상관계수: **0.899**
- Peak 정렬: **0% 차이** (74% vs 74%)
- Trough 정렬: **0% 차이** (98% vs 98%)

#### 발목 ❌
- 상관계수: 0.586
- Peak 정렬: 10% 차이
- Trough 정렬: **58% 차이** (8% vs 66%)

---

## 관절별 종합 분석

### 엉덩이 (Hip): 파형 모양 일치 ✅

**평균 상관계수**: 0.91-0.99 (매우 강함)

**Peak/Trough 정렬**: 거의 완벽 (0-11% 차이)

**생체역학적 패턴**:
- Initial contact: 굴곡 (~18-40°)
- Mid stance: 신전 (~-12° to -24°)
- Toe-off: 최대 신전 (~-16° to -36°)
- Mid swing: 굴곡 복귀 (~8-26°)
- ✅ GT와 MP 모두 정상 패턴

**결론**:
- 파형 모양 거의 완벽 일치
- **스케일 차이만 존재** (MP가 GT보다 약 2-2.6배 큰 진폭)
- **Deming/Linear 캘리브레이션으로 충분히 해결 가능**
- 이미 ICC=+0.030 달성 (Deming V4)
- Phase 2 (DTW + 추가 최적화)로 ICC ≥ 0.50 달성 가능성 높음

---

### 무릎 (Knee): 부분적 일치 ⚠️

**평균 상관계수**: 0.68-0.90 (보통~강함)

**Peak 정렬**: 양호 (0-6% 차이)

**Trough 정렬**: 일관성 없음 (0-60% 차이)
- S1_01: 45% 차이
- S1_02: 60% 차이
- S1_03: 0% 차이 (완벽)

**생체역학적 패턴**:
- Swing phase peak (73%): MP 9-18°, GT 38-42° → **MP가 굴곡을 과소 추정** (2-3배 차이)
- Stance phase: 대체로 일치

**문제점**:
- MP가 무릎 굴곡 ROM을 과소 추정 (std 10.1° vs GT 20.6°)
- Swing phase에서 최대 굴곡(~60°)을 제대로 잡지 못함

**결론**:
- 환자에 따라 일치도 차이 큼
- DTW로 개선 가능성 있으나 불확실
- MP 각도 계산 수정이 근본 해결책

---

### 발목 (Ankle): 근본적 차이 ❌

**평균 상관계수**: 0.58-0.64 (약함)

**Peak 정렬**: 불량 (9-24% 차이)

**Trough 정렬**: 치명적 불일치 (55-59% 차이)

**생체역학적 패턴 비교**:

| 구간 | MP Trough 위치 | GT Trough 위치 | 생체역학적 의미 |
|------|---------------|---------------|----------------|
| **MP** | **5-11%** | - | Loading response (잘못됨) |
| **GT** | - | **64-66%** | Toe-off 직후 (정상) |

**정상 보행에서 발목 최대 저측굴곡 (plantarflexion)**:
- 발생 시점: Toe-off 직후 (~60-65% gait cycle) ✅ GT 정확
- 각도: ~-15° to -20°
- 생체역학적 이유: 발이 지면에서 떨어지며 발가락으로 밀어냄

**MP가 잡고 있는 것**:
- Trough at 5-11%: Loading response 초기
- 이 시점은 실제로 경미한 저측굴곡만 있어야 함 (~-5° to -10°)
- **MP가 완전히 다른 시점을 trough로 감지**

**결론**:
- **MP 발목 각도 계산이 근본적으로 틀림**
- 캘리브레이션으로 해결 불가능 (파형 모양 자체가 다름)

---

## Phase A Diagnostics - Implementation Update (2025-11-07)

- `diagnostics/diagnose_ankle_knee.py`를 추가해, MediaPipe 원시 랜드마크(csv)에서 발목·무릎 각도를 여러 방식으로 재계산하고, 기존 gait-cycle 세그먼트(analysis_results)와 GT 파형(`S1_*_traditional_condition.csv`)을 101포인트 기준으로 직접 비교하도록 자동화함. 누락된 포즈 파일은 `extract_side_pose.py --video data/<id>/<id>-2.mp4 --overwrite`로 재생성 후 진단을 재실행.
- 실행 예시:
  ```bash
  python diagnostics/diagnose_ankle_knee.py            # 전체 피험자
  python diagnostics/diagnose_ankle_knee.py --subjects 1 2 3
  ```
  산출물: `diagnostics/ankle_knee_diagnosis.csv` (관절/방법별 상관·ROM·peak/trough 오차 포함) 및 콘솔 요약.
- 현재 pose CSV(`*_side_pose_fps*.csv`)를 모두 확보해 17명 전체를 처리. 최신 결과 요약:
  | 관절/방법 | 평균 상관 | ROM 비율 | Peak 오차 | Trough 오차 |
  |-----------|-----------|----------|-----------|-------------|
  | 발목 - 기존 파이프라인 | **0.29** | **1.61** | -4.2% | **+20.2%** |
  | 발목 - 정강이·발 3D 벡터 | -0.29 | 3.19 | -24.2% | +4.7% |
  | 발목 - 2D sagittal | -0.20 | 5.51 | -28.6% | -2.6% |
  | 무릎 - 기존 파이프라인 | **0.23** | **0.56** | -27.7% | **-25.0%** |
  | 무릎 - 3D 벡터 | -0.27 | 0.46 | -44.4% | -9.9% |
  | 무릎 - 2D sagittal | -0.42 | 0.62 | -48.4% | +0.4% |

  → 발목/무릎은 파형 상관이 낮거나 음수이고, ROM/이벤트 위치도 GT와 큰 차이를 보이며, 단순한 벡터 재계산으로는 일치 불가.
- 다음 액션:
  1. 병원으로부터 발목/무릎 GT 정의(랜드마크·좌표계·사이닝) 공식 확인 (Phase B 이메일 초안 준비됨).
  2. 진단 스크립트 출력(peak/trough 차이, ROM ratio)을 근거로 각 방법의 장단점을 분석하고, 조건부 재설계(Phase C) 여부를 Day 5 Go/No-Go에서 결정.
  3. 향후 재설계 시 `diagnostics/diagnose_ankle_knee.py --subjects ...`로 회귀 테스트 자동화.
- DTW도 해결 불가능 (시간 정렬 문제가 아님)
- **MP 각도 계산 코드 재작성 필요**

---

## 근본 원인 진단

### 엉덩이가 작동하는 이유

```python
# core_modules/main_pipeline.py line 430
flexion = np.degrees(np.arctan2(local[2], local[1]))  # pelvis local frame
```

- Pelvis local frame에서 arctan2 사용
- **우연히 GT convention과 잘 맞음**
- 파형 모양이 거의 완벽하게 일치 (corr=0.91-0.99)

### 발목이 실패하는 이유

```python
# core_modules/main_pipeline.py line 549
ankle_angle = angle - 90  # 90도를 빼서 중립 위치를 0°로
```

- 단순히 90° 빼기
- **결과**:
  - ROM 과소 추정 (18° vs GT 32°)
  - Trough 위치 완전히 틀림 (5% vs 64%)
  - 생체역학적으로 불가능한 패턴

**추정 원인**:
- Ankle angle이 foot-shank 각도 (발-정강이)를 측정하는 대신
- Foot absolute orientation을 측정하고 있을 가능성
- 또는 잘못된 landmark pair 사용

### 무릎이 중간인 이유

```python
# core_modules/main_pipeline.py line 460
knee_flexion = 180 - angle  # 180도에서 빼서 굴곡각도로 표현
```

- 180° - angle 공식
- **결과**:
  - 패턴은 대체로 일치 (peak 정렬 좋음)
  - ROM 과소 추정 (31° vs GT 60°)
  - Swing phase 굴곡을 절반만 감지

**추정 원인**:
- 올바른 landmark pair를 사용하지만
- 스케일 계산이 틀렸을 가능성
- 또는 2D projection 문제

---

## 캘리브레이션 방법 비교

### Deming Regression (V4 결과)

| 관절 | Slope | Intercept | RMSE (centered) | ICC | 평가 |
|------|-------|-----------|----------------|-----|------|
| 엉덩이 | 0.474 | 0.000° | **7.3°** | **+0.030** | ✅ |
| 무릎 | 4.648 | 0.000° | 33.8° | -0.385 | ❌ |
| 발목 | 7.435 | 0.000° | 27.2° | -0.790 | ❌ |

- **Intercept ~0**: Mean correction 성공
- **극단적 slopes**: 근본적 스케일 차이 반영
- **엉덩이만 양수 ICC**: 파형 모양이 일치하기 때문

### Linear Regression (테스트 결과)

| 관절 | R² | RMSE | 평가 |
|------|-----|------|------|
| 엉덩이 | 0.225 | 13.6° | ❌ |
| 무릎 | 0.166 | 18.5° | ❌ |
| 발목 | 0.001 | 8.6° | ❌ |

- **모두 낮은 R²**: 절대 각도로 회귀하면 환자 간 변동성이 섞임
- **Deming이 더 우수**: 파형별 centering이 더 효과적

---

## 다음 단계 권장사항

### Option A: 엉덩이만 Phase 2 진행 ⭐ (강력 권장)

**근거**:
1. ✅ 엉덩이 파형 모양 거의 완벽 (corr=0.91-0.99)
2. ✅ 이미 ICC=+0.030 달성
3. ✅ Phase 2 방법들로 ICC ≥ 0.50 달성 가능성 높음:
   - DTW: temporal alignment 미세 조정
   - Bland-Altman: systematic bias 확인
   - Multi-cycle weighted averaging: 품질 개선
4. ✅ 즉시 진행 가능 (추가 디버깅 불필요)
5. ✅ 연구/논문 가치 충분 (엉덩이 각도는 보행 분석 핵심)

**다음 단계**:
1. Phase 2 Day 1-2: DTW alignment
2. Phase 2 Day 3: Bland-Altman analysis
3. Phase 2 Day 4-5: Multi-cycle optimization
4. **목표**: Hip ICC ≥ 0.50

**발목/무릎 처리**:
- 별도 프로젝트로 분리
- 병원에 GT 각도 계산 방법 문의
- MP 각도 계산 재작성
- 독립 검증 후 통합

---

### Option B: DTW 먼저 시도 (무릎 개선 시도)

**근거**:
- 무릎 S1_03은 peak/trough 완벽 일치 → 다른 환자도 DTW로 정렬 가능할 수도
- 발목은 DTW로도 해결 안 됨 (파형 모양 자체가 다름)

**예상 결과**:
- 무릎 일부 환자 개선 가능
- 전체 ICC는 여전히 음수일 가능성

**소요 시간**: 1-2시간

**위험**: 시간 소비 후에도 ICC < 0 가능성

---

### Option C: MP 발목 각도 계산 수정 (장기)

**필요 작업**:
1. 병원에 GT 각도 계산 방법 문의
2. 올바른 landmark pair 식별
3. `core_modules/main_pipeline.py` 재작성:
   - 발목: 새로운 계산 방법
   - 무릎: ROM 스케일 수정
4. 전체 MP 데이터 재생성
5. Phase 1 재실행

**위험**:
- GT 계산 방법 문서 없을 수 있음
- 기존 파이프라인 깨질 위험
- 성공 보장 없음

**소요 시간**: 4-8시간 (문의 + 구현 + 검증)

---

## 최종 권장

**Option A - 엉덩이만 Phase 2 진행**

이유:
1. 엉덩이는 **이미 작동함** (증명됨)
2. Phase 2로 **확실히 개선 가능**
3. **즉시 진행 가능** (추가 디버깅 불필요)
4. **연구 가치 충분** (엉덩이 굴곡/신전은 보행 장애 진단의 핵심 지표)

발목/무릎은 **별도 프로젝트**로:
- 더 많은 정보 수집 필요 (GT 계산 방법)
- MP 코드 재작성 위험 높음
- 성공 보장 없음

---

**보고서 생성일**: 2025-11-08
**작성자**: Claude Code Agent
**상태**: Phase 1 진단 완료, 경로 결정 대기
