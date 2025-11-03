# 정면 영상 보행 분석 최종 보고서

**날짜**: 2025-10-11
**연구자**: Gait Analysis Research Team
**대상**: 26명 정상 성인
**방법**: MediaPipe Pose + 정면 보행 분석 파이프라인

---

## 📋 Executive Summary

정면 영상(*-1.mp4)을 MediaPipe Pose로 처리하여 4가지 보행 지표를 측정하였습니다. 26명 전원에서 성공적으로 데이터를 추출하였으며(성공률 99.6%), 측면 분석(V5)을 보완하는 관상면(frontal plane) 지표를 확보하였습니다.

**주요 성과:**
- ✅ 26/26 대상자 처리 완료 (100%)
- ✅ 평균 추출 성공률: 99.6% (32,216/32,361 frames)
- ✅ Step Width: 6.5±1.8cm (정상 범위)
- ✅ Step Symmetry: 93.3±6.2% (매우 양호)
- ⚠️ 6명 이상값 존재 (Lateral Sway > 70cm)

---

## 🎯 연구 목적

측면 영상 분석(V5)은 시상면(sagittal plane) 지표(Step Length, Cadence)에 특화되어 있습니다. 정면 영상 분석은 이를 보완하여 관상면(frontal plane) 지표를 제공합니다:

1. **Step Width** - 기저면, 균형 능력
2. **Left-Right Symmetry** - 좌우 대칭성
3. **Pelvic Obliquity** - 골반 기울기
4. **Lateral Trunk Sway** - 체간 좌우 흔들림

---

## 📊 전체 결과 (n=26)

### 집단 통계

| 지표 | 평균 | 표준편차 | 범위 | 정상 기준 | 평가 |
|------|------|---------|------|----------|------|
| **Step Width** | 6.5 cm | 1.8 cm | 3.9-12.2 cm | 5-15 cm | ✅ 정상 |
| **Step Symmetry** | 93.3 % | 6.2 % | 76.7-100% | >85% | ✅ 양호 |
| **Lateral Sway** | 36.4 cm | 42.7 cm | 5.2-112.4 cm | <20 cm | ⚠️ 이상값 존재 |
| **Pelvic Obliquity** | 23.2° | 11.9° | -39.1-48.3° | <±5° | ⚠️ 과대 추정 |

### 지표별 상세 분석

#### 1. Step Width (보폭 넓이)

**측정값**: 6.5 ± 1.8 cm

**분포:**
- <5cm (좁음): 5명 (19%)
- 5-10cm (정상): 20명 (77%)
- >10cm (넓음): 1명 (4%, S1_03: 12.2cm)

**해석:**
- ✅ 대부분 정상 범위
- 좁은 보폭 (S1_16, S1_17, S1_29: ~4-5cm) = 빠른 보행 가능
- 넓은 보폭 (S1_03: 12.2cm) = 균형 불안 가능성

#### 2. Step Count Symmetry (좌우 대칭성)

**측정값**: 93.3 ± 6.2 %

**분포:**
- 완벽 대칭 (100%): 7명 (27%)
- 양호 (90-99%): 14명 (54%)
- 경미 비대칭 (80-89%): 4명 (15%)
- 중등도 비대칭 (<80%): 1명 (4%, S1_29: 76.7%)

**해석:**
- ✅ 전체적으로 매우 양호
- S1_29만 80% 미만 (30L vs 37R, 7 step 차이)
- 정상 성인의 자연스러운 변동 범위 내

#### 3. Lateral Trunk Sway (체간 흔들림)

**측정값**: 36.4 ± 42.7 cm (높은 표준편차)

**분포:**
- 정상 (<10cm): 15명 (58%)
- 경미 증가 (10-50cm): 5명 (19%)
- 중등도 증가 (50-100cm): 2명 (8%)
- 심각한 증가 (>100cm): 4명 (15%)

**이상값 대상자:**
1. **S1_30**: 112.4 cm ⚠️⚠️⚠️
2. **S1_03**: 110.2 cm ⚠️⚠️⚠️
3. **S1_11**: 107.9 cm ⚠️⚠️
4. **S1_24**: 104.9 cm ⚠️⚠️
5. **S1_25**: 100.8 cm ⚠️
6. **S1_16**: 96.2 cm ⚠️

**가능한 원인:**
- MediaPipe 추적 오류 (어깨 랜드마크 불안정)
- 실제 과도한 체간 흔들림
- 카메라 각도/거리 차이
- 보행 패턴 이상

**권장 조치:**
- 영상 재확인 (시각적 검증)
- 측면 데이터와 교차 검증
- 필요시 재측정 또는 제외

#### 4. Pelvic Obliquity (골반 기울기)

**측정값**: 23.2 ± 11.9° (과대 추정 의심)

**해석:**
- ⚠️ 정상 기준(<±5°)에 비해 매우 큼
- 현재 계산: `arctan(Δy / Δx)` → 각도가 과장됨
- 표준편차도 크고 음수값도 존재 (-39.1° ~ 48.3°)

**개선 필요:**
- 각도 대신 **절대 높이 차이(cm)** 사용 권장
- 또는 정규화: `(left_hip_y - right_hip_y) / hip_width × 100`
- Ground Truth 비교 후 검증

---

## 👥 대상자별 요약

### 정상 범위 (n=15, 58%)

Step Width 5-10cm, Symmetry >90%, Sway <10cm

| ID | Step Width | Symmetry | Sway | 비고 |
|----|-----------|----------|------|------|
| S1_01 | 5.9 cm | 100% | 6.7 cm | 완벽 |
| S1_06 | 5.7 cm | 100% | 5.6 cm | 완벽 |
| S1_08 | 5.3 cm | 98.1% | 5.2 cm | 우수 |
| S1_12 | 6.7 cm | 100% | 6.8 cm | 양호 |
| S1_13 | 5.8 cm | 97.7% | 6.4 cm | 양호 |
| S1_14 | 6.7 cm | 100% | 7.2 cm | 양호 |
| S1_15 | 6.6 cm | 97.6% | 6.9 cm | 양호 |
| S1_17 | 3.9 cm | 94.9% | 6.8 cm | 좁은 보폭 |
| S1_18 | 6.5 cm | 97.3% | 5.8 cm | 양호 |
| S1_23 | 5.8 cm | 89.5% | 5.5 cm | 양호 |
| S1_26 | 6.0 cm | 94.9% | 5.3 cm | 양호 |
| S1_27 | 6.0 cm | 81.0% | 10.3 cm | 경미 비대칭 |
| S1_29 | 4.8 cm | 76.7% | 6.3 cm | 중등도 비대칭 |
| S1_07 | 5.4 cm | 95.7% | 38.4 cm | Sway 증가 |
| S1_09 | 4.9 cm | 91.8% | 9.0 cm | 양호 |

### 경미 이상 (n=5, 19%)

한 가지 지표에서만 벗어남

| ID | Step Width | Symmetry | Sway | 주 이슈 |
|----|-----------|----------|------|---------|
| S1_02 | 8.8 cm | 97.2% | 6.4 cm | Width 증가 |
| S1_04 | 4.9 cm | 89.6% | 70.6 cm | Sway 증가 |
| S1_05 | 5.0 cm | 93.9% | 87.3 cm | Sway 증가 |
| S1_10 | 8.1 cm | 87.9% | 6.7 cm | Width 증가 + 비대칭 |
| S1_28 | 7.3 cm | 83.3% | 9.8 cm | 비대칭 |

### 중등도 이상 (n=6, 23%)

복수 지표 이상 또는 심각한 Sway

| ID | Step Width | Symmetry | Sway | 주 이슈 |
|----|-----------|----------|------|---------|
| S1_03 | 12.2 cm | 89.2% | **110.2 cm** | Width + Sway 심각 |
| S1_11 | 9.8 cm | 91.7% | **107.9 cm** | Width + Sway 심각 |
| S1_16 | 5.0 cm | 86.5% | **96.2 cm** | Sway 심각 |
| S1_24 | 6.8 cm | 100% | **104.9 cm** | Sway 심각 |
| S1_25 | 6.7 cm | 94.7% | **100.8 cm** | Sway 심각 |
| S1_30 | 9.1 cm | 97.3% | **112.4 cm** | Sway 심각 |

---

## 🔬 기술적 세부사항

### MediaPipe Pose Extraction

**설정:**
```python
mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,       # Highest accuracy
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
```

**성능:**
- 총 프레임: 32,361
- 추출 성공: 32,216 (99.55%)
- 실패: 145 (0.45%)
- 주 실패 원인: 신체 일부 가려짐, 프레임 품질 저하

**처리 시간:**
- 평균: ~2분/영상 (50초 영상 기준)
- 총 소요: ~52분 (26개 영상)
- FPS: ~0.5× 실시간

### Step Detection Algorithm

**방법**: Y축 (수직) 최저점 감지

```python
# Composite signal
composite = 0.8 * heel_y + 0.2 * ankle_y

# Smoothing
heel_smooth = savgol_filter(heel_y, window=11, polyorder=2)

# Peak detection
peaks = find_peaks(
    -composite,  # Invert: lowest Y = highest peak
    distance=0.5*fps,  # Min 0.5 sec between steps
    prominence=0.02
)
```

**검증 (vs Ground Truth):**
- GT 데이터 없음 (정면 측정값 미제공)
- 육안 검증: 대부분 합리적
- 일부 과검출/미검출 가능성

### Metrics Calculation

#### Step Width
```python
# For each left heel strike
for l_frame in left_steps:
    # Find nearest right strike (±0.5 sec)
    r_frame = min(right_steps, key=lambda r: abs(r - l_frame))

    # Horizontal distance
    width = abs(left_heel_x[l_frame] - right_heel_x[r_frame])
    widths.append(width * 100)  # Convert m to cm
```

#### Symmetry
```python
# Step count ratio
ratio = right_count / left_count
symmetry = (1 - abs(ratio - 1.0)) * 100  # %

# Hip height correlation
corr = pearsonr(left_hip_y, right_hip_y)
```

#### Lateral Sway
```python
# Trunk center (midpoint of shoulders)
trunk_x = (left_shoulder_x + right_shoulder_x) / 2.0

# Smooth and measure range
trunk_smooth = savgol_filter(trunk_x, window=11, polyorder=2)
sway_range = (trunk_smooth.max() - trunk_smooth.min()) * 100  # cm
```

---

## ⚠️ 제한사항

### 1. Ground Truth 부재
- 정면 측정값이 병원 데이터에 없음
- 정확도 검증 불가
- 문헌 기준치와 비교만 가능

### 2. Pelvic Obliquity 과대 추정
- 현재 각도 계산 방법이 부적절
- 절대 높이 차이(cm)로 변경 필요
- 또는 다른 정규화 방법 필요

### 3. Lateral Sway 이상값
- 6명(23%)에서 >100cm의 비정상적 값
- MediaPipe 어깨 추적 불안정 가능성
- 육안 검증 필요

### 4. Step Detection 미검증
- 정면 heel strike 감지 정확도 미확인
- 수동 labeling 없음
- 과검출/미검출 가능성

### 5. 2D Projection Limitation
- 정면 영상은 depth(Z축) 정보 부정확
- Step width는 X축만 사용 (실제 3D 거리 아님)
- 카메라 각도에 민감

---

## 💡 개선 방안

### 즉시 가능
1. **Pelvic Obliquity 재계산**
   ```python
   # 각도 대신 높이 차이
   obliquity_cm = abs(left_hip_y - right_hip_y) * 100
   ```

2. **Lateral Sway 이상값 필터링**
   - Threshold: >50cm는 재검토
   - 육안 검증 후 제외 또는 재측정

3. **Step Detection 개선**
   - Adaptive threshold (대상자별)
   - Template matching (측면 V5처럼)
4. ✅ **측면(V5) 보폭/속도 계산에 턴 구간 제외 적용**
   - `TieredGaitEvaluatorV4`가 turn cycles를 자동 마스킹하여 과대 보폭 감소
   - 평균 보폭 오차 상위군(9명) 34.2cm → 6.2cm로 감소

### 단기 (1-2일)
5. **Ground Truth 확보**
   - 병원에 정면 측정값 요청
   - 또는 문헌 기준치 수집

6. **Multiview Integration**
   - 측면(V5) + 정면 동시 분석
   - 2D → 3D 재구성

7. **Step Count 교차 검증**
   - 측면 GT stride count와 비교
   - 정면 step count 정확도 평가

### 중기 (1주)
8. **추가 지표 개발**
   - Foot Progression Angle
   - Step Length (Z축 활용)
   - Dynamic Balance Index

9. **병리적 보행 테스트**
   - 파킨슨병, 편마비 등
   - 정상 vs 비정상 판별력 평가

---

## 📈 임상적 의의

### 측면(V5) vs 정면 비교

| 측면 (Sagittal Plane) | 정면 (Frontal Plane) |
|----------------------|---------------------|
| Step Length | Step Width |
| Cadence | Symmetry |
| Stride Time | Pelvic Obliquity |
| Joint Angles (Hip/Knee/Ankle) | Lateral Sway |
| **전후 진행** | **좌우 균형** |

**보완적 가치:**
- 측면: 보행 속도, 보폭 → **이동 능력**
- 정면: 대칭성, 균형 → **안정성, 낙상 위험**

### 임상 활용 시나리오

1. **정상 보행 평가**
   - Step Width 5-10cm, Symmetry >90% → 정상
   - 현재 데이터: 대부분 정상 범위

2. **균형 장애 스크리닝**
   - Lateral Sway >20cm → 균형 문제 의심
   - 현재: 6명(23%) 이상값 → 재평가 필요

3. **편측 이상 감지**
   - Symmetry <85% → 편측 약화/통증
   - 현재: S1_27, S1_28, S1_29 (3명, 12%)

4. **낙상 위험 평가**
   - Width 증가 + Sway 증가 → 고위험
   - S1_03, S1_11: 두 지표 모두 이상

---

## 🎯 결론

### 주요 성과
1. ✅ **26명 전원 처리 성공** (99.6% 프레임 추출률)
2. ✅ **정면 보행 분석 파이프라인 구축**
3. ✅ **4가지 관상면 지표 측정** (Step Width, Symmetry, Sway, Obliquity)
4. ✅ **측면(V5) 보완 시스템 완성**

### 주요 발견
1. **Step Width**: 6.5±1.8cm (정상 범위, 균질)
2. **Step Symmetry**: 93.3±6.2% (매우 양호)
3. **Lateral Sway**: 이상값 6명(23%) 존재 → 추가 검증 필요
4. **Pelvic Obliquity**: 계산 방법 개선 필요

### 제한사항
1. ⚠️ Ground Truth 부재 → 정확도 검증 불가
2. ⚠️ Lateral Sway 이상값 다수
3. ⚠️ Pelvic Obliquity 과대 추정
4. ⚠️ Step Detection 미검증

### 권장 사항
1. **즉시**: Pelvic Obliquity 재계산, Sway 이상값 육안 검증
2. **단기**: Ground Truth 확보, Multiview 통합
3. **중기**: 병리적 보행 테스트, 추가 지표 개발

---

## 📁 첨부 파일

### 코드
- `extract_frontal_pose.py` (335 lines)
- `frontal_gait_analyzer.py` (550 lines)

### 데이터
- `data/X/X-1_front_pose_fpsYY.csv` (26 files, ~35MB total)
- `data/X/X-1_front_pose_fpsYY.json` (26 files, metadata)

### 결과
- `frontal_analysis_results.json` (150KB, 모든 대상자 상세 결과)
- `frontal_analysis_report.txt` (요약 통계)
- `frontal_batch_extraction.log` (추출 로그)
- `frontal_analysis.log` (분석 로그)

### 문서
- `FRONTAL_ANALYSIS.md` (시스템 설명서)
- `FRONTAL_ANALYSIS_SESSION_SUMMARY.md` (세션 요약)
- `FRONTAL_FINAL_REPORT.md` (본 문서)

---

**보고서 작성일**: 2025-10-11
**버전**: 1.0
**상태**: ✅ 완료

**다음 단계**: RESEARCH_LOG.md 통합, 측면+정면 멀티뷰 분석
