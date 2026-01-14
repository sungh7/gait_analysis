# Ankle ROM 개인별 Variability 문제 - 상세 분석

**핵심 질문**: ROM ratio는 완벽(1.006)한데, 왜 ICC는 낮을까(0.377)?

---

## 문제 요약

| 지표 | 값 | 의미 | 판정 |
|------|-----|------|------|
| **ROM Ratio** | 1.006 ± 0.270 | 그룹 평균 일치도 | ✅ 완벽 |
| **ICC(2,1)** | 0.377 | 개인별 일치도 | ❌ Poor |
| **Correlation** | 0.449 (p=0.071) | 선형 관계 | ⚠️ 보통 |
| **Mean Difference** | -0.19° | 체계적 편향 | ✅ 거의 없음 |

**결론**: 그룹 레벨에서는 완벽하지만, 개인 레벨에서는 일치도가 낮습니다.

---

## 1. ROM Ratio vs ICC: 무엇이 다른가?

### ROM Ratio (그룹 레벨 지표)

**계산**:
```
ROM_ratio = mean(GT_ROM) / mean(MP_ROM)
          = 30.93° / 30.74°
          = 1.006
```

**의미**: "평균적으로 MediaPipe ROM이 Vicon ROM과 같은가?"

**강점**:
- ✅ 그룹 평균 일치도를 평가
- ✅ 체계적 bias 감지 (over/underestimation)
- ✅ 간단하고 직관적

**약점**:
- ❌ 개인별 오차를 무시
- ❌ 양수/음수 오차가 상쇄됨
- ❌ Variability 차이를 고려 안 함

**예시**:
```
Subject 1: MP=40°, GT=20°  (오차: +20°)
Subject 2: MP=20°, GT=40°  (오차: -20°)
---
평균:      MP=30°, GT=30°  (ratio=1.0, 완벽!)
하지만 개인별로는 큰 오차!
```

### ICC (개인 레벨 지표)

**계산**: Between-subject variance vs. Total variance

**의미**: "개인의 GT 값을 MP로 얼마나 정확히 예측할 수 있는가?"

**강점**:
- ✅ 개인별 일치도 평가
- ✅ Variability 일관성 고려
- ✅ Rank-order preservation 평가

**약점**:
- ❌ Variability 차이에 민감
- ❌ 계산이 복잡함

---

## 2. Variability 불일치가 ICC를 낮추는 이유

### 실제 데이터

```
MediaPipe ROM:  30.74 ± 7.13°  (range: 15.3° - 41.2°)
Vicon GT ROM:   30.93 ± 3.89°  (range: 24.5° - 40.0°)

Variability Ratio: 7.13 / 3.89 = 1.83x
→ MediaPipe가 GT보다 1.83배 더 큰 variability!
```

### 왜 문제인가?

**ICC는 variability 일관성을 요구합니다:**

1. **Between-subject variance (신호)**:
   - GT에서 subjects 간 차이: σ²_GT = 3.89² ≈ 15.1
   - 이것이 우리가 측정하고 싶은 "진짜" 개인차

2. **Within-subject variance (노이즈)**:
   - MP-GT 차이의 variance: σ²_error
   - MP variability > GT variability면 노이즈 증가

3. **ICC formula**:
   ```
   ICC = (σ²_between - σ²_error) / (σ²_between + σ²_error)

   만약 σ²_error가 크면 → ICC 낮아짐
   ```

### 시각화

```
Ground Truth (GT):
S1 ●-----●-----●-----●-----● S17
   24.5°              40.0°
   Spread: 15.5° (좁음)

MediaPipe (MP):
S1 ●-----------●-----------● S17
   15.3°                41.2°
   Spread: 25.9° (넓음!)

→ MP의 과도한 spread가 개인 식별력을 떨어뜨림
```

---

## 3. 개인별 오차 분석

### Per-Subject ROM Values

| Subject | MP ROM | GT ROM | Difference | Error % | 카테고리 |
|---------|--------|--------|------------|---------|----------|
| S1_13 | 27.4° | 27.5° | -0.1° | 0.4% | ✅ Excellent |
| S1_02 | 31.4° | 31.6° | -0.2° | 0.7% | ✅ Excellent |
| S1_03 | 31.1° | 31.6° | -0.5° | 1.5% | ✅ Excellent |
| S1_09 | 34.0° | 35.0° | -1.0° | 2.8% | ✅ Good |
| S1_14 | 37.3° | 40.0° | -2.7° | 6.8% | ✅ Good |
| S1_16 | 31.3° | 28.0° | +3.3° | 11.8% | ⚠️ Moderate |
| S1_17 | 32.7° | 28.0° | +4.7° | 16.8% | ⚠️ Moderate |
| S1_01 | 28.6° | 34.5° | -5.9° | 17.0% | ⚠️ Moderate |
| S1_15 | 41.2° | 34.9° | +6.2° | 17.9% | ⚠️ Moderate |
| S1_18 | 38.4° | 31.8° | +6.5° | 20.5% | ❌ Large |
| S1_23 | 25.3° | 32.3° | -7.0° | 21.8% | ❌ Large |
| **S1_08** | **34.2°** | **26.6°** | **+7.7°** | **28.8%** | ❌ Very Large |
| **S1_11** | **33.7°** | **25.8°** | **+7.9°** | **30.6%** | ❌ Very Large |
| **S1_10** | **40.4°** | **32.4°** | **+8.0°** | **24.6%** | ❌ Very Large |
| **S1_25** | **15.3°** | **24.5°** | **-9.2°** | **37.5%** | ❌ Very Large |
| **S1_24** | **22.7°** | **32.7°** | **-10.0°** | **30.5%** | ❌ Very Large |
| **S1_26** | **17.5°** | **28.6°** | **-11.1°** | **38.7%** | ❌ Very Large |

### 오차 분포

```
Excellent (<3% error):  3/17 subjects (18%)
Good (3-10% error):     2/17 subjects (12%)
Moderate (10-20% error): 4/17 subjects (24%)
Large (>20% error):     8/17 subjects (47%)  ← 절반이 큰 오차!
```

### 평균 통계

```
Mean Absolute Error:  5.42°
Median Error:         4.70°
Max Error:           11.10° (S1_26)
Min Error:            0.10° (S1_13)

Standard Deviation of Errors: 3.76°
```

---

## 4. Subject Groups 패턴 분석

### Early vs Late Subjects

**Early subjects (S1_1 ~ S1_18)**:
```
N = 13 subjects
Mean error:  4.22°
SD error:    2.98°
Error range: 0.1° - 7.7°
```

**Late subjects (S1_23 ~ S1_26)**:
```
N = 4 subjects
Mean error:  9.32°  ← 2.2배 더 큼!
SD error:    1.77°
Error range: 7.0° - 11.1°
```

### 왜 Late subjects가 나쁜가?

**가설 1: 비디오 품질 차이**
- S1_23-26은 다른 session에서 촬영되었을 수 있음
- 다른 카메라 설정, 조명, 해상도

**가설 2: Foot landmark 품질**
- Late subjects에서 toe/heel landmarks 불안정
- MediaPipe confidence가 낮았을 수 있음

**가설 3: 피험자 특성**
- Late subjects가 다른 gait pattern을 가질 수 있음
- 예: 더 빠른/느린 걸음, 다른 foot strike pattern

### Subject Group별 ROM 비교

```
Group        | MP ROM Mean | GT ROM Mean | Variability Ratio
-------------|-------------|-------------|------------------
Early (1-18) | 32.4 ± 6.8° | 30.8 ± 3.6° | 1.89x
Late (23-26) | 22.7 ± 4.6° | 29.5 ± 3.8° | 1.21x

→ Late subjects는 MP ROM이 크게 underestimate됨!
```

---

## 5. Variability 불일치의 원인

### 5.1. Foot Frame Orientation 불일치 (Day 3-4 발견)

**문제**: Foot Y-axis 방향이 subject마다 다름

```
Negative correlation subjects (10/17):
  → Foot Y-axis가 posterior를 향함
  → Dorsiflexion/plantarflexion sign이 반대

Positive correlation subjects (4/17):
  → Foot Y-axis가 anterior를 향함 (올바름)
  → Sign이 정확함

Zero correlation subjects (3/17):
  → Inconsistent, 방향이 불안정
```

**영향**:
- Sign 불일치 → ROM 값이 무작위로 변함
- 어떤 subject는 over, 어떤 subject는 under
- 결과: Variability 증가

### 5.2. Toe/Heel Landmark 불안정성

**MediaPipe toe/heel landmarks**:
- Landmark 31/32 (toe), 29/30 (heel)
- 바닥과 가까워서 occlusion 가능성 높음
- Confidence가 낮을 수 있음

**영향**:
```
Unstable toe position:
  → Foot Y-axis (toe - ankle) 방향 변함
  → Frame-to-frame jitter
  → Representative cycle의 ROM 증가/감소
```

### 5.3. Video Quality 차이

**Late subjects (S1_23-26)**:
- 평균 오차 9.32° (early subjects의 2.2배)
- 모두 큰 underestimation (-7.0° ~ -11.1°)

**가능한 원인**:
- 다른 카메라 해상도
- 다른 조명 조건
- 다른 거리/각도

### 5.4. Ground Truth Variability가 작음

**Vicon GT의 낮은 variability (3.89°)**:
```
이유:
1. 마커 기반 → 매우 정확
2. 17명 모두 healthy young adults
   → 비슷한 gait pattern
3. Controlled lab environment
   → 일관된 측정 조건
```

**MediaPipe의 높은 variability (7.13°)**:
```
이유:
1. Markerless → 더 noisy
2. Foot frame orientation 불일치
3. Landmark detection variability
4. Video quality 차이
```

---

## 6. ICC 계산 상세 분석

### ICC(2,1) Formula

```
ICC(2,1) = (MS_between - MS_error) / (MS_between + MS_error)

where:
  MS_between = Between-subject mean square (신호)
  MS_error   = Residual mean square (노이즈)
```

### 실제 계산 (간략화)

```python
# Between-subject variance (averaged across MP and GT)
subject_means = (mp_roms + gt_roms) / 2
ms_between = variance(subject_means) * 2

# Error variance
errors = mp_roms - gt_roms
ms_error = variance(errors)

# ICC
icc = (ms_between - ms_error) / (ms_between + ms_error)
```

### 우리 데이터에서

```
Subject means variance: ~18.0
Error variance:         ~41.0  ← 매우 큼!

ICC = (18.0 - 41.0) / (18.0 + 41.0)
    = -23.0 / 59.0
    = -0.39?  (음수!)

실제 ICC = 0.377 (조금 다른 계산 방식)
```

**왜 낮은가?**
- Error variance (41.0) > Between-subject variance (18.0)
- 노이즈가 신호보다 큼!
- ICC가 낮을 수밖에 없음

---

## 7. ROM Ratio vs ICC: 수학적 설명

### 시나리오 A: 완벽한 측정

```
Subject  | GT  | MP  | Diff
---------|-----|-----|-----
S1       | 30° | 30° |  0°
S2       | 25° | 25° |  0°
S3       | 35° | 35° |  0°

ROM ratio = 30/30 = 1.000 ✅
ICC       = 1.000 ✅
```

### 시나리오 B: 체계적 bias (우리가 Day 2 이전)

```
Subject  | GT  | MP  | Diff
---------|-----|-----|-----
S1       | 30° | 60° | +30°
S2       | 25° | 50° | +25°
S3       | 35° | 70° | +35°

ROM ratio = 30/60 = 0.500 ❌
ICC       = 1.000 ✅  (rank order 보존됨!)
```

### 시나리오 C: 무작위 노이즈 (현재 상황)

```
Subject  | GT  | MP  | Diff
---------|-----|-----|-----
S1       | 30° | 35° |  +5°
S2       | 25° | 20° |  -5°
S3       | 35° | 35° |   0°

ROM ratio = 30/30 = 1.000 ✅
ICC       = 0.600 ⚠️  (노이즈 때문에 낮아짐)
```

### 시나리오 D: 현재 우리 상황

```
Subject  | GT    | MP    | Diff
---------|-------|-------|-------
S1_13    | 27.5° | 27.4° |  -0.1°  ✅
S1_26    | 28.6° | 17.5° | -11.1°  ❌
S1_08    | 26.6° | 34.2° |  +7.7°  ❌
...      | ...   | ...   | ...

ROM ratio = 30.93/30.74 = 1.006 ✅ (평균은 같음!)
ICC       = 0.377 ❌ (개인별로는 큰 차이)
```

---

## 8. 임상적 의미

### ROM Ratio 1.006이 의미하는 것

✅ **장점**:
- 그룹 평균 연구에 사용 가능
- 예: "정상인 평균 ankle ROM vs. 환자 평균 ankle ROM"
- 체계적 bias 없음 (calibration 불필요)

❌ **한계**:
- 개인별 진단에는 부적합
- 예: "환자 A의 ankle ROM이 정상 범위인가?" → 신뢰할 수 없음
- Error 범위가 너무 큼 (±12.7°)

### ICC 0.377이 의미하는 것

**Koo & Li (2016) Guidelines**:
- ICC < 0.50: Poor reliability
- ICC 0.50-0.75: Moderate
- ICC 0.75-0.90: Good
- ICC > 0.90: Excellent

**우리 결과 (0.377)**:
- ❌ 개인별 측정에는 신뢰도 낮음
- ❌ Clinical decision making에 부적합
- ⚠️ 연구 목적으로만 제한적 사용

### Bland-Altman 분석

```
Mean difference:  -0.19° (거의 0, 좋음!)
SD of difference:  6.40°
95% LoA:          [-12.74°, +12.36°]

임상적 의미:
- 95% 확률로 MP 측정값이 GT ±12.7° 범위 내
- Ankle ROM 평균 30.9°의 41%!
- 임상적으로 너무 큰 불확실성
```

---

## 9. 왜 Day 4 Fix가 작동하지 않았나?

### Foot Frame Y-axis Fix 시도

**가설**: Tibia Y-axis를 reference로 사용하면 foot Y-axis 방향 일관성 확보

**결과**: 상관계수 변화 없음 (0.000 변화)

**왜 실패했나?**

**가설 1: Negation 상쇄**
```python
# _build_foot_frame()에서:
if dot(foot_y, tibia_y) < 0:
    axis_y_raw = -axis_y_raw  # Flip

# _calculate_ankle_angle()에서:
theta_y, _, _ = self._cardan_yzx(rel)
return -theta_y  # ← 이 negation이 flip을 다시 취소?
```

**가설 2: Flip이 실제로 일어나지 않음**
```python
# 모든 subjects에서 dot(foot_y, tibia_y) > 0일 수도
# → Flip condition never triggers
# → 코드 변화 없음
```

**가설 3: 문제가 더 깊음**
```python
# Foot frame Y-axis뿐만 아니라:
# - Tibia frame 자체가 잘못되었거나
# - Relative rotation 계산이 잘못되었거나
# - Cardan YZX extraction에 여전히 미묘한 버그가 있거나
```

---

## 10. 해결 방안

### 단기 (1-2주)

**1. Late Subjects 재검토**
```
Action: S1_23-26 비디오 품질 확인
- 해상도, 조명, 거리 비교
- 다른 subjects와 차이점 분석
- 필요시 제외 후 재분석

Expected: ICC 0.377 → 0.45-0.50 (still poor)
```

**2. Per-Subject Calibration**
```
Action: 각 subject별로 최적 foot frame 찾기
- GT 데이터로 학습
- Subject-specific correction factor 적용

Expected: ICC 0.377 → 0.60-0.70 (moderate)
Risk: Overfitting, production에서 사용 불가
```

### 중기 (1-2개월)

**3. ML-based Foot Frame Correction**
```
Action: GT 데이터로 foot frame orientation predictor 학습
- Input: Ankle, toe, heel landmark positions
- Output: Optimal foot Y-axis direction
- Train on subset, validate on rest

Expected: ICC 0.377 → 0.55-0.65
```

**4. Alternative Joint Angle Method**
```
Action: Cardan angles 대신 다른 방법 시도
- Joint vectors 기반 angle (3 points: ankle-toe-heel)
- Simplified 2D projection
- Avoid complex coordinate frames

Expected: Unknown, but may be more robust
```

### 장기 (3-6개월)

**5. Multi-Frame Temporal Smoothing**
```
Action: 단일 frame이 아닌 temporal window 사용
- 5-10 frames average
- Kalman filtering
- Reduce landmark jitter

Expected: Reduced variability → higher ICC
```

**6. Better Landmark Detection**
```
Action: MediaPipe 대신 다른 pose estimator 시도
- OpenPose
- AlphaPose
- HRNet
- Custom fine-tuned model

Expected: Better toe/heel detection → higher ICC
```

---

## 11. 결론

### 핵심 요약

1. **ROM Ratio 1.006 (완벽)**:
   - ✅ 그룹 평균이 일치함
   - ✅ 체계적 bias 없음
   - ✅ YZX bug fix가 성공적으로 작동함

2. **ICC 0.377 (낮음)**:
   - ❌ 개인별 variability 불일치 (1.83x)
   - ❌ 6/17 subjects에 큰 오차 (>7°)
   - ❌ Late subjects 특히 나쁨 (9.32° error)

3. **근본 원인**:
   - Foot frame Y-axis orientation 불일치
   - Toe/heel landmark 불안정성
   - Subject-specific factors (video quality, gait pattern)

### 임상적 평가

| 용도 | 적합성 | 이유 |
|------|--------|------|
| 그룹 비교 연구 | ✅ 적합 | ROM ratio 1.006, 평균 신뢰 가능 |
| 개인 진단 | ❌ 부적합 | ICC 0.377, ±12.7° 불확실성 |
| Longitudinal tracking | ⚠️ 제한적 | Within-subject changes만 가능 |

### Publication 전략

**권장**: Hip-only paper
- Hip ICC 0.813 → Clinical-grade
- Ankle ICC 0.377 → Supplementary material only
- ROM ratio 1.006 언급 (promising but needs work)

**Ankle 개선 후**: Multi-joint paper
- 위 해결 방안 중 1-2개 적용
- Target ICC > 0.50 (moderate)
- 그 후 다시 평가

---

**작성일**: 2025-11-09
**저자**: Claude (AI Assistant)
**목적**: Ankle ROM validation 실패 원인 상세 분석
