# 각도 방향 불일치 문제 보고서

## traditional_normals.csv 실행 결과

✅ 올바른 GT 소스 사용: `processed/S1_XX_traditional_normals.csv`
✅ 컬럼: `normal_average__y` (절대 각도 정상 범위)

---

## 결과

### Deming 회귀 파라미터

| 관절 | Slope | 95% CI | Intercept | 상관계수 | 상태 |
|------|-------|--------|-----------|---------|------|
| **발목** | -5.15 | [-5.82, -4.49] | 0.000° | **-0.69** | ❌ 음수 기울기! |
| **무릎** | 3.95 | [3.65, 4.26] | 0.003° | **+0.82** | ✅ 양의 상관 |
| **엉덩이** | -0.43 | [-0.44, -0.41] | 0.000° | **-0.92** | ❌ 음수 기울기! |

---

## 🚨 핵심 문제: 각도 방향 불일치

### 상관계수 분석 (Centered Curves)

**음의 상관 = MP가 증가할 때 GT가 감소**

```
발목(ANKLE):   상관계수 = -0.691  ❌
무릎(KNEE):    상관계수 = +0.820  ✅
엉덩이(HIP):   상관계수 = -0.919  ❌
```

### Subject 1 예시

#### ANKLE (음의 상관 -0.69)
```
Gait Cycle:     0%    1%    2%    3%    4%    5%    6%    7%    8%    9%
MP (centered):  7.7   8.5   9.4  10.1  11.1  11.2  10.6   9.5   8.3   7.2
GT (centered): -2.1  -2.0  -2.2  -2.5  -3.2  -4.2  -5.1  -6.1  -6.7  -7.0
→ MP가 증가할 때 GT가 감소!
```

#### KNEE (양의 상관 +0.82)
```
Gait Cycle:      0%     1%     2%     3%     4%     5%     6%     7%     8%     9%
MP (centered):  -6.8   -5.1   -2.6   -0.1    2.2    3.6    4.8    6.0    6.5    7.0
GT (centered): -23.6  -22.7  -21.6  -20.2  -18.7  -17.2  -15.8  -14.5  -13.2  -11.9
→ 둘 다 같은 방향으로 변화! (스케일만 다름)
```

#### HIP (강한 음의 상관 -0.92)
```
Gait Cycle:      0%     1%     2%     3%     4%     5%     6%     7%     8%     9%
MP (centered): -37.0  -38.5  -35.0  -32.0  -26.4  -20.4  -17.1  -16.7  -17.4  -16.7
GT (centered):  13.0   13.1   13.3   13.4   13.5   13.6   13.6   13.5   13.3   13.1
→ MP가 감소할 때 GT가 증가!
```

---

## 원인 분석

### 가설 1: MediaPipe 각도 계산 Convention 문제

**MediaPipe 각도 정의**:
- 3D 랜드마크 벡터로부터 각도 계산
- 특정 해부학적 convention 사용 (예: 굴곡=양수, 신전=음수)

**병원 GT 각도 정의**:
- 전통적 gait lab 시스템
- 다른 해부학적 convention 가능 (예: 굴곡=음수, 신전=양수)

**증거**:
- 무릎만 방향이 일치 (+0.82 상관)
- 발목과 엉덩이는 방향이 반대 (-0.69, -0.92)

### 가설 2: GT normal_average__y 컬럼이 실제로는 다른 값

**가능성**:
- normal_average__y가 "정상으로부터의 편차"일 수도
- 또는 특정 정규화된 표현

**반증**:
- 범위를 보면 절대 각도처럼 보임:
  - ANKLE: -11.6° to 15.1°
  - KNEE: 2.4° to 63.6°
  - HIP: -4.6° to 34.2°

---

## 해결 방안

### Option 1: MP 각도 부호 반전 (빠름, 추천)

**발목과 엉덩이 MP 각도에 -1 곱하기**:

```python
# 각도 로딩 후
if joint in ['ankle_dorsi_plantarflexion', 'hip_flexion_extension']:
    mp_curve = -mp_curve  # 부호 반전
```

**장점**:
- 빠른 검증 (5분)
- MP와 GT convention 일치시킴
- 무릎은 이미 방향이 맞음

**단점**:
- MP 각도 계산이 잘못되었다면 근본 해결이 아님

### Option 2: GT 다른 컬럼 시도

**normal_average__x 또는 normal_average__z 시도**:
- 이미 시도했지만 스케일이 더 작았음
- 방향만 반대일 가능성

### Option 3: MP 각도 계산 로직 재검토 (근본적)

**MediaPipe 3D → 각도 변환 확인**:
- `mediapipe_sagittal_extractor.py` 또는 각도 계산 스크립트
- 해부학적 convention 확인
- 벡터 방향 검증

**장점**:
- 근본 원인 해결
- 올바른 해부학적 의미 보장

**단점**:
- 시간 소요 (1-2시간)
- MP 로직이 다른 곳에서도 사용 중이면 영향 범위 큼

---

## 권장 조치

### 1단계: MP 부호 반전 빠른 테스트 (5분)

```python
# improve_calibration_deming.py의 stack_joint_arrays에 추가
mp_curve_trimmed = mp_curve[:length]

# 발목과 엉덩이는 방향 반전
if joint in ['ankle_dorsi_plantarflexion', 'hip_flexion_extension']:
    mp_curve_trimmed = -mp_curve_trimmed

# 중심화
mp_curve_centered = mp_curve_trimmed - mp_curve_trimmed.mean()
```

**예상 결과**:
- ANKLE slope: 0.8-1.2 범위 (현재 -5.15)
- HIP slope: 0.8-1.2 범위 (현재 -0.43)
- KNEE slope: 0.8-1.2 범위 유지 (현재 3.95는 여전히 문제)

### 2단계: 무릎 스케일 조정

무릎은 방향은 맞지만 스케일이 4배 차이:
- MP std: 10.1°
- GT std: 18.6°
- 비율: ~0.5x

**원인 가능성**:
- MP가 무릎 ROM을 과소 추정
- GT가 더 큰 변동성을 포착

**조치**:
- Deming slope 3.95 그대로 사용 (MP × 3.95 = GT scale)
- 또는 MP 무릎 각도 계산 재검토

### 3단계: 검증

```
예상 결과:
- ANKLE: slope 0.8-1.2, corr +0.7~+0.9
- KNEE: slope 3.5-4.5, corr +0.8
- HIP: slope 0.8-1.2, corr +0.8~+0.9
```

모든 관절이 양의 상관, 발목/엉덩이 slope ~1.0 달성 시 **Phase 1 Day 4-5 진행 가능**.

---

## 현재 상태

**Phase 1 Day 1-3 블로커**:
- ✅ Deming 회귀 구현 완료
- ✅ 올바른 GT 소스 확인 (traditional_normals.csv)
- ✅ 무릎 방향 일치 확인 (상관 +0.82)
- ❌ 발목/엉덩이 각도 방향 반대 (상관 -0.69, -0.92)
- 🔧 **MP 부호 반전 테스트 필요**

**다음 단계**:
1. MP 발목/엉덩이 부호 반전 추가
2. Deming 회귀 재실행
3. 3개 관절 모두 양의 상관 + slope 확인
4. 성공 시 Phase 1 Day 4-5 (필터링) 진행

---

**보고서 생성일**: 2025-11-07
**상태**: 각도 방향 불일치 발견, MP 부호 반전 테스트 대기 중
