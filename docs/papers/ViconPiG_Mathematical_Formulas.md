# Vicon Plug-in Gait 각도 측정 공식: 수학적 상세

## Executive Summary

Vicon Plug-in Gait는 **Cardan (Euler) Angles**를 사용하여 관절 각도를 계산합니다.
정확한 회전 순서와 축 정의가 매우 중요합니다.

---

## Part 1: Cardan Angles 개념 (YXZ 회전 순서)

### 회전 행렬의 순차적 적용

```
일반 원칙:
R_total = R_Y(θ₁) × R_X'(θ₂) × R_Z''(θ₃)

where:
  R_Y(θ₁) = Y축 중심 회전 (각도 θ₁)
  R_X'(θ₂) = R_Y 회전으로 인해 이동한 X'축 중심 회전
  R_Z''(θ₃) = 두 번의 회전으로 이동한 Z''축 중심 회전

핵심: 각 회전 후 축이 새로운 위치로 이동 (X' 표기, X'' 표기)
```

### YXZ 회전 행렬 (Plug-in Gait 표준)

```
최종 회전 행렬:

R = [
  c₁c₃ - s₁s₂s₃     -c₂s₃      s₁c₃ + c₁s₂s₃  ]
    [
  c₁s₃ + s₁s₂c₃      c₂c₃      s₁s₃ - c₁s₂c₃  ]
    [
  -s₁c₂             s₂         c₁c₂            ]

where:
  c₁ = cos(θ₁), s₁ = sin(θ₁)  # Y축 회전
  c₂ = cos(θ₂), s₂ = sin(θ₂)  # X'축 회전
  c₃ = cos(θ₃), s₃ = sin(θ₃)  # Z''축 회전
```

### 역계산: 회전 행렬에서 각도 추출

```
R 행렬이 주어졌을 때, 세 개의 각도를 역으로 계산:

θ₁ (Y축 회전, 1번째):
  tan(θ₁) = R[0,2] / R[2,2]
  θ₁ = atan2(R[0,2], R[2,2])

θ₂ (X'축 회전, 2번째):
  sin(θ₂) = R[2,1]
  θ₂ = asin(R[2,1])
  또는: θ₂ = atan2(R[2,1], sqrt(R[2,0]² + R[2,2]²))

θ₃ (Z''축 회전, 3번째):
  tan(θ₃) = -R[1,0] / R[1,1]
  θ₃ = atan2(-R[1,0], R[1,1])

참고: atan2(y, x) = arctan(y/x) with quadrant adjustment
```

---

## Part 2: 관절별 계산 공식

### 1. 엉덩이 (Hip Angles) - YXZ 회전 순서

#### 좌표계 정의

```
Pelvis segment:
  Origin: 골반 중심
  Y_pelvis: RASI → LASI (오른쪽에서 왼쪽)
  Z_pelvis: 수직축 (골반 평면 수직)
  X_pelvis: 오른손 법칙으로 계산

Femur segment (대퇴골):
  Origin: Hip Joint Center (HJC)
  Y_femur: Knee direction (HJC → KJC, 아래쪽)
  X_femur: 전방 방향 (Femur와 정측 마커 정의)
  Z_femur: 오른손 법칙
```

#### 계산 과정

```
Step 1: 각 segment의 좌표계 결정
  - Pelvis frame: P_x, P_y, P_z 축
  - Femur frame: F_x, F_y, F_z 축

Step 2: Relative rotation matrix 계산
  R_rel = R_pelvis^T × R_femur
         = Pelvis 좌표계에서 Femur의 상대 방향

Step 3: R_rel에서 YXZ 각도 추출
  Hip_flexion = θ₁ (Y축 1번째 회전)
  Hip_adduction = θ₂ (X'축 2번째 회전)
  Hip_rotation = θ₃ (Z''축 3번째 회전)

예시 계산 (Python):
import numpy as np

# 1. Pelvis 벡터에서 좌표계 구성
rasi = np.array([...])  # Right ASIS marker
lasi = np.array([...])  # Left ASIS marker
sacr = np.array([...])  # Sacral marker

y_pelvis = (lasi - rasi) / np.linalg.norm(lasi - rasi)
temp = sacr - (rasi + lasi) / 2
z_pelvis = np.cross(y_pelvis, temp)
z_pelvis = z_pelvis / np.linalg.norm(z_pelvis)
x_pelvis = np.cross(y_pelvis, z_pelvis)

# 2. Femur 벡터에서 좌표계 구성
hips_center = (rasi + lasi) / 2 + offset  # offset 계산 필요
knee_center = (derived from KNE marker)
y_femur = (knee_center - hips_center) / norm
# ... x_femur, z_femur 유사하게

# 3. 회전 행렬 계산
R_pelvis = np.column_stack([x_pelvis, y_pelvis, z_pelvis])
R_femur = np.column_stack([x_femur, y_femur, z_femur])
R_rel = R_pelvis.T @ R_femur

# 4. YXZ 각도 추출
theta1 = np.arctan2(R_rel[0,2], R_rel[2,2])  # Flexion
theta2 = np.arcsin(R_rel[2,1])                # Adduction
theta3 = np.arctan2(-R_rel[1,0], R_rel[1,1]) # Rotation

hip_flexion = np.degrees(theta1)
hip_adduction = np.degrees(theta2)
hip_rotation = np.degrees(theta3)
```

#### 부호 규칙 (Left/Right 차이)

```
Left Hip:
  Flexion +: 앞으로 굴곡
  Adduction +: 안쪽 (가운데로)
  Rotation +: 안쪽 회전

Right Hip:
  Flexion +: 앞으로 굴곡
  Adduction +: 바깥쪽 (반대)  ← LEFT와 다름!
  Rotation +: 바깥쪽 회전 ← LEFT와 다름!
```

### 2. 무릎 (Knee Angles) - YXZ 회전 순서

#### 좌표계 정의

```
Femur segment (Thigh):
  Origin: Hip Joint Center (HJC)
  (위에서 정의한 것과 동일)

Tibia segment (Shank):
  Origin: Knee Joint Center (KJC)
  Y_tibia: Ankle direction (KJC → AJC, 아래쪽)
  X_tibia: 전방 방향 (Tibia와 정측 마커 정의)
  Z_tibia: 오른손 법칙
  
중요: Tibia의 X축은 Femur의 X축과 일직선상에 있어야 함
      (flexion axis alignment)
```

#### 계산 과정

```
Step 1: 각 segment의 좌표계 결정
  - Femur frame: F_x, F_y, F_z
  - Tibia frame: T_x, T_y, T_z

Step 2: Relative rotation matrix
  R_rel = R_femur^T × R_tibia
        = Femur 좌표계에서 Tibia의 상대 방향

Step 3: R_rel에서 YXZ 각도 추출
  Knee_flexion = θ₁ (Y축 회전, Sagittal)
  Knee_varus = θ₂ (X'축 회전, Frontal)
  Knee_rotation = θ₃ (Z''축 회전, Transverse)

공식:
theta1 = np.arctan2(R_rel[0,2], R_rel[2,2])  # Flexion
theta2 = np.arcsin(R_rel[2,1])                # Varus/Valgus
theta3 = np.arctan2(-R_rel[1,0], R_rel[1,1]) # Rotation
```

#### Vicon Convention: 180도 변환

```
Plug-in Gait에서 사용:
  KneeFlexion_output = 180° - θ₁

이유:
  - 0°: 완전 신전 (straight)
  - 90°: 직각
  - 180°: 완전 굴곡 (이론적)

따라서:
  θ₁ = 0° → KneeFlexion = 180°  (완전 신전)
  θ₁ = 90° → KneeFlexion = 90° (직각)
  θ₁ = 180° → KneeFlexion = 0°  (완전 굴곡)

실제로는 θ₁이 항상 양수이므로:
  KneeFlexion = 180° - arctan2(...) = 180° - θ₁
```

### 3. 발목 (Ankle Angles) - **YZX 회전 순서** (특별함!)

#### 좌표계 정의

```
Tibia segment (Shank):
  (위에서 정의한 것과 동일)

Foot segment:
  Origin: Ankle Joint Center (AJC)
  Y_foot: Toe direction (AJC → TOE)
  X_foot: 외측 방향
  Z_foot: 오른손 법칙
```

#### 계산 과정 (YZX - 다른 순서!)

```
Step 1: 각 segment의 좌표계 결정
  - Tibia frame: T_x, T_y, T_z
  - Foot frame: F_x, F_y, F_z

Step 2: Relative rotation matrix
  R_rel = R_tibia^T × R_foot

Step 3: R_rel에서 YZX 각도 추출 (YXZ와 다름!)
  θ₁ (Y축): Dorsiflexion/Plantarflexion
  θ₂' (Z축, 2번째): Inversion/Eversion
  θ₃'' (X축, 3번째): Internal/External Rotation

YZX 역계산 공식:
theta1 = np.arctan2(R_rel[0,2], R_rel[2,2])    # Dorsiflexion
theta2 = np.arcsin(-R_rel[0,1])                # Inversion (Y와 Z의 중간)
theta3 = np.arctan2(R_rel[1,0], R_rel[1,1])    # Rotation

발목 출력:
  Dorsiflexion = θ₁ (양수 = dorsiflexion, 음수 = plantarflexion)
  Inversion = θ₂'
  Rotation = θ₃''
```

---

## Part 3: Joint Center (JC) 계산 - CHORD 함수

### 무릎 관절 중심 (KJC) 계산

```
목표: 마커(KNE)로부터 실제 관절 중심(KJC) 위치 결정

공식 (CHORD function):
1. Offset 거리 계산:
   KneeOffset = (MarkerDiameter + KneeWidth) / 2
   
   where:
     MarkerDiameter ≈ 14mm (표준 마커)
     KneeWidth = 대상자 측정값 (정상: 6-10cm)

2. JC 위치 결정:
   - KNE 마커 위치: P_kne
   - THI 마커 위치: P_thi (허벅지 외측)
   - HJC (엉덩이 관절 중심): P_hjc
   
   평면 정의: KNE, THI, HJC 3점
   법선 벡터: N = (P_thi - P_kne) × (P_hjc - P_kne)
   
   KJC = P_kne + KneeOffset × (N / |N|)
   
   제약조건: angle(KNE, KJC, HJC) = 90°

코드 예시:
import numpy as np

def calculate_knee_jc(p_kne, p_thi, p_hjc, knee_width=0.08):
    """
    Calculate knee joint center from marker
    
    Args:
        p_kne: KNE marker position
        p_thi: THI marker position
        p_hjc: Hip joint center
        knee_width: measured subject knee width (meters)
    
    Returns:
        p_kjc: Knee joint center
    """
    marker_diameter = 0.014  # 14mm
    knee_offset = (marker_diameter + knee_width) / 2
    
    # 평면 법선
    v1 = p_thi - p_kne
    v2 = p_hjc - p_kne
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # KJC = KNE + offset × normal
    p_kjc = p_kne + knee_offset * normal
    
    # 검증: angle(KNE, KJC, HJC) ≈ 90°
    v_kne_kjc = p_kjc - p_kne
    v_kne_hjc = p_hjc - p_kne
    cos_angle = np.dot(v_kne_kjc, v_kne_hjc) / (np.linalg.norm(v_kne_kjc) * np.linalg.norm(v_kne_hjc))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return p_kjc, np.degrees(angle)  # 각도는 90°에 가까워야 함
```

### 발목 관절 중심 (AJC) 계산

```
동일한 원리:

AnkleOffset = (MarkerDiameter + AnkleWidth) / 2

AJC = P_ank + AnkleOffset × (N / |N|)

where N = (P_tib - P_ank) × (P_kjc - P_ank)
```

---

## Part 4: 당신의 MP 코드와의 비교

### 발목 계산 - 현재 MP vs 올바른 Plug-in Gait

```
현재 MP 코드 (추정):
  ankle_angle = angle - 90  # 단순 90도 뺄셈

문제점:
  1. 어떤 'angle'을 사용하는가?
  2. YZX 회전 순서를 사용하는가?
  3. AJC를 올바르게 계산하는가?

올바른 방법:
  1. AJC 계산 (위 공식)
  2. Shank & Foot 좌표계 구성
  3. Relative rotation matrix 계산
  4. YZX 각도 추출
  5. Dorsiflexion = θ₁ (음수 = plantarflexion)
```

### 무릎 계산 - 현재 MP vs 올바른 Plug-in Gait

```
현재 MP 코드:
  knee_flexion = 180 - angle

문제점:
  1. 'angle'이 올바른 YXZ 1번째 각도인가?
  2. 아니면 다른 계산인가?

올바른 방법:
  1. KJC 계산 (위 공식)
  2. Femur & Tibia 좌표계 구성
  3. YXZ 회전 순서로 각도 추출
  4. KneeFlexion = 180 - θ₁

만약 현재 'angle'이 이미 올바른 θ₁이면:
  knee_flexion = 180 - θ₁ (정확함)

하지만 당신의 ROM이 축소되어 있다면:
  'angle' 계산 자체가 잘못되었을 가능성
```

---

## Part 5: Python 구현 예제

### 완전한 발목 각도 계산 함수

```python
import numpy as np

def calculate_ankle_angles_cardan_yzx(p_tibia, p_ankle, p_kjc, p_toe, 
                                     ankle_width=0.065):
    """
    Calculate ankle angles using Cardan YZX rotation order
    (Vicon Plug-in Gait standard)
    
    Args:
        p_tibia: Tibia (TIB) marker position
        p_ankle: Ankle (ANK) marker position
        p_kjc: Knee joint center
        p_toe: Toe (TOE) marker position
        ankle_width: measured ankle width (meters)
    
    Returns:
        dorsiflexion: positive = dorsiflexion, negative = plantarflexion (degrees)
        inversion: positive = inversion
        rotation: positive = internal rotation
    """
    
    marker_diameter = 0.014
    ankle_offset = (marker_diameter + ankle_width) / 2
    
    # Step 1: Calculate AJC (Ankle Joint Center)
    v1 = p_tibia - p_ankle
    v2 = p_kjc - p_ankle
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    p_ajc = p_ankle + ankle_offset * normal
    
    # Step 2: Construct Tibia coordinate frame
    y_tibia = (p_kjc - p_ajc) / np.linalg.norm(p_kjc - p_ajc)  # toward knee
    x_tibia_temp = p_tibia - p_ankle
    z_tibia = np.cross(y_tibia, x_tibia_temp)
    z_tibia = z_tibia / np.linalg.norm(z_tibia)
    x_tibia = np.cross(y_tibia, z_tibia)
    
    R_tibia = np.column_stack([x_tibia, y_tibia, z_tibia])
    
    # Step 3: Construct Foot coordinate frame
    y_foot = (p_toe - p_ajc) / np.linalg.norm(p_toe - p_ajc)  # toward toe
    x_foot_temp = np.array([1, 0, 0])  # 또는 다른 정의
    z_foot = np.cross(y_foot, x_foot_temp)
    z_foot = z_foot / np.linalg.norm(z_foot)
    x_foot = np.cross(y_foot, z_foot)
    
    R_foot = np.column_stack([x_foot, y_foot, z_foot])
    
    # Step 4: Calculate relative rotation (Tibia -> Foot)
    R_rel = R_tibia.T @ R_foot
    
    # Step 5: Extract YZX angles from relative rotation matrix
    # YZX 순서:
    theta1_rad = np.arctan2(R_rel[0, 2], R_rel[2, 2])  # Y축 (Dorsiflexion)
    theta2_rad = np.arcsin(-R_rel[0, 1])                # Z축 (Inversion)
    theta3_rad = np.arctan2(R_rel[1, 0], R_rel[1, 1])  # X축 (Rotation)
    
    # Convert to degrees
    dorsiflexion = np.degrees(theta1_rad)
    inversion = np.degrees(theta2_rad)
    rotation = np.degrees(theta3_rad)
    
    return dorsiflexion, inversion, rotation, p_ajc
```

### 완전한 무릎 각도 계산 함수

```python
def calculate_knee_angles_cardan_yxz(p_femur, p_hjc, p_kjc, p_tibia,
                                    knee_width=0.08):
    """
    Calculate knee angles using Cardan YXZ rotation order
    (Vicon Plug-in Gait standard)
    
    Args:
        p_femur: Femur (THI) marker
        p_hjc: Hip joint center
        p_kjc: Knee joint center
        p_tibia: Tibia (TIB) marker
        knee_width: measured knee width
    
    Returns:
        knee_flexion: positive = flexion (180° at full extension)
        knee_varus: positive = varus
        knee_rotation: positive = internal rotation
    """
    
    # Step 1: Construct Femur (Thigh) frame
    y_femur = (p_kjc - p_hjc) / np.linalg.norm(p_kjc - p_hjc)
    x_femur_temp = p_femur - p_hjc
    z_femur = np.cross(y_femur, x_femur_temp)
    z_femur = z_femur / np.linalg.norm(z_femur)
    x_femur = np.cross(y_femur, z_femur)
    
    R_femur = np.column_stack([x_femur, y_femur, z_femur])
    
    # Step 2: Construct Tibia frame
    y_tibia = (p_kjc - p_tibia) / np.linalg.norm(p_kjc - p_tibia)  # toward knee
    x_tibia_temp = p_tibia
    z_tibia = np.cross(y_tibia, x_tibia_temp)
    z_tibia = z_tibia / np.linalg.norm(z_tibia)
    x_tibia = np.cross(y_tibia, z_tibia)
    
    R_tibia = np.column_stack([x_tibia, y_tibia, z_tibia])
    
    # Step 3: Calculate relative rotation
    R_rel = R_femur.T @ R_tibia
    
    # Step 4: Extract YXZ angles
    theta1_rad = np.arctan2(R_rel[0, 2], R_rel[2, 2])  # Y축 (Flexion)
    theta2_rad = np.arcsin(R_rel[2, 1])                 # X'축 (Varus)
    theta3_rad = np.arctan2(-R_rel[1, 0], R_rel[1, 1]) # Z''축 (Rotation)
    
    # Convert to degrees
    theta1_deg = np.degrees(theta1_rad)
    theta2_deg = np.degrees(theta2_rad)
    theta3_deg = np.degrees(theta3_rad)
    
    # Vicon convention: 180° - flexion
    knee_flexion = 180 - theta1_deg
    knee_varus = theta2_deg
    knee_rotation = theta3_deg
    
    return knee_flexion, knee_varus, knee_rotation
```

---

## Part 6: 당신의 데이터로 검증

### 진단 코드

```python
# 당신의 MP 좌표와 GT 좌표를 비교

for frame_idx in range(num_frames):
    # MP 데이터
    mp_ankle_mp_method = calculate_ankle_angles_cardan_yzx(
        mp_tibia[frame_idx],
        mp_ankle[frame_idx],
        mp_kjc[frame_idx],
        mp_toe[frame_idx]
    )[0]  # dorsiflexion만
    
    # GT 데이터
    gt_ankle = gt_angles[frame_idx]['ankle']
    
    # 비교
    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}:")
        print(f"  MP (Cardan YZX): {mp_ankle_mp_method:.1f}°")
        print(f"  GT: {gt_ankle:.1f}°")

# 전체 상관계수
correlation = np.corrcoef(mp_ankle_mp_method_series, gt_ankle_series)[0, 1]
print(f"\n상관계수: {correlation:.3f}")
print(f"현재 MP 상관계수: 0.58-0.64")
print(f"개선됨? {correlation > 0.65}")
```

---

## 결론

Vicon Plug-in Gait의 각도 계산은:

1. **Cardan (Euler) Angles** 사용
2. **회전 순서가 매우 중요**
   - Hip/Knee: YXZ
   - Ankle: YZX (다름!)
3. **Joint Center(JC) 정확한 계산 필수**
4. **부호 규칙은 Left/Right 다름**

당신의 MP 코드가 이 모든 요소를 정확히 따르는지 확인해야 합니다.

특히:
- 발목 YZX vs 무릎 YXZ 구분
- Joint Center offset 계산
- Rotation matrix 올바른 구성

위 Python 예제를 실행하면 당신의 MP 결과가 올바른지 검증할 수 있을 것입니다.

