# 정면 영상 보행 분석 (Frontal View Gait Analysis)

## 개요

측면 영상 분석(V5 파이프라인)에 추가하여, 정면 영상(*-1.mp4)을 이용한 보행 분석 시스템을 개발하였습니다. 정면 영상에서는 측면에서 측정 불가능한 좌우 대칭성, 보폭 넓이, 골반 기울기 등을 분석할 수 있습니다.

**개발일**: 2025-10-11
**상태**: ✅ 파일럿 테스트 완료, 배치 처리 진행 중

---

## 시스템 구성

### 1. Pose Extraction ([extract_frontal_pose.py](extract_frontal_pose.py))

정면 영상에서 MediaPipe Pose를 이용하여 3D 랜드마크를 추출합니다.

**주요 기능:**
- MediaPipe Pose (model_complexity=2)를 사용한 고정밀 포즈 추정
- World coordinates (3D, meters, hip-centered)
- 13개 주요 랜드마크 추출:
  - 하체: 양측 고관절, 무릎, 발목, 발뒤꿈치, 발가락
  - 상체: 양측 어깨, 코 (체간 분석용)
- CSV 및 메타데이터 JSON 저장

**사용법:**
```bash
# 단일 영상 처리
python3 extract_frontal_pose.py --video data/1/1-1.mp4

# 배치 처리 (전체 대상자)
python3 extract_frontal_pose.py --batch --data-dir data

# 테스트 (대상자 1만)
python3 extract_frontal_pose.py --test
```

**출력 형식:**
- `data/X/X-1_front_pose_fpsYY.csv` - 3D 랜드마크 시계열 데이터
- `data/X/X-1_front_pose_fpsYY.json` - 메타데이터 (FPS, 프레임 수, 추출 날짜 등)

---

### 2. Frontal Gait Analysis ([frontal_gait_analyzer.py](frontal_gait_analyzer.py))

정면 영상 랜드마크로부터 보행 지표를 계산합니다.

#### 2.1 측정 지표

##### A. Step Width (보폭 넓이, 기저면)
- **정의**: 양발 발뒤꿈치 간 좌우 거리 (X축)
- **임상 의미**: 균형 능력 지표 (넓을수록 불안정성 증가)
- **정상 범위**: 5-15 cm
- **계산 방법**:
  - 좌우 heel strike 시점의 heel X 좌표 차이
  - ±0.5초 이내의 대응되는 반대측 heel strike와 매칭

##### B. Left-Right Symmetry (좌우 대칭성)
- **Step Count Ratio**: 좌우 스텝 수 비율
- **Step Count Symmetry**: `(1 - |ratio - 1.0|) × 100%`
- **Hip Height Correlation**: 양측 고관절 높이의 상관계수
- **임상 의미**: 편측 마비, 통증, 신경계 질환에서 비대칭 증가

##### C. Pelvic Obliquity (골반 기울기)
- **정의**: 양측 고관절 높이 차이로 계산한 골반 경사각
- **계산**: `arctan(Δy / Δx)` (양측 고관절 간)
- **정상 범위**: ±5도 이내
- **임상 의미**: 다리 길이 차이, 골반 변형, 척추측만증

##### D. Lateral Trunk Sway (체간 좌우 흔들림)
- **정의**: 체간 중심(양측 어깨 중점)의 좌우 변위 범위
- **측정**: Peak-to-peak amplitude (cm)
- **임상 의미**: 균형 제어 능력 (흔들림 클수록 균형 불량)

#### 2.2 Step Detection (정면 뷰)

측면과 달리 정면에서는 **수직 방향(Y축) 움직임**으로 heel strike 감지:
- Heel Y 좌표의 최저점 (가장 낮을 때 = 지면 접촉)
- Savitzky-Golay 필터로 스무딩 (window_size=11, polyorder=2)
- Peak detection (prominence=0.02, distance=0.5초)

#### 2.3 사용법

```bash
# 단일 대상자 분석
python3 frontal_gait_analyzer.py --csv data/1/1-1_front_pose_fps23.csv

# 배치 분석 (전체 대상자)
python3 frontal_gait_analyzer.py --batch --data-dir data

# 결과 저장
python3 frontal_gait_analyzer.py --batch \
  --output frontal_analysis_results.json \
  --report frontal_analysis_report.txt
```

---

## 파일럿 테스트 결과 (S1_01)

**영상 정보:**
- FPS: 23
- 총 프레임: 1334 (58초)
- 추출 성공률: 100%

**보행 지표:**

| 지표 | 측정값 | 단위 | 비고 |
|------|--------|------|------|
| **Step Width** | 5.9 ± 4.0 | cm | 정상 범위 (좁은 편) |
| **Step Count** | L: 48, R: 48 | - | - |
| **Step Symmetry** | 100.0 | % | 완벽한 대칭 |
| **Pelvic Obliquity** | 34.06 ± 77.37 | deg | ⚠️ 각도 큼 (방법론 검토 필요) |
| **Lateral Sway Range** | 6.72 | cm | 정상 |
| **Hip Height Correlation** | [계산됨] | - | - |

**평가:**
- ✅ Step detection 작동 (48 steps 검출)
- ✅ Step width 정상 범위
- ✅ 완벽한 좌우 대칭성
- ⚠️ Pelvic obliquity 각도가 과도하게 큼 → 계산 방식 재검토 필요
  - 현재: `arctan(Δy / Δx)` (양측 고관절)
  - 개선안: 절대값 사용, 또는 단순 높이 차이(cm) 보고

---

## 배치 처리 현황

**시작 시간**: 2025-10-11 13:08
**총 대상자**: 26명 (일부 대상자는 정면 영상 있음)
**상태**: 진행 중 (백그라운드)

**처리 과정:**
1. MediaPipe Pose Extraction: ~1분/대상자 (58초 영상 기준)
2. Frontal Gait Analysis: ~1초/대상자

**예상 완료 시간**: ~30분

---

## 측면 vs 정면 분석 비교

| 항목 | 측면 (V5 Pipeline) | 정면 (Frontal Analyzer) |
|------|-------------------|------------------------|
| **주요 지표** | Step Length, Cadence, Stride Time | Step Width, Symmetry, Pelvic Obliquity |
| **검증 상태** | ✅ 완료 (75% 개선) | 🔄 파일럿 테스트 중 |
| **Ground Truth** | 병원 데이터 있음 | ⚠️ 확인 필요 |
| **정확도** | RMSE 30.2cm, 0.93× detection | TBD |
| **임상 활용** | 보행 속도, 보폭 | 균형, 대칭성 |

---

## 다음 단계

### 즉시 (배치 완료 후)
1. ✅ **전체 대상자 정면 영상 추출** (진행 중)
2. ⬜ **전체 대상자 정면 보행 분석**
3. ⬜ **Pelvic obliquity 계산 방법 개선**
4. ⬜ **집단 통계 분석** (평균, 표준편차, 범위)
5. ✅ **측면(V5) 보폭·속도 계산에 턴 마스킹 공유** (2025-10-11 적용)

### 단기 (1-2일)
6. ⬜ **Ground Truth 비교**
   - 병원 데이터에 정면 측정값 있는지 확인
   - 있다면: ICC, RMSE 계산
   - 없다면: 문헌값과 비교 (정상 성인 기준치)
7. ⬜ **Multiview Integration**
   - 측면 + 정면 통합 분석 시스템
   - 2D/3D 궤적 시각화

### 중기 (1주)
8. ⬜ **추가 지표 개발**
   - Foot Progression Angle (발 진행 각도)
   - Step Length Asymmetry (정면에서 추정)
   - Dynamic Balance Index
9. ⬜ **병리적 보행 테스트**
   - 파킨슨병, 편마비 등 비정상 보행 패턴 검증

---

## 파일 구조

```
/data/gait/
├── extract_frontal_pose.py         # MediaPipe 정면 포즈 추출
├── frontal_gait_analyzer.py        # 정면 보행 분석 파이프라인
├── FRONTAL_ANALYSIS.md             # 본 문서
├── frontal_batch_extraction.log    # 배치 추출 로그
├── frontal_test_results.json       # 파일럿 결과 (S1_01)
├── frontal_test_report.txt         # 파일럿 요약
└── data/
    ├── 1/
    │   ├── 1-1.mp4                 # 정면 영상
    │   ├── 1-1_front_pose_fps23.csv   # 추출된 랜드마크
    │   └── 1-1_front_pose_fps23.json  # 메타데이터
    ├── 2/
    │   └── ...
    └── frontal_extraction_results.json  # 전체 추출 결과
```

---

## 기술적 세부사항

### MediaPipe 설정
- **Model Complexity**: 2 (highest accuracy)
- **Min Detection Confidence**: 0.7
- **Min Tracking Confidence**: 0.7
- **Output**: `pose_world_landmarks` (3D world coordinates, meters)

### Signal Processing
- **Smoothing**: Savitzky-Golay filter (window=11, polyorder=2)
- **Peak Detection**: scipy.signal.find_peaks
- **Normalization**: Z-score for template matching (not used in frontal yet)

### 좌표계
- **X축**: 좌우 (left-right), 오른쪽이 양수
- **Y축**: 상하 (vertical), 위쪽이 양수
- **Z축**: 전후 (depth), 카메라 방향이 양수

---

## 제한사항 및 개선 방향

### 현재 제한사항
1. **Pelvic Obliquity 과대 추정**
   - 현재 각도 계산이 과도하게 큼
   - 해결: 절대 높이 차이(cm)로 변경 고려

2. **Ground Truth 부재**
   - 정면 지표에 대한 병원 측정값 미확인
   - 해결: 문헌 기준치와 비교 또는 측정 의뢰

3. **Step Detection 정확도 미검증**
   - 정면 뷰 step detection이 GT와 일치하는지 미확인
   - 해결: 수동 labeling으로 검증 필요

4. **Depth 정보 미활용**
   - Z축(depth) 정보를 현재 사용하지 않음
   - 해결: Step length를 정면에서도 추정 가능

### 개선 방향
1. **멀티뷰 융합**
   - 측면(V5) + 정면을 결합한 3D 궤적 재구성
   - Epipolar geometry로 정밀도 향상

2. **딥러닝 기반 Step Detection**
   - 현재: Rule-based peak detection
   - 개선: LSTM/Transformer 기반 이벤트 감지

3. **실시간 처리**
   - 현재: Offline batch processing
   - 개선: Streaming pipeline for real-time feedback

---

## 참고 문헌

1. Bazarevsky, V., et al. (2020). "BlazePose: On-device Real-time Body Pose tracking." arXiv:2006.10204.
2. Lim, H., et al. (2020). "Validity of stance-phase gait parameters measured with a single inertial measurement unit." Journal of Biomechanics.
3. Chou, L.S., et al. (2003). "Medio-lateral motion of the center of mass during obstacle crossing distinguishes elderly individuals with imbalance." Gait & Posture, 18(3), 125-133.
4. Dingwell, J.B., et al. (2001). "Nonlinear time series analysis of normal and pathological human walking." Chaos, 10(4), 848-863.

---

**업데이트 히스토리:**
- 2025-10-11 13:10: 초안 작성 (파일럿 테스트 완료, 배치 진행 중)
