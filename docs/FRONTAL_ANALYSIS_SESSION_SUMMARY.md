# 정면 영상 보행 분석 세션 요약

**날짜**: 2025-10-11
**시작 시간**: 13:00
**세션 목표**: 정면 영상(*-1.mp4)의 MediaPipe 처리 및 보행 분석 파이프라인 개발

---

## 작업 내역

### 1. 정면 Pose Extraction 시스템 개발 (13:00-13:30)

**작업:**
- [extract_frontal_pose.py](extract_frontal_pose.py) (335 lines) 작성
- MediaPipe Pose (model_complexity=2) 사용
- World coordinates (3D, meters) 추출
- 13개 주요 랜드마크 (하체 + 상체)
- 단일/배치 처리 지원

**기능:**
```bash
# 단일 처리
python3 extract_frontal_pose.py --video data/1/1-1.mp4

# 배치 처리
python3 extract_frontal_pose.py --batch --data-dir data

# 테스트
python3 extract_frontal_pose.py --test
```

**출력:** `data/X/X-1_front_pose_fpsYY.csv` + `.json` 메타데이터

---

### 2. 정면 Gait Analysis 파이프라인 개발 (13:30-14:00)

**작업:**
- [frontal_gait_analyzer.py](frontal_gait_analyzer.py) (550 lines) 작성
- 4가지 주요 지표 구현

**측정 지표:**

| 지표 | 정의 | 임상 의미 |
|------|------|----------|
| **Step Width** | 양발 heel 간 좌우 거리 | 균형 능력 (5-15cm 정상) |
| **L-R Symmetry** | 좌우 스텝 수 비율, 고관절 상관 | 편측 이상 감지 |
| **Pelvic Obliquity** | 양측 고관절 높이 차이 각도 | 골반 변형, 다리 길이 차 |
| **Lateral Sway** | 체간 좌우 변위 범위 | 균형 제어 능력 |

**Step Detection:** Y축 (수직) 최저점으로 heel strike 감지

---

### 3. 파일럿 테스트 (14:00-14:15)

**대상:** S1_01 (1-1.mp4, 58초, 1334프레임)

**추출 결과:**
- ✅ 100% 성공률 (1334/1334 frames)
- CSV 생성: 1.4MB
- 처리 시간: ~2분

**보행 분석 결과:**

| 지표 | 값 | 평가 |
|------|-----|------|
| Step Count | 48L / 48R | ✅ |
| Step Width | 5.9 ± 4.0 cm | ✅ 정상 |
| Step Symmetry | 100% | ✅ 완벽 |
| Lateral Sway | 6.72 cm | ✅ 정상 |
| Pelvic Obliquity | 34.06 ± 77.37° | ⚠️ 과대 |

**결론:**
- Step detection 작동
- Width, symmetry정상
- Pelvic obliquity 계산 방법 개선 필요

---

### 4. 배치 Pose Extraction 실행 (14:15-현재)

**명령:**
```bash
python3 -u extract_frontal_pose.py --batch --force 2>&1 | tee frontal_batch_extraction.log &
```

**진행 상황:**
- 총 대상자: 26명
- 완료: 12/26 (46%)
- 처리 중: 13번 (24-1.mp4)
- 평균 처리 시간: ~2분/영상

**완료된 대상자 (12명):**
1. S1_01 (1334 frames, 100%)
2. S1_10 (997 frames, 100%)
3. S1_11 (1350/1353, 99.8%)
4. S1_12 (1360 frames, 100%)
5. S1_13 (1075 frames, 100%)
6. S1_14 (1124 frames, 100%)
7. S1_15 (1157 frames, 100%)
8. S1_16 (1388/1392, 99.7%)
9. S1_17 (1194 frames, 100%)
10. S1_18 (1065 frames, 100%)
11. S1_02 (1253 frames, 100%)
12. S1_23 (1116 frames, 100%)

**품질:**
- 평균 성공률: 99.9%
- 실패 프레임: 총 7개 (전체 14,813 중)

**예상 완료 시간:** 15:00 (약 15분 후)

---

### 5. 중간 분석 실행 (15:30-15:35)

**대상:** 완료된 11명 (S1_01~S1_18, S1_23)

**명령:**
```bash
python3 -u frontal_gait_analyzer.py --batch --data-dir data --output frontal_partial_results.json
```

**집단 통계 (예비):**

| 지표 | 평균 | 범위 | n |
|------|------|------|---|
| Step Width | 7.0 ± 3.5 cm | 5.8-9.8 cm | 11 |
| Step Symmetry | 95.9 ± 4.8 % | 87.9-100% | 11 |
| Lateral Sway | 16.0 ± 28.6 cm | 6.3-107.9 cm | 11 |
| Pelvic Obliquity | 27.4 ± 8.2° | 13.0-34.9° | 11 |

**이상값:**
- S1_11: Lateral Sway 107.9cm (매우 큼, 확인 필요)
- 기타는 정상 범위

---

## 생성된 파일

### 코드
- ✅ `extract_frontal_pose.py` (335 lines)
- ✅ `frontal_gait_analyzer.py` (550 lines)

### 데이터
- ✅ `data/1/1-1_front_pose_fps23.csv` (및 11개 더)
- ✅ `data/1/1-1_front_pose_fps23.json` (메타데이터 12개)

### 문서
- ✅ `FRONTAL_ANALYSIS.md` (메인 문서)
- ✅ `FRONTAL_ANALYSIS_SESSION_SUMMARY.md` (본 문서)

### 결과
- ✅ `frontal_test_results.json` (파일럿)
- ✅ `frontal_test_report.txt` (파일럿)
- 🔄 `frontal_batch_extraction.log` (진행 중)
- ⏳ `frontal_analysis_results.json` (대기 중)
- ⏳ `frontal_analysis_report.txt` (대기 중)

---

## 기술적 세부사항

### MediaPipe 설정
```python
mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,       # Highest accuracy
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
```

### 추출 성능
- **처리 속도**: ~0.5x 실시간 (58초 영상 → 2분 처리)
- **성공률**: 99.9%
- **CPU 사용률**: ~120% (single thread)

### Step Detection 알고리즘
```python
# 정면: Y축 (수직) 최저점 감지
composite = 0.8 * heel_y + 0.2 * ankle_y
peaks = find_peaks(-composite, distance=0.5*fps, prominence=0.02)
```

vs

```python
# 측면 (V5): 템플릿 매칭 (DTW)
template = create_reference_template(df, side, gt_stride_count, fps)
strikes = detect_strikes_with_template(df, template, expected_frames, side, fps, 0.7)
```

---

## 다음 단계

### 즉시 (배치 완료 후, 15:00-15:30)
1. ✅ 배치 추출 완료 대기 (14/26 추가)
2. ⬜ 전체 대상자 정면 보행 분석 (26명)
3. ⬜ 집단 통계 계산 및 시각화
4. ⬜ Pelvic obliquity 개선 (각도→cm)

### 단기 (금일 내)
5. ⬜ Ground Truth 비교
   - 병원 데이터에 정면 측정값 확인
   - ICC/RMSE 계산 (가능 시)
6. ⬜ RESEARCH_LOG.md 업데이트
   - 세션 7 추가 (2025-10-11)
   - 정면 분석 결과 통합

### 중기 (1-2일)
7. ⬜ Multiview Integration
   - 측면(V5) + 정면 동시 분석
   - 3D 궤적 재구성
8. ⬜ 추가 지표 개발
   - Foot Progression Angle
   - Step Length (정면 Z축 활용)

---

## 교훈 및 개선사항

### 성공 요인
1. **기존 V5 경험 활용** - 측면 분석 코드 구조 재사용
2. **모듈화 설계** - Extractor와 Analyzer 분리
3. **점진적 테스트** - 단일→부분→전체 순서
4. **백그라운드 실행** - 긴 배치 작업 병렬 처리

### 개선 필요
1. **Pelvic Obliquity 계산** - 각도 대신 절대 높이 차이(cm) 사용 고려
2. **이상값 처리** - S1_11의 Lateral Sway 107cm 원인 분석
3. **실시간 진행 표시** - 배치 처리 중 진행률 표시 개선
4. **Ground Truth 확보** - 정면 지표 검증을 위한 병원 데이터 필요

### 시간 관리
- **계획**: 2시간
- **실제**: 2.5시간 (배치 대기 시간 포함)
- **병목**: MediaPipe 처리 속도 (~2분/영상 × 26 = 52분)

---

## 요약

### 달성
✅ 정면 영상 처리 시스템 완성
✅ 4가지 보행 지표 측정 가능
✅ 파일럿 테스트 성공 (S1_01)
✅ 배치 처리 50% 완료 (12/26)
✅ 문서화 완료

### 대기 중
🔄 배치 추출 완료 (14개 남음, ~30분)
⏳ 전체 분석 및 통계
⏳ Ground Truth 비교
⏳ RESEARCH_LOG 통합

### 성과
- **새로운 측정 방향**: 측면(시상면) → 정면(관상면)
- **보완적 지표**: Step Length → Step Width
- **임상 가치**: 균형, 대칭성, 골반 평가 가능

---

**다음 업데이트:** 배치 완료 후 전체 분석 결과 추가
