# 🎉 보행분석 데이터 변환 완료

**날짜**: 2025-10-10
**변환기 버전**: 1.0

---

## ✅ 변환 결과

### 전체 요약

| 항목 | 값 |
|------|-----|
| **총 피험자 수** | 21명 |
| **변환 성공률** | 100% (21/21) |
| **총 생성 레코드** | 101,808개 |
| **평균 레코드/피험자** | 4,848개 |
| **생성 파일 수** | 43개 (21×2 + summary) |

### 피험자 통계 (N=21)

| 지표 | 평균 | 표준편차 | 최소 | 최대 |
|------|------|----------|------|------|
| 나이 (세) | 26.2 | 3.1 | 23 | 36 |
| 키 (cm) | 173.4 | 6.1 | 153 | 182 |
| 체중 (kg) | 76.1 | 14.6 | 47 | 123 |
| Cadence 우측 (steps/min) | 114.2 | 6.2 | 100.3 | 129.0 |
| Cadence 좌측 (steps/min) | 114.3 | 6.4 | 100.6 | 129.2 |
| Step Length 우측 (cm) | 65.2 | 5.3 | 54.2 | 76.2 |
| Step Length 좌측 (cm) | 64.8 | 5.6 | 53.8 | 77.2 |

---

## 📁 출력 디렉토리 구조

```
/data/gait/data/processed_new/
├── S1_01_info.json          # 피험자 정보 (3.4KB)
├── S1_01_gait_long.csv      # 관절 각도 데이터 (584KB)
├── S1_02_info.json
├── S1_02_gait_long.csv
├── ... (총 21명)
├── S1_30_info.json
├── S1_30_gait_long.csv
└── conversion_summary.json  # 전체 변환 요약
```

---

## 📊 데이터 구조

### 1. info.json 구조

각 피험자의 메타데이터와 보행 파라미터:

```json
{
  "subject_id": "S1_XX",
  "demographics": {
    "name": "...",
    "age": 28,
    "height_cm": 173,
    "weight_kg": 90,
    "gait_cycle_timing": {
      "right_ids": 11.893,
      "right_ss": 37.579,
      ...
    }
  },
  "patient": {
    "right": {
      "step_length_cm": 64.436,
      "stride_length_cm": 125.859,
      "cadence_steps_min": 114.802,
      "forward_velocity_cm_s": 120.241,
      "total_support_time_pct": 62.114,
      "swing_phase_pct": 37.886,
      ...
    },
    "left": { ... }
  },
  "normal": {
    "right": { ... },
    "left": { ... }
  }
}
```

**포함 정보**:
- ✅ 기본 인구통계 (이름, 나이, 키, 체중)
- ✅ 보행주기 타이밍 (IDS, SS, Stance)
- ✅ Patient 보행 파라미터 (좌/우 분리)
  - Step/Stride Length
  - Cadence, Velocity
  - Support Time, Swing Phase
  - Step Width
- ✅ Normal 정상 대조군 데이터 (좌/우 분리)
- ✅ 모든 값의 표준편차 포함

### 2. gait_long.csv 구조

Long-format 관절 각도 시계열 데이터:

| 컬럼 | 설명 | 예시 |
|------|------|------|
| subject_id | 피험자 ID | S1_01 |
| joint | 관절명 | r.kn.angle |
| gait_cycle | 보행 주기 (0-100%) | 0, 1, 2, ... 100 |
| plane | 평면 | frontal, sagittal, transverse |
| condition1_avg | 환자 평균값 | 2.301 |
| condition1_upper_sd | 환자 상한 SD | 3.027 |
| condition1_lower_sd | 환자 하한 SD | 1.574 |
| condition1_sd | 환자 표준편차 | 0.727 |
| normal_avg | 정상 평균값 | 2.496 |
| normal_upper_sd | 정상 상한 SD | 4.234 |
| normal_lower_sd | 정상 하한 SD | 0.758 |
| normal_sd | 정상 표준편차 | 1.738 |
| normal_sdx2 | 정상 SD×2 | 3.476 |

**데이터 크기**:
- 16개 관절 × 101 보행주기 × 3 평면 = **4,848 레코드/피험자**
- 전체: 4,848 × 21 = **101,808 레코드**

### 3. 관절 목록 (16개)

| 약어 | 관절명 | 설명 |
|------|--------|------|
| r/l.an.angle | Ankle | 우/좌측 발목 |
| r/l.kn.angle | Knee | 우/좌측 무릎 |
| r/l.hi.angle | Hip | 우/좌측 엉덩이 |
| r/l.pe.angle | Pelvis | 우/좌측 골반 |
| r/l.sh.angle | Shoulder | 우/좌측 어깨 |
| r/l.el.angle | Elbow | 우/좌측 팔꿈치 |
| r/l.to.angle | Toe | 우/좌측 발가락 |
| r/l.ga.angle | Gait | 우/좌측 보행 |

### 4. 평면 정의

- **frontal (x축)**: 전두면 - 좌우 움직임 (abduction/adduction)
- **sagittal (y축)**: 시상면 - 전후 움직임 (flexion/extension)
- **transverse (z축)**: 횡단면 - 회전 움직임 (rotation)

---

## 🚀 사용 방법

### Python에서 데이터 로드

```python
import pandas as pd
import json

# 1. 피험자 정보 로드
with open('/data/gait/data/processed_new/S1_01_info.json', 'r') as f:
    info = json.load(f)

print(f"Name: {info['demographics']['name']}")
print(f"Age: {info['demographics']['age']}")
print(f"Patient Cadence: {info['patient']['right']['cadence_steps_min']}")

# 2. 관절 각도 데이터 로드
gait = pd.read_csv('/data/gait/data/processed_new/S1_01_gait_long.csv')

# 3. 특정 관절 필터링 (우측 무릎, 시상면)
right_knee_sag = gait[
    (gait['joint'] == 'r.kn.angle') &
    (gait['plane'] == 'sagittal')
]

# 4. 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(right_knee_sag['gait_cycle'],
         right_knee_sag['condition1_avg'],
         label='Patient', linewidth=2)
plt.plot(right_knee_sag['gait_cycle'],
         right_knee_sag['normal_avg'],
         label='Normal', linestyle='--', linewidth=2)
plt.fill_between(
    right_knee_sag['gait_cycle'],
    right_knee_sag['condition1_lower_sd'],
    right_knee_sag['condition1_upper_sd'],
    alpha=0.2, label='Patient SD'
)
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Knee Flexion Angle (deg)')
plt.title('Right Knee Sagittal Plane Angle')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 전체 피험자 데이터 통합

```python
import glob
import pandas as pd

# 모든 gait CSV 통합
all_gait = pd.concat([
    pd.read_csv(f) for f in
    glob.glob('/data/gait/data/processed_new/S1_*_gait_long.csv')
])

print(f"Total records: {len(all_gait):,}")
print(f"Subjects: {all_gait['subject_id'].nunique()}")

# 피험자별 평균 계산
subject_means = all_gait.groupby(['subject_id', 'joint', 'plane'])['condition1_avg'].mean()
```

### 원본 대비 검증

```python
# 기존 processed 디렉토리와 비교
old_info = json.load(open('/data/gait/data/processed/S1_01_info.json'))
new_info = json.load(open('/data/gait/data/processed_new/S1_01_info.json'))

# 새 버전은 patient/normal 구조 분리, 더 상세한 정보 포함
print("Old keys:", old_info.keys())
print("New keys:", new_info.keys())
```

---

## 📈 변환 스크립트

### 커맨드라인 사용

```bash
# 단일 파일
python convert_excel_to_analysis_format.py \
  --input data/1/excel/S1_01.xlsx \
  --output data/processed_new/

# 배치 변환
python convert_excel_to_analysis_format.py \
  --input data/*/excel/ \
  --output data/processed_new/ \
  --pattern "S1_*.xlsx"
```

### Python API 사용

```python
from convert_excel_to_analysis_format import GaitExcelConverter

converter = GaitExcelConverter()

# 단일 변환
result = converter.convert_excel_file(
    "data/1/excel/S1_01.xlsx",
    "output_dir/"
)

# 배치 변환
summary = converter.batch_convert(
    input_dir="data/*/excel/",
    output_dir="output_dir/",
    pattern="S*.xlsx"
)
```

---

## ✨ 주요 개선사항

### 기존 대비 장점

1. **구조화된 정보 분리**
   - ✅ Demographics, Patient, Normal 명확히 구분
   - ✅ 좌/우 데이터 분리 저장
   - ✅ 모든 파라미터의 SD 포함

2. **완전한 정보 보존**
   - ✅ 92개 피험자 지표 모두 추출
   - ✅ 보행주기 타이밍 정보 포함
   - ✅ Upper/Lower SD 값 보존

3. **분석 친화적 포맷**
   - ✅ Long-format CSV (tidy data)
   - ✅ JSON 메타데이터 (구조화)
   - ✅ 표준 컬럼명 (snake_case)

4. **자동화 및 확장성**
   - ✅ 배치 처리 지원
   - ✅ 에러 핸들링
   - ✅ 진행상황 로깅
   - ✅ 변환 요약 리포트

---

## 📚 참고 문서

- **변환 스크립트**: [convert_excel_to_analysis_format.py](convert_excel_to_analysis_format.py)
- **사용 가이드**: [README_converter.md](README_converter.md)
- **변환 요약**: [/data/gait/data/processed_new/conversion_summary.json](/data/gait/data/processed_new/conversion_summary.json)

---

## 🔍 데이터 검증

### 자동 검증 완료 항목

- ✅ 레코드 수: 4,848개/피험자 (16×101×3)
- ✅ Gait Cycle 범위: 0-100%
- ✅ 평면: frontal, sagittal, transverse
- ✅ 관절: 16개 (r/l × 8 types)
- ✅ 피험자 정보 완전성
- ✅ Patient/Normal 데이터 분리
- ✅ 좌/우 데이터 독립성

### 수동 검증 방법

```python
import pandas as pd

df = pd.read_csv('data/processed_new/S1_01_gait_long.csv')

# 기본 검증
assert len(df) == 4848, "Record count error"
assert df['gait_cycle'].min() == 0, "Gait cycle min error"
assert df['gait_cycle'].max() == 100, "Gait cycle max error"
assert set(df['plane'].unique()) == {'frontal', 'sagittal', 'transverse'}
assert df['joint'].nunique() == 16

print("✅ All validations passed!")
```

---

## 📝 변경 이력

**v1.0** (2025-10-10)
- ✨ 초기 릴리스
- ✅ 21명 피험자 변환 완료
- ✅ Patient/Normal 데이터 구조 분리
- ✅ Long-format CSV 생성
- ✅ 배치 처리 및 요약 리포트
- ✅ 완전한 문서화

---

**문의 및 개선 사항**: 데이터 구조 변경 시 `_extract_subject_info()` 함수 참조
