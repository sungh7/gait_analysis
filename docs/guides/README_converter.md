# 보행분석 엑셀 데이터 변환기

병원 전통적 3D 모션캡처 보행분석 엑셀 파일(S1_*.xlsx)을 Python 분석에 최적화된 포맷(JSON + CSV)으로 변환하는 도구입니다.

## 📋 목차

- [개요](#개요)
- [입력 데이터 구조](#입력-데이터-구조)
- [출력 데이터 구조](#출력-데이터-구조)
- [사용법](#사용법)
- [출력 예시](#출력-예시)

---

## 개요

### 변환 전략

엑셀 파일을 **두 개의 독립된 파일**로 분리:

1. **`{subject}_info.json`**: 피험자 인구통계 및 보행 시공간 파라미터
2. **`{subject}_gait_long.csv`**: Long-format 관절 각도 시계열 데이터

### 처리 규모

- **16개 관절** × **101개 보행주기(0-100%)** × **3축(x, y, z)** = **4,848개 레코드/피험자**

---

## 입력 데이터 구조

### 엑셀 파일 구조 (S1_*.xlsx)

```
행 1-2:   헤더 정보
행 3-1618: 관절 각도 데이터 (16 joints × 101 cycles)
  - A열: 관절명 (r.an.angle, l.kn.angle, ...)
  - B열: Gait Cycle (0-100%)
  - D-O열: Condition 1 데이터 (x,y,z + SD)
  - AB-AP열: Normal 데이터 (x,y,z + SD)

AR1-AV94: 피험자 정보 (92개 지표)
  - 행 3-16: 기본 인구통계 (이름, 나이, 키, 체중, ...)
  - 행 18-34: 보행주기 타이밍 (IDS, SS, Stance, ...)
  - 행 36-63: Patient 보행 파라미터 (Step Length, Cadence, ...)
  - 행 64-91: Normal 정상 대조군 데이터
```

### 관절 목록

```python
16개 관절:
- r/l.an.angle: 우/좌측 발목 (ankle)
- r/l.kn.angle: 우/좌측 무릎 (knee)
- r/l.hi.angle: 우/좌측 엉덩이 (hip)
- r/l.pe.angle: 우/좌측 골반 (pelvis)
- r/l.sh.angle: 우/좌측 어깨 (shoulder)
- r/l.el.angle: 우/좌측 팔꿈치 (elbow)
- r/l.to.angle: 우/좌측 발가락 (toe)
- r/l.ga.angle: 우/좌측 보행 (gait)
```

### 평면 정의

- **x축 (frontal)**: 전두면 (좌우 움직임)
- **y축 (sagittal)**: 시상면 (전후 움직임)
- **z축 (transverse)**: 횡단면 (회전 움직임)

---

## 출력 데이터 구조

### 1. info.json 구조

```json
{
  "subject_id": "S1_01",
  "demographics": {
    "name": "JH Kwak",
    "hospital_id": "01",
    "age": 28,
    "height_cm": 173,
    "weight_kg": 90,
    "gait_cycle_timing": {
      "right_ids": 11.893,
      "right_ss": 37.579,
      "left_ids": 13.031,
      "left_ss": 37.886
    }
  },
  "patient": {
    "right": {
      "step_length_cm": 64.436,
      "step_length_sd": 1.492,
      "cadence_steps_min": 114.802,
      "total_support_time_pct": 62.114,
      "swing_phase_pct": 37.886
    },
    "left": {
      "step_length_cm": 61.305,
      "cadence_steps_min": 115.403,
      ...
    }
  },
  "normal": {
    "right": {
      "step_length_cm": 65.231,
      "cadence_steps_min": 98.434,
      ...
    },
    "left": { ... }
  }
}
```

### 2. gait_long.csv 구조

| subject_id | joint | gait_cycle | plane | condition1_avg | condition1_sd | normal_avg | normal_sd |
|------------|-------|------------|-------|----------------|---------------|------------|-----------|
| S1_01 | r.an.angle | 0 | frontal | 0.166 | 0.721 | -0.25 | 0.399 |
| S1_01 | r.an.angle | 0 | sagittal | -0.312 | 0.989 | 1.588 | 1.648 |
| ... | ... | ... | ... | ... | ... | ... | ... |

**총 4,848행** (16 관절 × 101 사이클 × 3 평면)

---

## 사용법

### 설치

```bash
pip install openpyxl pandas numpy
```

### 기본 사용법

#### 1. 단일 파일 변환

```bash
python convert_excel_to_analysis_format.py \
  --input data/1/excel/S1_01.xlsx \
  --output data/processed/
```

#### 2. 디렉토리 일괄 변환

```bash
python convert_excel_to_analysis_format.py \
  --input data/1/excel/ \
  --output data/processed/ \
  --pattern "S1_*.xlsx"
```

#### 3. 특정 패턴 필터링

```bash
# S1_0으로 시작하는 파일만 변환
python convert_excel_to_analysis_format.py \
  --input data/1/excel/ \
  --output data/processed/ \
  --pattern "S1_0*.xlsx"
```

### Python 코드에서 사용

```python
from convert_excel_to_analysis_format import GaitExcelConverter

# 변환기 초기화
converter = GaitExcelConverter()

# 단일 파일 변환
result = converter.convert_excel_file(
    excel_path="data/1/excel/S1_01.xlsx",
    output_dir="data/processed/"
)

print(f"Success: {result['success']}")
print(f"Records: {result['record_count']}")

# 배치 변환
summary = converter.batch_convert(
    input_dir="data/1/excel/",
    output_dir="data/processed/",
    pattern="S*.xlsx"
)

print(f"Total: {summary['total_files']}")
print(f"Success: {summary['successful']}")
print(f"Total Records: {summary['total_records']:,}")
```

---

## 출력 예시

### 변환 성공 메시지

```
INFO:Processing S1_01...
INFO:  ✓ Saved: S1_01_info.json, S1_01_gait_long.csv
INFO:Processing S1_02...
INFO:  ✓ Saved: S1_02_info.json, S1_02_gait_long.csv

============================================================
변환 완료: 21/21 성공
총 레코드: 101,808
요약 파일: data/processed/conversion_summary.json
============================================================
```

### conversion_summary.json

```json
{
  "total_files": 21,
  "successful": 21,
  "failed": 0,
  "total_records": 101808,
  "subjects": [
    {
      "subject_id": "S1_01",
      "success": true,
      "record_count": 4848
    },
    ...
  ]
}
```

---

## 데이터 분석 예시

### pandas로 분석

```python
import pandas as pd
import json

# 1. 피험자 정보 로드
with open('data/processed/S1_01_info.json', 'r') as f:
    info = json.load(f)

print(f"Name: {info['demographics']['name']}")
print(f"Age: {info['demographics']['age']}")
print(f"Patient Cadence (Right): {info['patient']['right']['cadence_steps_min']}")

# 2. 관절 각도 데이터 로드
gait_df = pd.read_csv('data/processed/S1_01_gait_long.csv')

# 3. 특정 관절 필터링
right_knee = gait_df[
    (gait_df['joint'] == 'r.kn.angle') &
    (gait_df['plane'] == 'sagittal')
]

# 4. 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(right_knee['gait_cycle'], right_knee['condition1_avg'], label='Patient')
plt.plot(right_knee['gait_cycle'], right_knee['normal_avg'], label='Normal', linestyle='--')
plt.fill_between(
    right_knee['gait_cycle'],
    right_knee['condition1_avg'] - right_knee['condition1_sd'],
    right_knee['condition1_avg'] + right_knee['condition1_sd'],
    alpha=0.2
)
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Knee Angle (deg)')
plt.title('Right Knee Sagittal Angle')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 다중 피험자 비교

```python
import pandas as pd
import glob
import json

# 모든 피험자 정보 수집
all_subjects = []
for info_file in glob.glob('data/processed/S1_*_info.json'):
    with open(info_file, 'r') as f:
        data = json.load(f)
        all_subjects.append({
            'subject_id': data['subject_id'],
            'age': data['demographics']['age'],
            'height': data['demographics']['height_cm'],
            'cadence_right': data['patient']['right']['cadence_steps_min'],
            'cadence_left': data['patient']['left']['cadence_steps_min']
        })

subjects_df = pd.DataFrame(all_subjects)
print(subjects_df.describe())

# 전체 관절 데이터 결합
all_gait = pd.concat([
    pd.read_csv(f) for f in glob.glob('data/processed/S1_*_gait_long.csv')
])

print(f"Total records: {len(all_gait):,}")
print(f"Unique subjects: {all_gait['subject_id'].nunique()}")
print(f"Unique joints: {all_gait['joint'].nunique()}")
```

---

## 데이터 검증

### 자동 검증 항목

- ✅ 4,848 레코드 (16 관절 × 101 사이클 × 3 평면)
- ✅ Gait cycle 범위: 0-100
- ✅ 관절명 유효성 검사
- ✅ 평면 유효성 검사 (frontal, sagittal, transverse)
- ✅ 피험자 정보 완전성

### 수동 검증 방법

```python
import pandas as pd

# CSV 검증
df = pd.read_csv('data/processed/S1_01_gait_long.csv')

assert len(df) == 4848, "Record count mismatch"
assert set(df['gait_cycle'].unique()) == set(range(101)), "Gait cycle range error"
assert df['plane'].isin(['frontal', 'sagittal', 'transverse']).all(), "Invalid plane"

print("✅ Validation passed!")
```

---

## 문제 해결

### 일반적인 오류

**1. openpyxl 경고 무시**
```
UserWarning: Cannot parse header or footer
```
→ 무시해도 됩니다. 데이터 추출에 영향 없음.

**2. 레코드 수 불일치**
```python
# 엑셀 구조 확인
python -c "
import openpyxl
wb = openpyxl.load_workbook('data/1/excel/S1_01.xlsx')
ws = wb.active
print(f'Rows: {ws.max_row}, Cols: {ws.max_column}')
"
```

**3. 컬럼 매핑 오류**
- AR(44)=Label, AS(45)=Right, AT(46)=Left 구조 확인
- 행 번호가 정확한지 검증

---

## 라이센스 및 기여

이 도구는 연구 목적으로 개발되었습니다.

**문의**: 데이터 구조 변경 시 `_extract_subject_info()` 및 `_extract_gait_angles()` 함수 수정 필요

---

## 변경 이력

- **v1.0** (2025-10-10): 초기 버전
  - 16개 관절 지원
  - Patient/Normal 데이터 분리
  - Long-format CSV 출력
  - 배치 처리 기능
