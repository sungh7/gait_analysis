# 보행분석 데이터 자동 변환 시스템

엑셀 형식의 병원 보행분석 데이터를 Python 분석에 최적화된 형태로 자동 변환하는 시스템입니다.

## 📁 파일 구조

```
/data/gait/
├── gait_parser.py         # 엑셀 파일 파싱 클래스
├── convert_all.py         # 일괄 변환 스크립트
├── utils.py               # 분석용 유틸리티 함수
├── processed/             # 변환된 데이터 저장소
│   ├── S1_01_info.json           # 피험자 정보
│   ├── S1_01_gait_long.csv       # 관절 각도 데이터 (Long format)
│   ├── all_subjects_combined.csv # 전체 피험자 통합 데이터
│   └── conversion_summary.json   # 변환 요약 리포트
└── data/                  # 원본 엑셀 파일들
    └── */excel/*.xlsx
```

## 🚀 사용 방법

### 1. 데이터 변환

모든 엑셀 파일을 자동으로 변환:

```bash
python3 convert_all.py
```

**결과:**
- ✅ 21개 파일 처리 완료
- 📊 101,808개 레코드 생성
- 📂 `/data/gait/processed/` 폴더에 저장

### 2. Python에서 데이터 로드 및 분석

```python
from utils import *

# 피험자 목록 확인
subjects = list_available_subjects()
print(subjects)  # ['S1_01', 'S1_02', ...]

# 피험자 정보 로드
info = load_subject_info('S1_01')
print(info['name'], info['age'], info['height_cm'])

# 보행 데이터 로드
df = load_gait_data('S1_01')
print(df.shape)  # (4848, 13)

# 전체 피험자 데이터 로드
all_data = load_all_subjects()
print(all_data['subject_id'].unique())
```

### 3. 특정 관절 분석

```python
# 오른쪽 무릎(sagittal plane) 데이터 추출
knee_data = filter_joint_plane(df, 'r.kn.angle', 'y')

# 정상 범위와 비교 분석
knee_with_dev = calculate_deviation(knee_data)
print(knee_with_dev[['gait_cycle', 'condition1_avg', 'normal_avg',
                      'deviation_normalized', 'is_outside_normal_sd']])

# 특정 관절 비교
comparison = get_joint_comparison('S1_01', 'r.kn.angle', 'y')
print(comparison)
```

### 4. 좌우 비교 분석

```python
# 무릎 좌우 비교
bilateral = get_bilateral_comparison('S1_01', 'kn', 'y')
print(bilateral[['gait_cycle', 'condition1_avg_right', 'condition1_avg_left', 'difference']])
```

## 📊 데이터 구조

### Long Format CSV 구조

각 `{subject_id}_gait_long.csv` 파일은 다음 열을 포함합니다:

| 열 이름 | 설명 | 예시 |
|---------|------|------|
| `subject_id` | 피험자 ID | S1_01 |
| `joint` | 관절 코드 | r.kn.angle |
| `gait_cycle` | 보행 주기 (0-100) | 50 |
| `plane` | 해부학적 평면 | x, y, z |
| `condition1_avg` | 피험자 측정값 평균 | 62.34 |
| `condition1_upper_sd` | 피험자 상위 표준편차 | 65.12 |
| `condition1_lower_sd` | 피험자 하위 표준편차 | 59.56 |
| `condition1_sd` | 피험자 표준편차 | 2.78 |
| `normal_avg` | 정상 참조값 평균 | 60.15 |
| `normal_upper_sd` | 정상 상위 표준편차 | 63.89 |
| `normal_lower_sd` | 정상 하위 표준편차 | 56.41 |
| `normal_sd` | 정상 표준편차 | 3.74 |
| `normal_sdx2` | 정상 2배 표준편차 | 7.48 |

**총 레코드 수**: 16개 관절 × 101개 gait cycle × 3개 plane = **4,848 rows/subject**

### 관절 코드

| 코드 | 관절 이름 | 코드 | 관절 이름 |
|------|-----------|------|-----------|
| `r.an.angle` | Right Ankle | `l.an.angle` | Left Ankle |
| `r.kn.angle` | Right Knee | `l.kn.angle` | Left Knee |
| `r.hi.angle` | Right Hip | `l.hi.angle` | Left Hip |
| `r.ga.angle` | Right Gait | `l.ga.angle` | Left Gait |
| `r.pe.angle` | Right Pelvis | `l.pe.angle` | Left Pelvis |
| `r.to.angle` | Right Torso | `l.to.angle` | Left Torso |
| `r.sh.angle` | Right Shoulder | `l.sh.angle` | Left Shoulder |
| `r.el.angle` | Right Elbow | `l.el.angle` | Left Elbow |

### 해부학적 평면

- **x (Frontal/Coronal)**: 좌우 움직임 (abduction/adduction)
- **y (Sagittal)**: 앞뒤 움직임 (flexion/extension) - 가장 많이 사용
- **z (Transverse/Horizontal)**: 회전 움직임 (rotation)

## 🛠️ 주요 함수

### 데이터 로드

```python
load_subject_info(subject_id)         # 피험자 정보 로드
load_gait_data(subject_id)            # 피험자 보행 데이터 로드
load_all_subjects()                   # 전체 피험자 통합 데이터 로드
list_available_subjects()             # 사용 가능한 피험자 목록
```

### 데이터 필터링

```python
filter_joint_plane(df, joint, plane)  # 특정 관절/평면 필터링
filter_joints(df, joint_list)         # 여러 관절 필터링
```

### 분석 함수

```python
calculate_deviation(df)                        # 정상 범위와 편차 계산
get_outlier_summary(df, by='joint')           # 이상치 요약
get_joint_comparison(subject_id, joint, plane) # 관절별 정상 비교
get_bilateral_comparison(subject_id, 'kn')    # 좌우 비교
```

### 헬퍼 함수

```python
get_joint_name_mapping()              # 관절 코드 → 이름 매핑
get_plane_name_mapping()              # 평면 코드 → 이름 매핑
create_gait_cycle_pivot(df, ...)      # Pivot 테이블 생성
```

## 📈 분석 예시

### 1. 정상 범위 벗어난 시점 찾기

```python
df = load_gait_data('S1_01')
knee = filter_joint_plane(df, 'r.kn.angle', 'y')
knee_dev = calculate_deviation(knee)

# 1 SD 벗어난 시점
outliers = knee_dev[knee_dev['is_outside_normal_sd']]
print(f"정상 범위 벗어난 시점: {len(outliers)}개")
print(outliers[['gait_cycle', 'condition1_avg', 'normal_avg', 'deviation_normalized']])
```

### 2. 전체 관절 이상치 비율

```python
df = load_gait_data('S1_01')
df_dev = calculate_deviation(df)
summary = get_outlier_summary(df_dev, by='joint')
print(summary.sort_values('pct_outside_1sd', ascending=False))
```

### 3. 여러 피험자 비교

```python
all_data = load_all_subjects()
knee_all = filter_joint_plane(all_data, 'r.kn.angle', 'y')

# Pivot: gait_cycle × subject_id
pivot = create_gait_cycle_pivot(knee_all, 'r.kn.angle', 'y')
print(pivot)

# 시각화
import matplotlib.pyplot as plt
pivot.plot(legend=False, alpha=0.5)
plt.title('Right Knee Angle (Sagittal) - All Subjects')
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Angle (degrees)')
plt.show()
```

## 🔄 재변환

새 엑셀 파일 추가 후 재변환:

```bash
# 전체 재변환
python3 convert_all.py

# 특정 파일만 변환
python3 -c "
from gait_parser import GaitDataParser
parser = GaitDataParser('/data/gait/data/1/excel/S1_01.xlsx')
info = parser.extract_subject_info()
gait = parser.extract_gait_data_long('S1_01')
gait.to_csv('/data/gait/processed/S1_01_gait_long.csv', index=False)
"
```

## ✅ 검증 완료

- ✅ 21개 엑셀 파일 변환 성공
- ✅ 101,808개 레코드 생성 (21 subjects × 4,848 records)
- ✅ 피험자 정보 추출 정상
- ✅ 정상 참조값 포함
- ✅ 16개 관절, 3개 평면, 101개 gait cycle 검증
- ✅ 유틸리티 함수 테스트 완료

## 📝 참고사항

### Plane 방향 정의
- **x-frontal**: 좌우(coronal) 평면
- **y-sagittal**: 앞뒤 평면 (가장 중요)
- **z-transverse**: 수평(horizontal) 평면

### Condition 1
- 피험자의 측정값 (병원 검사 결과)
- Upper/Lower SD: 측정 신뢰 구간

### Normal
- 연령별 정상 참조 데이터
- SD, SDX2: 통계적 정상 범위 판단용

### Gait Cycle Timing 정보
- **IDS**: Initial Double Support
- **SS**: Single Support
- **SLS**: Second Late Stance
- Cadence: 분당 걸음 수

## 🆘 문제 해결

### "File not found" 에러
```bash
# 엑셀 파일 경로 확인
ls /data/gait/data/*/excel/*.xlsx
```

### "Module not found" 에러
```bash
pip install pandas numpy openpyxl tqdm
```

### 변환 결과 확인
```bash
cat /data/gait/processed/conversion_summary.json
```
