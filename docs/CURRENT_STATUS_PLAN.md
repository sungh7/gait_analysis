# 진행 현황 및 향후 계획

## 1. 현재 진행 상황
- **좌표계 검증**: `coordinate_system_validator.py` 실행 → MediaPipe 좌표계 축 정렬 가이드 (`validation_results_improved/coordinate_system_report.json`) 확보.
- **병원 각도 역공학**: `angle_reverse_engineering.py` 통해 병원 데이터 범위/패턴 분석 및 좌표계 검증 결과 포함한 리포트 생성.
- **표준 전처리 파이프라인**: `preprocessing_pipeline.py` 도입 및 `mediapipe_csv_processor.py` 통합 → 101 포인트 정규화, 가시성 필터, DTW 전처리 준비 완료.
- **각도 변환 + DTW**: `angle_converter.py`, `dtw_alignment.py`, `sagittal_validation_metrics.py` 개선으로 교차검증과 DTW 정렬 파이프라인 구현.
- **SPM 분석**: `spm_analysis.py`에서 FDR 보정·잔차 진단·시각화 지원.
- **통합 실행 스크립트**: `run_improved_validation.py`로 전체 개선 흐름 자동화, 결과 `validation_results_improved/improved_validation_report.json` 작성.
- **회귀 테스트**: `regression_test.py`로 핵심 단계(전처리, 좌표계, 변환, DTW, SPM) 점검.
- **시각화/리포트**: `visualize_improvements.py` 및 `IMPROVEMENT_REPORT.md`로 베이스라인 대비 성능 비교 및 분석 정리.

## 2. 산출물 요약
- 개선 리포트: `validation_results_improved/improved_validation_report.json`
- 좌표계/병원 분석: `validation_results_improved/coordinate_system_report.json`, `validation_results_improved/hospital_angle_analysis.json`
- 회귀 테스트 결과: `validation_results_improved/regression_report.json`
- 비교 시각화: `validation_results_improved/icc_comparison.png`, `validation_results_improved/rmse_comparison.png`
- 개선 요약 문서: `IMPROVEMENT_REPORT.md`

## 3. 남은 과제
- **우측 관절 ICC 악화 원인 분석**: 변환된 각도 vs 병원 값 비교, 피험자별 히트맵/시계열 검토.
- **각도 변환 튜닝**: 가중치/메서드 탐색 폭 확대, 오프셋 정규화 또는 정규화 제약 도입 검토.
- **힐스트라이크 융합 가중치 보정**: 발목 ICC 하락 원인 파악 후 가중치/속도 기준 재학습.
- **SPM 비모수 대안**: 잔차 진단이 비정상일 때 permutation 테스트 도입.
- **CI 연결**: `regression_test.py`를 자동 실행(예: pre-commit 혹은 CI)으로 설정.

## 4. 향후 일정 제안
| 기간 | 작업 | 비고 |
|---|---|---|
| Day 1 | 우측 관절/발목 사례 분석 및 변환 파라미터 튜닝 | 검증 데이터 3~4명 집중 분석 |
| Day 2 | 힐스트라이크 융합 재학습, 회귀 테스트 업데이트 | 융합 가중치 자동 탐색 스크립트 작성 |
| Day 3 | SPM permutation 테스트 프로토타입, 성능 재평가 | 개선 후 `run_improved_validation.py` 재실행 |
| Day 4 | CI 파이프라인 적용 및 문서 업데이트 | GitHub Actions / pre-commit 설정 |
