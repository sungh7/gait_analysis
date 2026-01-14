# Tiered Evaluation Worklog

## 작업 개요
- 목적: MediaPipe 기반 보행 분석과 병원 데이터(info.json, gait_long.csv)를 비교하는 3단계 평가 파이프라인 구축 및 실행
- 산출물: `tiered_evaluation.py`, 수행 결과 `tiered_evaluation_report.json`, 요약 보고서 `tiered_evaluation_summary.md`

## 수행 단계
1. **기존 코드 검토**
   - `mediapipe_csv_processor.py`, `dtw_alignment.py`, `spm_analysis.py` 구조 파악
   - info.json과 gait_long.csv의 필드/포맷 확인 (`processed/S1_xx_info.json`, `processed/S1_xx_gait_long.csv`)
2. **평가 모듈 작성** (`tiered_evaluation.py`)
   - Temporal/Spatial 파라미터 비교 (ICC, RMSE, MAE) → info.json 기준
   - Joint angle 파형 비교 (DTW 정렬, 상관계수, ICC, SPM)
   - 왕복 보행 구간 분리: 고관절 궤적 극값 + turn buffer 활용하여 outbound/inbound/turn 라벨링
   - 방향별 cycle 통계로 stride, cadence, 스탠스 비율 등 재계산
   - 전체 피험자 집계(temporal ICC, SPM, directional metrics)를 JSON으로 저장
3. **파이프라인 실행**
   - 명령: `python tiered_evaluation.py`
   - 출력: `tiered_evaluation_report.json`
4. **결과 요약 작성**
   - `tiered_evaluation_summary.md`: 3단계 결과(ICC 테이블, SPM 의미, 방향 대칭) 및 개선 제안 정리
   - 본 문서(`tiered_evaluation_worklog.md`): 작업 과정 + 주요 수치 기록

## 핵심 결과
- 피험자 수: 21명
- Temporal ICC: 대부분 음수(예: `strides_left` ICC=-0.63), cadence는 병원 측 프로토콜과 불일치(예: MediaPipe 평균 70 vs info.json 49)
- SPM 결과: 6개 관절 모두 85–100% 유의 → 각도 오프셋/스케일 문제 시사
- 방향 분석: inbound에서 stance%가 outbound보다 ~6–7% 긴 경향 (좌측 46.24→53.03, 우측 43.31→49.70)

## 개선 권장 사항
- Cadence 산출 시 병원 측 측정 구간(편도 전용 등)에 맞춰 window 조정
- 7–7.5 m 보행로 기반 거리 스케일을 적용하여 stride/stance 차이 완화
- `angle_converter.py` 조정 또는 바이어스 보정으로 파형 오프셋 제거 후 SPM 재평가
- 피험자 속도 기반 turn buffer 자동 튜닝으로 outbound/inbound 분류 정밀화

