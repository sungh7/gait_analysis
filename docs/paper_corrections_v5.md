# Paper Corrections for V5 Results

## Errors Found in Original Draft

### 1. Strike Ratio Inaccuracy
**Original claim**: "힐 스트라이크 비율은 0.83×로 안정화"
**Actual data**: mean=0.875×, median=0.917×

**Correction needed**:
```
힐 스트라이크 비율은 평균 0.88×, 중앙값 0.92×로 측정되었다.
```

### 2. Missing Underdetection Discussion
**Issue**: Paper does not acknowledge that 12/21 subjects (57%) have strike ratio <0.8
**Actual data**:
- <0.8× count: 12 subjects
- >1.2× count: 0 subjects
- Range: ~0.5-1.3×

**Addition needed**:
```
그러나 21명 중 12명(57%)의 피험자에서 0.8 미만의 비율이 관찰되어,
일부 실제 힐 스트라이크를 놓치는 과소 감지 문제가 존재한다.
```

### 3. Missing S1_02 Outlier Discussion
**Issue**: Catastrophic failure case (60 steps/min error) not mentioned
**Actual data**: S1_02 has +23 steps/min error, 37% larger than second-worst case

**Addition needed** (in limitations section):
```
특히 S1_02 피험자의 경우 케이던스 오차가 60 steps/min로,
우측 다리의 과다 감지(GT: 13 vs 감지: 49)로 인한 치명적 실패가 발생했다.
이는 템플릿 매칭 방식의 한계를 보여주는 사례이다.
```

### 4. Dataset Size Ambiguity
**Original**: "21명 전체를 대상으로 측정"
**Issue**: Frontal analysis used n=26

**Correction**:
```
측면 분석은 21명, 정면 분석은 26명을 대상으로 수행하였다.
```

### 5. Trade-off Not Acknowledged
**Issue**: Paper presents improvement from baseline (3.45×) → V5 (0.83×) as pure gain
**Reality**: Traded overdetection for underdetection

**Addition needed**:
```
기준선의 과다 감지 문제(3.45×)를 개선하였으나,
일부 피험자에서 과소 감지 문제가 새롭게 발생하였다 (0.88× 평균).
이는 정밀도(precision)와 재현율(recall) 간의 트레이드오프로 이해할 수 있다.
```

## Corrected Abstract (Korean)

### Original Problem Statement
```
MediaPipe 기반 측·정면 보행 분석 파이프라인의 성능 개선 및 정량 검증

기존 힐 스트라이크 과다 감지 문제(비율 3.45×)를 해결하기 위해,
템플릿 기반 DTW 매칭과 RANSAC 케이던스 추정을 도입한 V5 파이프라인을 개발하였다.

결과:
- 힐 스트라이크 비율은 0.83×로 안정화
- 보폭 RMSE는 좌우 각각 11.2/12.6 cm
- 케이던스 MAE는 7.9 steps/min
- 21명 전체를 대상으로 측정
- 정면 분석: Step Width 6.5 cm, Symmetry 93.3%
```

### Corrected Version
```
MediaPipe 기반 측·정면 보행 분석 파이프라인의 성능 개선 및 정량 검증

**배경**: 기존 힐 스트라이크 과다 감지 문제(비율 3.45×)를 해결하기 위해,
템플릿 기반 DTW 매칭(threshold=0.7)과 RANSAC 케이던스 추정을 도입한 V5 파이프라인을 개발하였다.

**방법**:
- 측면 영상 21명, 정면 영상 26명의 병원 데이터로 검증
- 회전 구간 필터링 및 품질 기반 스케일 선택 적용
- ICC, RMSE, MAE 메트릭으로 병원 gold standard와 비교

**결과**:
- 힐 스트라이크 비율: 평균 0.88×, 중앙값 0.92× (과다 감지 문제 해결)
- 보폭 RMSE: 좌 11.2 cm, 우 12.6 cm (ICC: 0.23/0.05)
- 케이던스 MAE: 7.9 steps/min (ICC: 0.21)
- 정면 분석 (n=26): Step Width 6.5±1.8 cm, Symmetry 93.3±6.2%

**한계**:
- 12/21 피험자(57%)에서 힐 스트라이크 과소 감지 (<0.8× 비율)
- S1_02 케이던스 오차 60 steps/min (치명적 실패 사례)
- ICC 점수 낮음 (목표 >0.75, 실제 0.05-0.28)
- 골반 경사각 과대 추정 (23.2±11.9 deg)

**결론**:
V5 파이프라인은 과다 감지를 성공적으로 억제하였으나,
일부 피험자에서 과소 감지 문제와 낮은 상관도를 보여
임상 적용을 위한 추가 개선이 필요하다.
```

## Key Changes Summary

| Item | Original | Corrected | Reason |
|------|----------|-----------|--------|
| Strike ratio | 0.83× | 0.88× (mean), 0.92× (median) | Actual data |
| Underdetection | Not mentioned | 12/21 subjects <0.8× | Transparency |
| S1_02 outlier | Not mentioned | +60 steps/min error | Acknowledge failure |
| Dataset size | "21명 전체" | n=21 (sagittal), n=26 (frontal) | Clarity |
| ICC scores | Not emphasized | 0.05-0.28 (poor) | Honest reporting |
| Trade-off | Not discussed | Precision vs recall balance | Context |
| Limitations | Minimal | Comprehensive section added | Scientific honesty |

## Recommendations for Paper Structure

### Add New Section: "5. Limitations and Failure Analysis"

```markdown
## 5. Limitations and Failure Analysis

### 5.1 Underdetection in Subset of Subjects
템플릿 매칭 임계값 0.7은 전체 코호트의 평균 최적값이나,
개별 피험자의 보행 패턴 차이로 인해 12/21 피험자에서 과소 감지가 발생하였다.
향후 피험자별 적응형 임계값 설정이 필요하다.

### 5.2 Catastrophic Failure Case (S1_02)
S1_02 피험자는 우측 다리 과다 감지(GT: 13 → 감지: 49)로 인해
케이던스 오차 60 steps/min을 기록하였다.
원인 분석 결과, 템플릿 품질 저하 또는 비정상적 보행 패턴으로 추정된다.
이는 현재 파이프라인의 robustness 한계를 보여준다.

### 5.3 Low ICC Scores
Step length ICC (0.05-0.28)는 임상 검증 기준 (>0.75)에 미치지 못한다.
이는 스케일링 오차 누적 및 회전 구간 오감지가 원인으로 분석된다.

### 5.4 Frontal Analysis Issues
- 골반 경사각(23.2±11.9 deg): 절대값 과대 추정
- Lateral sway 고분산(36.4±42.7 cm): 6명의 outlier 존재
```

## Action Items

- [ ] Update abstract with corrected numbers
- [ ] Add comprehensive limitations section
- [ ] Create S1_02 case study subsection
- [ ] Add threshold tuning discussion (why 0.7 was chosen)
- [ ] Update all tables/figures with correct strike ratio
- [ ] Add scatter plot showing S1_02 as clear outlier
- [ ] Revise conclusion to acknowledge limitations honestly

## Timeline

- **Phase 1 (Immediate)**: Correct numbers in abstract and results
- **Phase 2 (This week)**: Add limitations section
- **Phase 3 (Next week)**: Create supplementary materials with detailed failure analysis
