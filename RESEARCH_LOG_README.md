# Research Log System - Quick Start

**Created:** 2025-10-10
**Purpose:** Maintain academic-style documentation of gait analysis validation research
**Status:** Active

---

## 📁 File Structure

```
/data/gait/
├── RESEARCH_LOG.md                    ⭐ MAIN DOCUMENT (논문 형식)
├── LOG_USAGE_GUIDE.md                 📖 사용 가이드
├── RESEARCH_LOG_README.md             📄 이 파일 (빠른 시작)
│
├── supplementary/                      📂 상세 자료
│   ├── methods/                        🔬 방법론 상세
│   │   └── P1_stride_based_scaling_method.md
│   ├── results/                        📊 결과 상세
│   │   ├── P0_baseline_analysis.md
│   │   ├── P0_baseline_audit_results.json
│   │   ├── P1_scaling_results.md
│   │   ├── P1_scaling_test_results.json
│   │   └── baseline_metrics_20251010.json
│   └── experiments/                    🧪 실험 로그
│       └── README.md
│
└── [기존 프로젝트 파일들...]
```

---

## 🚀 빠른 시작

### 작업 완료 후 바로 기록하기

#### 1. 작은 작업 (버그 수정, 테스트) - 5분

`RESEARCH_LOG.md` → **Section 9 (Session Log)** 에 추가:

```markdown
**HH:MM-HH:MM | 작업명**
- 수행한 내용
- 결과: 메트릭 변화 등
```

#### 2. Phase 완료 (주요 마일스톤) - 30-45분

1. **해당 Phase 섹션 작성** (Section 4, 5, 6...)
   - Methods, Results, Discussion 구조
   - 통계 분석 포함

2. **Progress Tracker 업데이트** (Section 7)
   ```markdown
   | P1 | Step Length | 49.9 cm | 21.7 cm | <10 cm | 65% | 🟡 Partial |
   ```

3. **Session Log 기록** (Section 9)

4. **Supplementary 문서 작성**
   - `supplementary/methods/P{N}_method.md`
   - `supplementary/results/P{N}_results.md`

#### 3. 실험 테스트 - 20분

1. 새 파일 생성: `supplementary/experiments/2025-10-11_test_name.md`
2. Session Log에 참조 기록

---

## 📝 현재 상태 (2025-10-10 기준)

### 완료된 작업

| Phase | 내용 | 상태 | 문서 |
|-------|------|------|------|
| **P0** | Baseline Audit | ✅ 완료 | [P0 Analysis](supplementary/results/P0_baseline_analysis.md) |
| **P1** | Stride-based Scaling | 🟡 부분 (5-subject test) | [P1 Method](supplementary/methods/P1_stride_based_scaling_method.md) |

### 주요 발견

- **Step length error:** 49.9 cm → 21.7 cm (-54% improvement, P1 test)
- **Strike over-detection:** 3.45× inflation (전체 21명)
- **Cadence ICC:** -0.033 (Poor, 개선 필요)

### 다음 단계

1. P1 전체 코호트 적용 (n=21)
2. P2 Cadence Refactor 시작
3. P3 Strike Detection Tuning

---

## 💡 사용 시나리오

### Scenario 1: 매일 작업 종료 시

```bash
# 1. RESEARCH_LOG.md 열기
vim RESEARCH_LOG.md

# 2. Section 9으로 이동 (하단)
# 3. 오늘 작업 요약 추가
**16:00-18:00 | 작업 내용**
- 완료한 내용
- 다음 할 일

# 4. 저장 및 커밋
git add RESEARCH_LOG.md
git commit -m "Session 2025-10-11: 작업 요약"
```

### Scenario 2: Phase 완료

```bash
# 1. Supplementary 문서 작성
vim supplementary/methods/P2_ransac_cadence.md
vim supplementary/results/P2_test_results.json

# 2. RESEARCH_LOG.md 해당 Phase 섹션 작성
vim RESEARCH_LOG.md  # Section 5 업데이트

# 3. Progress Tracker 업데이트
# Section 7 테이블 수정

# 4. 커밋 및 태그
git add RESEARCH_LOG.md supplementary/
git commit -m "Phase 2 complete: RANSAC cadence (ICC: 0.38)"
git tag P2-complete
```

### Scenario 3: 논문 작성 준비

```bash
# RESEARCH_LOG.md를 기반으로 논문 초안 생성
cp RESEARCH_LOG.md manuscript_draft.md

# 편집:
# - Section 9 (Session Log) 삭제
# - "Status: Planned" 섹션 제거
# - References 완성
# - 저자, 소속 추가
```

---

## 📖 상세 가이드

- **사용법 전체:** [LOG_USAGE_GUIDE.md](LOG_USAGE_GUIDE.md)
- **Main Log:** [RESEARCH_LOG.md](RESEARCH_LOG.md)

### 주요 섹션 구조

`RESEARCH_LOG.md` 구조:
- **Section 1-2:** Introduction, Methods (고정)
- **Section 3:** Baseline Results (P0, 완료)
- **Section 4:** Phase 1 Results (P1, 진행중)
- **Section 5-6:** Phase 2-3 (계획)
- **Section 7:** Progress Tracker (실시간 업데이트)
- **Section 8:** Software/Code 정보
- **Section 9:** Session Log (매일 업데이트)
- **Section 10:** Discussion (최종 정리)

---

## ✅ 체크리스트

### 매 작업 후

- [ ] Session Log 업데이트 (Section 9)
- [ ] Git commit
- [ ] 다음 할 일 기록

### Phase 완료 후

- [ ] Phase 섹션 작성 (Methods + Results + Discussion)
- [ ] Progress Tracker 업데이트 (Section 7)
- [ ] Supplementary 문서 작성
- [ ] Session Log에 요약 기록
- [ ] Git commit + tag

### 주간 (매주 금요일)

- [ ] Progress Tracker 정확성 확인
- [ ] 오래된 Session Log 아카이브 (>10 entries 시)
- [ ] Supplementary 파일 정리

---

## 🔧 유용한 명령어

### 통계 확인

```bash
# 전체 라인 수
wc -l RESEARCH_LOG.md

# 섹션 구조 확인
grep "^##" RESEARCH_LOG.md

# Session Log 항목 수
grep "^\*\*[0-9]" RESEARCH_LOG.md | wc -l
```

### 빠른 네비게이션

```bash
# Section 9으로 바로 이동 (vim)
vim +/'^## 9' RESEARCH_LOG.md

# Progress Tracker 확인 (Section 7)
sed -n '/^## 7\./,/^## 8\./p' RESEARCH_LOG.md | less
```

### 백업

```bash
# 타임스탬프 백업
DATE=$(date +%Y%m%d_%H%M)
cp RESEARCH_LOG.md backups/RESEARCH_LOG_${DATE}.md
```

---

## 📚 예시 Entry

### Good Session Log Entry ✅

```markdown
**09:00-12:00 | Phase 2: RANSAC Cadence Implementation**

**Objective:** Replace heuristic cadence estimator with RANSAC-based method

**Activities:**
1. Implemented `estimate_cadence_ransac()` (185 lines)
2. Added minimum stride interval enforcement (0.6s)
3. Tested on 5-subject validation set

**Results:**
- Cadence ICC: -0.033 → **0.38** (p = 0.002)
- RMSE: 19.3 → 13.5 steps/min (-30%)
- All 5 subjects improved

**Deliverables:**
- `cadence_estimation_v4.py`
- `P2_cadence_test_results.json`
- `supplementary/methods/P2_ransac_cadence.md`

**Next:** Full cohort validation (n=21)
```

### Bad Entry ❌

```markdown
**Today | Worked on cadence**
- Fixed some bugs
- It works better now
```

---

## 🆘 문제 해결

### Q: Session Log가 너무 길어졌어요

**A:** 오래된 세션을 아카이브로 이동:
```bash
# 최근 3개 세션만 유지, 나머지는 아카이브
# supplementary/experiments/session_archive_2025-10.md로 이동
```

### Q: 어디에 업데이트해야 할지 모르겠어요

**A:** 매핑 참고:
- 일일 작업 → Section 9
- 메트릭 변화 → Section 7 + Section 9
- Phase 완료 → 해당 Phase 섹션 + Section 7 + Section 9
- 상세 방법론 → supplementary/methods/
- 상세 결과 → supplementary/results/

### Q: LaTeX 수식이 렌더링 안 돼요

**A:**
- VS Code: Markdown Preview 사용
- 온라인: https://dillinger.io 또는 https://stackedit.io
- 기존 수식 복사해서 수정하기

---

## 📞 도움말

더 자세한 정보:
1. [LOG_USAGE_GUIDE.md](LOG_USAGE_GUIDE.md) - 전체 사용법
2. [RESEARCH_LOG.md](RESEARCH_LOG.md) Section 1-6 - 기존 예시 참고
3. Phase 1 섹션 - 완성된 예시로 활용

---

**버전:** 1.0
**마지막 업데이트:** 2025-10-10
**유지 관리:** 연구팀
