# Research Log Usage Guide

**Document Purpose:** Instructions for maintaining and updating `RESEARCH_LOG.md`
**Target Users:** Research team members, future investigators
**Last Updated:** 2025-10-10

---

## 1. Overview

`RESEARCH_LOG.md` is the **single source of truth** for all gait analysis validation work. It follows academic paper structure (Abstract, Methods, Results, Discussion) for publishability while tracking ongoing progress.

### File Structure

```
/data/gait/
├── RESEARCH_LOG.md              # Main document (THIS IS MASTER)
├── supplementary/
│   ├── methods/
│   │   └── P{N}_{method_name}.md    # Detailed methodology
│   ├── results/
│   │   └── P{N}_{analysis}.md       # Detailed results/figures
│   └── experiments/
│       └── YYYY-MM-DD_{test}.md     # Experimental logs
```

---

## 2. When to Update

### 2.1 Small Task Completion (10-30 min work)

**→ Update: Section 9 (Session Log)**

Add bullet point under current session:

```markdown
### Session 2025-10-10 (Total: 5 hours)

**14:30-15:00 | Bug Fix: Strike Detection Threshold**
- Fixed prominence parameter from 0.5 to 0.7
- Reduced over-detection ratio from 3.45× to 2.1×
- **Result:** Preliminary test on S1_01 shows improvement

**Next:** Full validation on test cohort
```

### 2.2 Phase Completion (Major Milestone)

**→ Update: Corresponding Phase Section + Section 9**

Example: After completing Phase 1 full cohort validation:

1. **Update Section 4 (Phase 1: Scaling Calibration)**
   - Add full cohort results table (n=21)
   - Update statistical analysis (t-test, effect size)
   - Finalize Discussion subsection

2. **Update Section 7 (Progress Tracker)**
   ```markdown
   | P1: Scaling Calibration | Step Length Error | 49.9 cm | 18.5 cm | <10 cm | 78% | 🟢 **Done** |
   ```

3. **Update Section 9 (Session Log)**
   ```markdown
   **15:00-17:00 | Phase 1: Full Cohort Validation**
   - Integrated stride-based scaling into V4
   - Ran all 21 subjects
   - Mean improvement: 62% (baseline: 49.9 cm → 18.5 cm)

   **Deliverables:**
   - `tiered_evaluation_v4.py` (modified)
   - `tiered_evaluation_report_v4.json`

   **Status:** ✅ Phase 1 complete, proceeding to Phase 2
   ```

### 2.3 New Experimental Test

**→ Create: `supplementary/experiments/YYYY-MM-DD_{test}.md`**

Then reference in Session Log:

```markdown
**10:00-11:00 | Experiment: RANSAC vs Percentile Cadence**
- Tested two cadence estimators on 5 subjects
- **See:** `supplementary/experiments/2025-10-11_cadence_comparison.md`
- **Result:** RANSAC ICC: 0.38 vs Percentile ICC: -0.03 (RANSAC wins)

**Decision:** Adopt RANSAC for Phase 2
```

### 2.4 Metric Change

**→ Update: Section 7.2 (Cumulative Improvements)**

When any metric improves:

```markdown
| Step Length ICC | -0.771 | 0.30 | **0.42** | 0.45 | ≥0.60 | 🟡 Approaching |
```

And note in Session Log:

```markdown
**Metric Update:** Step Length ICC improved from 0.30 (P1) to 0.42 (P2) after cadence refinement
```

---

## 3. Update Templates

### 3.1 Session Log Entry

```markdown
**HH:MM-HH:MM | Task Name: Brief Description**

**Objective:** What you're trying to achieve

**Activities:**
1. Action taken
2. Code written/modified
3. Analysis performed

**Results:**
- Key finding 1
- Metric change: X → Y (Z% improvement)

**Deliverables:**
- File created/modified
- Data outputs

**Next Steps:** What comes next
```

### 3.2 Phase Section (When Complete)

Use existing Phase 1 (Section 4) as template:
1. **Motivation** (1-2 paragraphs)
2. **Method Development** (formulas, code)
3. **Experimental Validation** (tables, stats)
4. **Residual Error Analysis** (if applicable)
5. **Discussion** (achievements, limitations, implications)
6. **Next Steps**

### 3.3 Supplementary Method Document

```markdown
# Phase {N}: {Method Name} - Technical Documentation

**Method Name:** Descriptive name
**Implementation File:** `{filename}.py`
**Date Developed:** YYYY-MM-DD
**Status:** {Developed | Validated | Integrated}

---

## 1. Background and Motivation
{Problem statement, hypothesis}

## 2. Proposed Method
{Mathematical formulation, algorithms}

## 3. Algorithm Implementation
{Code with detailed comments}

## 4. Validation Protocol
{How to test, acceptance criteria}

## 5. Results Summary
{Tables, statistical analysis}

## 6. Advantages and Limitations
{Pros, cons, edge cases}

## 7. Implementation Considerations
{How to integrate into main pipeline}

## 8. Future Enhancements
{Potential improvements}

## 9. Conclusion
{Summary, recommendations}
```

---

## 4. Writing Style Guidelines

### 4.1 Tone

**Maintain academic/scientific tone:**
- ✅ "Results showed significant improvement (p < 0.001)"
- ❌ "It worked great!"

- ✅ "Phase 1 addressed spatial scaling bias"
- ❌ "We fixed the scaling problem"

- ✅ "Residual error suggests underlying detection issue"
- ❌ "Still not perfect because detector is broken"

### 4.2 Quantification

**Always include numbers:**
- ✅ "Reduced error by 54.6% (47.7 → 21.7 cm, p < 0.001)"
- ❌ "Reduced error significantly"

- ✅ "All 21 subjects (100%) exceeded threshold"
- ❌ "Most subjects exceeded threshold"

### 4.3 Structure

**Use consistent formatting:**
- Sections numbered (1, 1.1, 1.1.1)
- Tables with headers and units
- Formulas with LaTeX ($$ ... $$)
- Code blocks with language tags (```python)
- Bold for **key findings**
- ✅/❌/🟡 for status indicators

---

## 5. Common Update Scenarios

### Scenario 1: Completed Phase 1 Full Validation

**Files to update:**
1. `RESEARCH_LOG.md` Section 4 (full results)
2. `RESEARCH_LOG.md` Section 7 (progress tracker)
3. `RESEARCH_LOG.md` Section 9 (session log)
4. `supplementary/results/P1_scaling_results.md` (if needed)

**Time estimate:** 30-45 minutes

### Scenario 2: Found Bug and Fixed It

**Files to update:**
1. `RESEARCH_LOG.md` Section 9 only (brief note)
2. No supplementary (unless major impact)

**Time estimate:** 5 minutes

### Scenario 3: Trying New Approach (Experiment)

**Files to create:**
1. `supplementary/experiments/{date}_{description}.md`

**Files to update:**
2. `RESEARCH_LOG.md` Section 9 (reference experiment)

**Time estimate:** 20 minutes (experiment doc) + 5 minutes (log entry)

### Scenario 4: Publishing Paper

**Process:**
1. Copy `RESEARCH_LOG.md` to `manuscript_draft.md`
2. Remove Section 9 (Session Log)
3. Remove "Status: Planned" phases
4. Finalize References section
5. Add author list, affiliations, acknowledgments
6. Format for target journal

**Keep `RESEARCH_LOG.md` as-is for future work!**

---

## 6. Backup and Version Control

### 6.1 Git Commits

**After each significant update:**

```bash
# After session
git add RESEARCH_LOG.md supplementary/
git commit -m "Session 2025-10-10: P1 full cohort validation complete"

# After phase completion
git add RESEARCH_LOG.md tiered_evaluation_v4.py supplementary/
git commit -m "Phase 1 complete: Stride-based scaling integrated"
git tag P1-complete
```

### 6.2 Daily Snapshots

Automated backup script (optional):

```bash
#!/bin/bash
# Save to logs/ with timestamp
DATE=$(date +%Y%m%d_%H%M)
cp RESEARCH_LOG.md logs/snapshots/RESEARCH_LOG_${DATE}.md
```

---

## 7. Quality Checks

Before committing updates, verify:

- [ ] **Numbers match data files** (check JSON outputs)
- [ ] **No TBD or TODO left** (unless marked as Future Work)
- [ ] **Section numbering consistent** (no skipped numbers)
- [ ] **Tables formatted properly** (aligned columns)
- [ ] **Formulas render correctly** (test in Markdown viewer)
- [ ] **File paths accurate** (check supplementary references)
- [ ] **Session Log has date** (YYYY-MM-DD format)
- [ ] **Progress Tracker updated** (if metrics changed)

---

## 8. Collaboration Guidelines

### 8.1 Multi-User Updates

**Rule:** One person updates main sections at a time.

**Workflow:**
1. Person A: "I'm updating Phase 2 results"
2. Others: Only update Session Log (merge-friendly)
3. Person A completes → commits
4. Others pull and continue

### 8.2 Conflict Resolution

If merge conflict occurs:
1. Keep most recent quantitative results
2. Merge session logs chronologically
3. Prefer more detailed version for Methods

---

## 9. Example: Complete Update Workflow

**Scenario:** Just completed Phase 2 implementation and testing.

**Step 1: Update Session Log** (5 min)
```markdown
### Session 2025-10-11 (Total: 6 hours)

**09:00-12:00 | Phase 2: RANSAC Cadence Implementation**
- Implemented `estimate_cadence_ransac()` in `cadence_estimation_v4.py`
- Replaced heuristic blend in `tiered_evaluation_v4.py`
- Added minimum stride interval enforcement (0.6s)

**12:00-15:00 | Phase 2: Validation on Test Cohort**
- Tested on 5 subjects (S1_01, S1_02, S1_03, S1_08, S1_09)
- ICC improved: -0.033 → 0.38 (p = 0.002)
- RMSE reduced: 19.3 → 13.5 steps/min

**Deliverables:**
- `cadence_estimation_v4.py` (new file, 185 lines)
- `P2_cadence_test_results.json`
- `supplementary/methods/P2_ransac_cadence.md`

**Next:** Full cohort validation (n=21)
```

**Step 2: Update Phase 2 Section** (20 min)
- Copy template from Phase 1
- Fill in Method, Results, Discussion
- Add tables with test results

**Step 3: Update Progress Tracker** (2 min)
```markdown
| P2: Cadence Refactor | Cadence ICC | -0.033 | 0.38 | ≥0.30 | 80% | 🟢 **Done** |
```

**Step 4: Create Supplementary** (15 min)
- Write `supplementary/methods/P2_ransac_cadence.md`
- Copy result data to `supplementary/results/P2_cadence_test_results.json`

**Step 5: Commit** (1 min)
```bash
git add RESEARCH_LOG.md cadence_estimation_v4.py supplementary/
git commit -m "Phase 2 complete: RANSAC cadence estimator (ICC: 0.38)"
git tag P2-complete
```

**Total time:** ~43 minutes

---

## 10. Troubleshooting

### Problem: "Can't find where to update X"

**Solution:** Use Section Mapping:
- Aggregate metrics → Section 3 or 7
- Detailed per-subject → `supplementary/results/`
- Algorithm explanation → `supplementary/methods/`
- Daily notes → Section 9 (Session Log)

### Problem: "Section 9 getting too long"

**Solution:** Archive old sessions:
```bash
# Keep only last 3 sessions in main log
# Move older ones to:
supplementary/experiments/session_archive_2025-10.md
```

### Problem: "Don't understand LaTeX formatting"

**Solution:** Copy from existing formulas in Sections 2-4 and modify.

**Test rendering:**
```bash
# VS Code with Markdown Preview
# Or use online: dillinger.io, stackedit.io
```

---

## 11. Quick Reference

| Task | Update Location | Time |
|------|----------------|------|
| Small fix/test | Session Log only | 5 min |
| Phase complete | Phase section + Progress + Session Log | 30-45 min |
| New experiment | Create supplementary/experiments/ + Session Log | 20 min |
| Metric improved | Progress Tracker + Session Log | 5 min |
| Daily work end | Session Log summary | 10 min |

---

## 12. Maintenance Schedule

**Daily (end of session):**
- [ ] Update Session Log with summary
- [ ] Commit to git

**Weekly:**
- [ ] Review Progress Tracker accuracy
- [ ] Archive old session logs if >10 entries

**Phase completion:**
- [ ] Write full Phase section
- [ ] Create supplementary documents
- [ ] Update all references
- [ ] Git tag

---

## Contact

For questions about log maintenance:
- Check existing Phase 1-2 sections as examples
- Review this guide
- Refer to similar academic papers for structure

---

**Document Version:** 1.0
**Maintained By:** Research team
**Last Updated:** 2025-10-10
