#!/usr/bin/env python3
"""Quick batch processing status check."""

import os
from pathlib import Path
from datetime import datetime
import glob

RESULTS_DIR = Path("/data/gait/2d_project/batch_results")
TOTAL_SUBJECTS = 26

# Count completed subjects
completed_plots = glob.glob(str(RESULTS_DIR / "*_comparison.png"))
completed_count = len(completed_plots)

# Extract subject IDs
completed_subjects = sorted([
    Path(p).stem.replace('_comparison', '')
    for p in completed_plots
])

# Estimate completion
if completed_count > 0:
    avg_time_per_subject = 5  # minutes (rough estimate)
    remaining = TOTAL_SUBJECTS - completed_count
    est_remaining_time = remaining * avg_time_per_subject

print("="*70)
print(" BATCH PROCESSING STATUS")
print("="*70)
print(f"\n📊 Progress: {completed_count}/{TOTAL_SUBJECTS} subjects ({100*completed_count/TOTAL_SUBJECTS:.1f}%)")
print(f"⏱️  Estimated time remaining: ~{est_remaining_time} minutes")

if completed_count > 0:
    print(f"\n✅ Completed subjects:")
    for i, subj in enumerate(completed_subjects, 1):
        print(f"   {i:2d}. {subj}")

    # Check if CSV exists
    summary_csv = RESULTS_DIR / "batch_results_summary.csv"
    if summary_csv.exists():
        print(f"\n✓ Summary CSV available: {summary_csv}")

if completed_count < TOTAL_SUBJECTS:
    print(f"\n⏳ Still processing... ({TOTAL_SUBJECTS - completed_count} subjects remaining)")
else:
    print(f"\n🎉 ALL SUBJECTS COMPLETE!")

print("\n" + "="*70)
