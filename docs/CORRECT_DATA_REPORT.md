# Correct Data Source Migration

## Why `processed_new`?
- The legacy `processed/` files contain incorrect cadence values. Example for `S1_01` (`processed/S1_01_info.json`): `right=12`, `left=38`, `average=49` steps/min.
- The new dataset at `data/processed_new/` provides the correct cadence and expanded metadata. Example (`data/processed_new/S1_01_info.json`):
  ```json
  {
    "patient": {
      "right": {
        "cadence_steps_min": 114.802,
        "step_length_cm": 64.436,
        "stride_length_cm": 125.859,
        "forward_velocity_cm_s": 120.241
      },
      "left": {
        "cadence_steps_min": 115.403,
        "step_length_cm": 61.305,
        "stride_length_cm": 125.828,
        "forward_velocity_cm_s": 121.110
      }
    },
    "demographics": {
      "right_strides": 15,
      "left_strides": 11,
      "gait_cycle_timing": {
        "right_stance": 62.114,
        "left_stance": 62.421
      }
    },
    "normal": {
      "right": {
        "cadence_steps_min": 98.434,
        "step_length_cm": 65.231
      }
    }
  }
  ```
- The refreshed structure exposes ground-truth spatial metrics (step length, stride length, forward velocity) plus normal ranges and standard deviations that were absent in the legacy files.

## Implementation Highlights
- Added `tiered_evaluation_v3.py` with default root `Path("/data/gait/data/processed_new")`.
- Updated the temporal pipeline to read the new JSON layout, compute spatial metrics from MediaPipe trajectories, and register them for aggregate statistics.
- Introduced helper tooling:
  - `compare_v1_v2_v3.py` to inspect ICC/RMSE changes across evaluation versions.
  - `diagnose_cadence.py` to analyse MediaPipe cadence against hospital recordings.
- Regenerated results with `tiered_evaluation_report_v3.json` (21 subjects) using the corrected data.

## Next Steps
- Use `TieredGaitEvaluatorV3` for all subsequent analyses and retire the legacy `processed/` directory to avoid regressions.
- Leverage the new ground-truth metrics (step/stride length, velocity, normal ranges) for richer validation and abnormality detection workflows.
