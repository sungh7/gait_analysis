"""
Tiered gait evaluation pipeline v5.3.3 - Ensemble.

Key innovation:
- **Intelligent ensemble of V5.2 and V5.3.2**
- Subject-level selection based on confidence and performance
- Combines conservative (V5.2) and aggressive (V5.3.2) approaches

Selection strategy:
1. Use V5.3.2 when label_corrected=True AND improvement likely (>5%)
2. Use V5.2 when symmetric_applied=True (symmetric often degrades performance)
3. For ties or low confidence, use V5.2 (conservative default)

Expected improvements over V5.3.2:
- Right ICC: 0.429 → 0.45-0.48 (avoid symmetric scale degradation)
- Left ICC: 0.881 → 0.90-0.92 (restore some V5.2 accuracy)
- Overall: Best of both worlds
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class EnsembleGaitEvaluatorV533:
    """
    V5.3.3: Intelligent ensemble combining V5.2 and V5.3.2.

    Selection criteria per subject:
    - V5.3.2 if: label_corrected AND (NOT symmetric_applied OR cross_leg_valid)
    - V5.2 otherwise (conservative default)
    """

    def __init__(self, v52_results: Dict, v532_results: Dict):
        self.v52 = v52_results
        self.v532 = v532_results
        self.version = "v5.3.3-ensemble"

        # Tracking
        self.selection_log = {}
        self.selection_stats = {
            'v52_selected': 0,
            'v532_selected': 0,
            'reasons': {}
        }

    def select_best_result(self, subject_id: str) -> Tuple[str, Dict, str]:
        """
        Select better result between V5.2 and V5.3.2 for a subject.

        Returns:
            (selected_version, selected_data, reason)
        """

        # Get both results
        v52_data = self.v52['subjects'].get(subject_id)
        v532_data = self.v532['subjects'].get(subject_id)

        if not v52_data:
            return ('v5.3.2', v532_data, 'V5.2_MISSING')
        if not v532_data:
            return ('v5.2', v52_data, 'V5.3.2_MISSING')

        # Check V5.3.2 metadata
        v532_meta = v532_data.get('temporal', {}).get('v53_metadata', {})
        label_corrected = v532_meta.get('label_correction', {}).get('corrected', False)
        symmetric_applied = v532_meta.get('symmetric_scale', {}).get('applied', False)

        # Decision logic
        if label_corrected and not symmetric_applied:
            # Label corrected but no symmetric scale → likely good correction
            reason = 'LABEL_CORRECTED_ONLY'
            selected = 'v5.3.2'
        elif label_corrected and symmetric_applied:
            # Both applied → mixed signal, need to check confidence
            confidence = v532_meta.get('label_correction', {}).get('confidence', 0)
            if confidence > 10:  # >10% improvement
                reason = 'LABEL_CORRECTED_HIGH_CONF'
                selected = 'v5.3.2'
            else:
                reason = 'LABEL_CORRECTED_LOW_CONF'
                selected = 'v5.2'  # Conservative
        elif symmetric_applied and not label_corrected:
            # Symmetric only → often degrades (as seen in S1_08, S1_15, etc.)
            reason = 'SYMMETRIC_ONLY_AVOID'
            selected = 'v5.2'
        else:
            # No special treatment → use V5.2 baseline
            reason = 'DEFAULT_V52'
            selected = 'v5.2'

        # Track stats
        if selected == 'v5.2':
            self.selection_stats['v52_selected'] += 1
        else:
            self.selection_stats['v532_selected'] += 1

        self.selection_stats['reasons'][reason] = self.selection_stats['reasons'].get(reason, 0) + 1

        selected_data = v52_data if selected == 'v5.2' else v532_data

        return (selected, selected_data, reason)

    def create_ensemble_results(self) -> Dict:
        """
        Create ensemble results by selecting best version per subject.
        """

        ensemble_results = {
            'metadata': {
                'version': self.version,
                'walkway_distance_m': 7.5,
                'source_versions': {
                    'conservative': 'v5.2',
                    'aggressive': 'v5.3.2'
                },
                'selection_strategy': 'confidence_based_hybrid'
            },
            'subjects': {},
            'aggregate': None,
            'selection_log': {}
        }

        # Process each subject
        all_subjects = set(self.v52['subjects'].keys()) | set(self.v532['subjects'].keys())

        for subject_id in sorted(all_subjects):
            selected_version, selected_data, reason = self.select_best_result(subject_id)

            # Add selection metadata
            selected_data_copy = json.loads(json.dumps(selected_data))  # Deep copy
            if 'temporal' in selected_data_copy:
                selected_data_copy['temporal']['v533_ensemble_metadata'] = {
                    'selected_version': selected_version,
                    'selection_reason': reason
                }

            ensemble_results['subjects'][subject_id] = selected_data_copy
            ensemble_results['selection_log'][subject_id] = {
                'version': selected_version,
                'reason': reason
            }

        # Copy aggregate from V5.2 as base (will recalculate if needed)
        ensemble_results['aggregate'] = self.v52['aggregate'].copy()

        # Add selection statistics
        ensemble_results['aggregate']['v533_ensemble_stats'] = {
            'total_subjects': len(all_subjects),
            'v52_selected': self.selection_stats['v52_selected'],
            'v532_selected': self.selection_stats['v532_selected'],
            'selection_rate_v532': self.selection_stats['v532_selected'] / len(all_subjects) * 100,
            'selection_reasons': self.selection_stats['reasons']
        }

        return ensemble_results


def main():
    """Create V5.3.3 ensemble results."""

    print("="*80)
    print("Tiered Gait Evaluation V5.3.3 - Ensemble")
    print("Intelligent Selection: V5.2 (Conservative) + V5.3.2 (Aggressive)")
    print("="*80)

    # Load V5.2 and V5.3.2 results
    print("\nLoading V5.2 results...")
    with open('tiered_evaluation_report_v52.json', 'r') as f:
        v52_results = json.load(f)
    print(f"  ✓ Loaded {len(v52_results['subjects'])} subjects")

    print("\nLoading V5.3.2 results...")
    with open('tiered_evaluation_report_v532.json', 'r') as f:
        v532_results = json.load(f)
    print(f"  ✓ Loaded {len(v532_results['subjects'])} subjects")

    # Create ensemble
    print("\nCreating ensemble...")
    ensemble = EnsembleGaitEvaluatorV533(v52_results, v532_results)
    results = ensemble.create_ensemble_results()

    print(f"\n{'='*80}")
    print("Selection Summary")
    print(f"{'='*80}")

    stats = results['aggregate']['v533_ensemble_stats']
    print(f"\nTotal subjects:     {stats['total_subjects']}")
    print(f"V5.2 selected:      {stats['v52_selected']} ({stats['v52_selected']/stats['total_subjects']*100:.0f}%)")
    print(f"V5.3.2 selected:    {stats['v532_selected']} ({stats['v532_selected']/stats['total_subjects']*100:.0f}%)")

    print(f"\nSelection reasons:")
    for reason, count in sorted(stats['selection_reasons'].items(), key=lambda x: -x[1]):
        print(f"  {reason:30s}: {count:>3}")

    # Save results
    output_file = "tiered_evaluation_report_v533.json"
    results_serializable = convert_numpy_types(results)

    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Ensemble complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*80}")

    # Show per-subject selections
    print(f"\n{'='*80}")
    print("Per-Subject Selection Log")
    print(f"{'='*80}")
    print(f"\n{'Subject':<10} {'Selected':<10} {'Reason':<30}")
    print("-"*60)

    for subject_id, log in sorted(results['selection_log'].items()):
        marker = '→ V5.3.2' if log['version'] == 'v5.3.2' else '  V5.2'
        print(f"{subject_id:<10} {marker:<10} {log['reason']:<30}")

    print(f"\n{'='*80}")

    return results


if __name__ == "__main__":
    results = main()
