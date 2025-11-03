"""
Right-side joint ICC degradation diagnosis tool.

Analyzes why right-side joints (especially knee) show worse ICC compared to left-side.
Generates comprehensive visualizations and reports.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


class RightSideDiagnostics:
    """Diagnose right-side joint performance issues."""

    def __init__(self, validation_report_path: str):
        """Load validation report and extract test set data."""
        self.report_path = Path(validation_report_path)
        with open(self.report_path, 'r') as f:
            self.report = json.load(f)

        self.test_subjects = self._extract_test_subjects()
        self.output_dir = Path('/data/gait/validation_results_improved')
        self.output_dir.mkdir(exist_ok=True)

    def _extract_test_subjects(self) -> List[Dict]:
        """Extract test split subjects from report."""
        subjects = self.report.get('subjects', [])
        test_subjects = [s for s in subjects if s.get('split') == 'test']
        return test_subjects

    def analyze_left_right_symmetry(self) -> Dict:
        """Compare left vs right joint angles for symmetry."""
        results = {}

        for joint_base in ['knee', 'hip', 'ankle']:
            left_key = f'left_{joint_base}'
            right_key = f'right_{joint_base}'

            left_angles = []
            right_angles = []

            for subject in self.test_subjects:
                # Extract MediaPipe converted angles
                if left_key in subject.get('mp_angles', {}):
                    left_angles.append(subject['mp_angles'][left_key])
                if right_key in subject.get('mp_angles', {}):
                    right_angles.append(subject['mp_angles'][right_key])

            if left_angles and right_angles:
                left_concat = np.concatenate(left_angles)
                right_concat = np.concatenate(right_angles)

                # Handle NaN
                valid = ~(np.isnan(left_concat) | np.isnan(right_concat))
                if valid.sum() > 10:
                    corr, p_val = pearsonr(left_concat[valid], right_concat[valid])
                    results[joint_base] = {
                        'correlation': corr,
                        'p_value': p_val,
                        'left_mean': float(np.nanmean(left_concat)),
                        'right_mean': float(np.nanmean(right_concat)),
                        'left_std': float(np.nanstd(left_concat)),
                        'right_std': float(np.nanstd(right_concat)),
                        'asymmetry': float(np.abs(np.nanmean(left_concat) - np.nanmean(right_concat))),
                    }

        return results

    def analyze_visibility(self) -> Dict:
        """Compare landmark visibility between left and right sides."""
        visibility_data = {}

        for subject in self.test_subjects:
            subject_id = subject.get('subject_id', 'unknown')
            vis = {}

            # Extract visibility from landmarks if available
            landmarks = subject.get('mp_landmarks', {})
            for side in ['left', 'right']:
                for joint in ['knee', 'hip', 'ankle', 'heel']:
                    key = f'visibility_{side}_{joint}'
                    if key in landmarks:
                        vis[f'{side}_{joint}'] = float(np.mean(landmarks[key]))

            if vis:
                visibility_data[subject_id] = vis

        return visibility_data

    def plot_timeseries_comparison(self, pdf: PdfPages):
        """Plot MediaPipe vs Hospital angles for left/right knee."""
        for subject in self.test_subjects:
            subject_id = subject.get('subject_id', 'unknown')

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f'Subject {subject_id}: Left vs Right Knee Angles', fontsize=16)

            # Left knee
            if 'left_knee' in subject.get('mp_angles', {}):
                mp_left = subject['mp_angles']['left_knee']
                hosp_left = subject.get('hospital_angles', {}).get('left_knee', [])

                x = np.arange(len(mp_left))
                axes[0, 0].plot(x, mp_left, label='MediaPipe', color='blue', alpha=0.7)
                if len(hosp_left) == len(mp_left):
                    axes[0, 0].plot(x, hosp_left, label='Hospital', color='red', alpha=0.7)
                axes[0, 0].set_title('Left Knee Time Series')
                axes[0, 0].set_xlabel('Gait Cycle %')
                axes[0, 0].set_ylabel('Angle (degrees)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)

            # Right knee
            if 'right_knee' in subject.get('mp_angles', {}):
                mp_right = subject['mp_angles']['right_knee']
                hosp_right = subject.get('hospital_angles', {}).get('right_knee', [])

                x = np.arange(len(mp_right))
                axes[0, 1].plot(x, mp_right, label='MediaPipe', color='blue', alpha=0.7)
                if len(hosp_right) == len(mp_right):
                    axes[0, 1].plot(x, hosp_right, label='Hospital', color='red', alpha=0.7)
                axes[0, 1].set_title('Right Knee Time Series')
                axes[0, 1].set_xlabel('Gait Cycle %')
                axes[0, 1].set_ylabel('Angle (degrees)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Scatter: Left knee
            if 'left_knee' in subject.get('mp_angles', {}):
                mp_left = subject['mp_angles']['left_knee']
                hosp_left = subject.get('hospital_angles', {}).get('left_knee', [])
                if len(hosp_left) == len(mp_left):
                    valid = ~(np.isnan(mp_left) | np.isnan(hosp_left))
                    if valid.sum() > 0:
                        axes[1, 0].scatter(hosp_left[valid], mp_left[valid], alpha=0.5, s=10)
                        # Identity line
                        lim = [min(axes[1, 0].get_xlim()[0], axes[1, 0].get_ylim()[0]),
                               max(axes[1, 0].get_xlim()[1], axes[1, 0].get_ylim()[1])]
                        axes[1, 0].plot(lim, lim, 'k--', alpha=0.5, label='Identity')
                        axes[1, 0].set_title('Left Knee: Hospital vs MediaPipe')
                        axes[1, 0].set_xlabel('Hospital Angle (degrees)')
                        axes[1, 0].set_ylabel('MediaPipe Angle (degrees)')
                        axes[1, 0].legend()
                        axes[1, 0].grid(True, alpha=0.3)

            # Scatter: Right knee
            if 'right_knee' in subject.get('mp_angles', {}):
                mp_right = subject['mp_angles']['right_knee']
                hosp_right = subject.get('hospital_angles', {}).get('right_knee', [])
                if len(hosp_right) == len(mp_right):
                    valid = ~(np.isnan(mp_right) | np.isnan(hosp_right))
                    if valid.sum() > 0:
                        axes[1, 1].scatter(hosp_right[valid], mp_right[valid], alpha=0.5, s=10)
                        # Identity line
                        lim = [min(axes[1, 1].get_xlim()[0], axes[1, 1].get_ylim()[0]),
                               max(axes[1, 1].get_xlim()[1], axes[1, 1].get_ylim()[1])]
                        axes[1, 1].plot(lim, lim, 'k--', alpha=0.5, label='Identity')
                        axes[1, 1].set_title('Right Knee: Hospital vs MediaPipe')
                        axes[1, 1].set_xlabel('Hospital Angle (degrees)')
                        axes[1, 1].set_ylabel('MediaPipe Angle (degrees)')
                        axes[1, 1].legend()
                        axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    def plot_heatmap(self, pdf: PdfPages):
        """Plot heatmap of angles across gait cycle for all test subjects."""
        for side in ['left', 'right']:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f'{side.capitalize()} Side: MediaPipe Angles Heatmap', fontsize=16)

            for idx, joint in enumerate(['knee', 'hip', 'ankle']):
                joint_key = f'{side}_{joint}'
                angles_matrix = []
                labels = []

                for subject in self.test_subjects:
                    if joint_key in subject.get('mp_angles', {}):
                        angles = subject['mp_angles'][joint_key]
                        if len(angles) == 101:  # Normalized gait cycle
                            angles_matrix.append(angles)
                            labels.append(subject.get('subject_id', 'unknown'))

                if angles_matrix:
                    angles_matrix = np.array(angles_matrix)

                    im = axes[idx].imshow(angles_matrix, aspect='auto', cmap='viridis')
                    axes[idx].set_title(f'{joint.capitalize()}')
                    axes[idx].set_xlabel('Gait Cycle %')
                    axes[idx].set_ylabel('Subject')
                    axes[idx].set_yticks(range(len(labels)))
                    axes[idx].set_yticklabels(labels)

                    # Colorbar
                    plt.colorbar(im, ax=axes[idx], label='Angle (degrees)')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    def plot_error_distribution(self, pdf: PdfPages):
        """Plot error distribution (MP - Hospital) for left vs right."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Error Distribution: MediaPipe - Hospital', fontsize=16)

        for col_idx, joint in enumerate(['knee', 'hip', 'ankle']):
            for row_idx, side in enumerate(['left', 'right']):
                joint_key = f'{side}_{joint}'
                errors = []

                for subject in self.test_subjects:
                    mp = subject.get('mp_angles', {}).get(joint_key, [])
                    hosp = subject.get('hospital_angles', {}).get(joint_key, [])

                    if len(mp) == len(hosp) and len(mp) > 0:
                        mp_arr = np.array(mp)
                        hosp_arr = np.array(hosp)
                        valid = ~(np.isnan(mp_arr) | np.isnan(hosp_arr))
                        if valid.sum() > 0:
                            err = mp_arr[valid] - hosp_arr[valid]
                            errors.extend(err.tolist())

                if errors:
                    axes[row_idx, col_idx].hist(errors, bins=50, alpha=0.7, edgecolor='black')
                    axes[row_idx, col_idx].axvline(0, color='red', linestyle='--', linewidth=2)
                    axes[row_idx, col_idx].axvline(np.mean(errors), color='blue',
                                                    linestyle='--', linewidth=2,
                                                    label=f'Mean={np.mean(errors):.1f}°')
                    axes[row_idx, col_idx].set_title(f'{side.capitalize()} {joint.capitalize()}')
                    axes[row_idx, col_idx].set_xlabel('Error (degrees)')
                    axes[row_idx, col_idx].set_ylabel('Frequency')
                    axes[row_idx, col_idx].legend()
                    axes[row_idx, col_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    def generate_report(self) -> str:
        """Generate comprehensive diagnosis report."""
        symmetry = self.analyze_left_right_symmetry()
        visibility = self.analyze_visibility()

        # Get ICC comparison from summary
        summary = self.report.get('summary_statistics', {})

        report_lines = [
            "# Right-Side Joint ICC Degradation Diagnosis",
            "",
            "## Test Set ICC Comparison",
            "",
            "| Joint | Left ICC | Right ICC | Δ (R-L) |",
            "|-------|----------|-----------|---------|"
        ]

        for joint in ['knee', 'hip', 'ankle']:
            left_key = f'left_{joint}'
            right_key = f'right_{joint}'

            left_icc = summary.get(left_key, {}).get('test', {}).get('icc', {})
            right_icc = summary.get(right_key, {}).get('test', {}).get('icc', {})

            if isinstance(left_icc, dict) and isinstance(right_icc, dict):
                left_val = left_icc.get('mean', float('nan'))
                right_val = right_icc.get('mean', float('nan'))
                delta = right_val - left_val
                report_lines.append(
                    f"| {joint.capitalize()} | {left_val:.3f} | {right_val:.3f} | {delta:+.3f} |"
                )

        report_lines.extend([
            "",
            "## Left-Right Symmetry Analysis",
            "",
            "| Joint | Correlation | Left Mean±SD | Right Mean±SD | Asymmetry |",
            "|-------|-------------|--------------|---------------|-----------|"
        ])

        for joint, data in symmetry.items():
            report_lines.append(
                f"| {joint.capitalize()} | "
                f"{data['correlation']:.3f} (p={data['p_value']:.3f}) | "
                f"{data['left_mean']:.1f}±{data['left_std']:.1f}° | "
                f"{data['right_mean']:.1f}±{data['right_std']:.1f}° | "
                f"{data['asymmetry']:.1f}° |"
            )

        report_lines.extend([
            "",
            "## Visibility Analysis",
            "",
            "Average landmark visibility by side and joint:",
            ""
        ])

        if visibility:
            # Aggregate visibility
            vis_summary = {}
            for subject_vis in visibility.values():
                for key, val in subject_vis.items():
                    if key not in vis_summary:
                        vis_summary[key] = []
                    vis_summary[key].append(val)

            report_lines.append("| Landmark | Mean Visibility | Std |")
            report_lines.append("|----------|-----------------|-----|")

            for key in sorted(vis_summary.keys()):
                vals = vis_summary[key]
                report_lines.append(
                    f"| {key} | {np.mean(vals):.3f} | {np.std(vals):.3f} |"
                )

        report_lines.extend([
            "",
            "## Key Findings",
            "",
            "### 1. ICC Degradation Pattern",
        ])

        # Analyze pattern
        knee_delta = None
        for joint in ['knee']:
            left_key = f'left_{joint}'
            right_key = f'right_{joint}'
            left_icc = summary.get(left_key, {}).get('test', {}).get('icc', {})
            right_icc = summary.get(right_key, {}).get('test', {}).get('icc', {})
            if isinstance(left_icc, dict) and isinstance(right_icc, dict):
                knee_delta = right_icc.get('mean', 0) - left_icc.get('mean', 0)

        if knee_delta is not None and knee_delta < -0.1:
            report_lines.append("- **Right knee ICC significantly worse than left** (Δ < -0.1)")
            report_lines.append("- Possible causes:")
            report_lines.append("  - Camera angle favoring left side (visibility asymmetry)")
            report_lines.append("  - Subject gait asymmetry (pathological or compensatory)")
            report_lines.append("  - Right landmark detection noise")

        report_lines.extend([
            "",
            "### 2. Symmetry Assessment",
        ])

        if 'knee' in symmetry:
            corr = symmetry['knee']['correlation']
            if corr < 0.5:
                report_lines.append(f"- **Low left-right knee correlation** (r={corr:.3f})")
                report_lines.append("  → Asymmetric gait pattern or measurement error")
            elif corr >= 0.8:
                report_lines.append(f"- **High left-right knee correlation** (r={corr:.3f})")
                report_lines.append("  → Symmetric gait, issue likely in conversion method")

        report_lines.extend([
            "",
            "### 3. Recommendations",
            "",
            "**Immediate Actions:**"
        ])

        if knee_delta is not None and knee_delta < -0.1:
            report_lines.append("1. **Use left-side data primarily** for angle conversion training")
            report_lines.append("2. **Investigate right landmark visibility** - may need filtering")
            report_lines.append("3. **Test side-specific conversion parameters** (left vs right)")

        report_lines.extend([
            "",
            "**Long-term Improvements:**",
            "1. Multi-view camera setup to reduce side bias",
            "2. Bilateral symmetry constraints in angle conversion",
            "3. Outlier detection for asymmetric frames",
            "",
            "---",
            f"*Report generated from: {self.report_path}*",
            f"*Test subjects analyzed: {len(self.test_subjects)}*"
        ])

        return "\n".join(report_lines)

    def run_full_diagnosis(self):
        """Run complete diagnostic analysis and save outputs."""
        print("Running right-side diagnostics...")

        # Generate PDF with all plots
        pdf_path = self.output_dir / 'right_side_diagnosis.pdf'
        with PdfPages(pdf_path) as pdf:
            print("  - Generating time series comparison plots...")
            self.plot_timeseries_comparison(pdf)

            print("  - Generating heatmaps...")
            self.plot_heatmap(pdf)

            print("  - Generating error distribution plots...")
            self.plot_error_distribution(pdf)

        print(f"✓ PDF saved: {pdf_path}")

        # Generate markdown report
        report_text = self.generate_report()
        report_path = self.output_dir / 'RIGHT_SIDE_DIAGNOSIS.md'
        report_path.write_text(report_text, encoding='utf-8')
        print(f"✓ Report saved: {report_path}")

        # Save JSON summary
        summary_data = {
            'symmetry_analysis': self.analyze_left_right_symmetry(),
            'visibility_analysis': self.analyze_visibility(),
        }
        json_path = self.output_dir / 'right_side_diagnosis.json'
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"✓ JSON saved: {json_path}")

        print("\n" + "="*60)
        print("DIAGNOSIS COMPLETE")
        print("="*60)
        print(f"\nView report: {report_path}")
        print(f"View plots: {pdf_path}")


def main():
    """Main entry point."""
    report_path = '/data/gait/validation_results_improved/improved_validation_report.json'

    diagnostics = RightSideDiagnostics(report_path)
    diagnostics.run_full_diagnosis()


if __name__ == '__main__':
    main()
