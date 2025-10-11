"""
Statistical Parametric Mapping (SPM) toolkit for gait cycle analysis.

Implements paired t-tests across the gait cycle with multiple-comparison
corrections (Benjamini-Hochberg FDR, optional Bonferroni) and provides
normality/independence diagnostics for per-point residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d


@dataclass
class StatisticalDiagnostics:
    shapiro_pvalue: float
    durbin_watson: float
    notes: str


class SPMAnalyzer:
    def __init__(self, alpha: float = 0.05, smooth_sigma: Optional[float] = None) -> None:
        self.alpha = alpha
        self.smooth_sigma = smooth_sigma

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------
    def paired_ttest_spm(self, group1: np.ndarray, group2: np.ndarray, use_permutation: bool = False, n_permutations: int = 10000) -> Dict[str, object]:
        """
        Perform SPM analysis with automatic fallback to permutation test.

        Args:
            group1: Array of shape (n_subjects, n_points)
            group2: Array of shape (n_subjects, n_points)
            use_permutation: If True, use permutation test. If False, check normality first.
            n_permutations: Number of permutations for permutation test

        Returns:
            Dictionary with test results
        """
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)
        if group1.shape != group2.shape:
            raise ValueError("Input arrays must have identical shape (n_subjects, n_points)")
        if group1.ndim != 2:
            raise ValueError("Input arrays must be 2D")

        # Check normality if not forcing permutation
        if not use_permutation:
            differences_test = group1 - group2
            residuals = differences_test.reshape(-1)
            residuals = residuals[~np.isnan(residuals)]

            if len(residuals) >= 3:
                shapiro_p = stats.shapiro(residuals).pvalue
                if shapiro_p < 0.05:
                    print(f"  Normality violated (Shapiro p={shapiro_p:.4f}), using permutation test...")
                    return self.permutation_spm(group1, group2, n_permutations)
                else:
                    print(f"  Normality OK (Shapiro p={shapiro_p:.4f}), using parametric test...")
            else:
                print("  Insufficient data for normality test, using parametric test...")

        differences = group1 - group2
        if self.smooth_sigma is not None:
            differences = gaussian_filter1d(differences, sigma=self.smooth_sigma, axis=1)

        mean_diff = np.mean(differences, axis=0)
        std_diff = np.std(differences, axis=0, ddof=1)
        n_subjects = group1.shape[0]
        se_diff = std_diff / np.sqrt(n_subjects)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stats = np.where(se_diff > 0, mean_diff / se_diff, 0.0)

        df = n_subjects - 1
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

        fdr_mask, fdr_threshold = self._benjamini_hochberg(p_values, self.alpha)
        bonferroni_threshold = self.alpha / len(p_values)
        bonferroni_mask = p_values < bonferroni_threshold

        clusters = self._find_clusters(fdr_mask, t_stats, p_values)
        diagnostics = self._diagnostics(differences)
        cohens_d = np.divide(mean_diff, std_diff, out=np.zeros_like(mean_diff), where=std_diff > 0)

        return {
            't_stats': t_stats,
            'p_values': p_values,
            'fdr_mask': fdr_mask,
            'fdr_threshold': fdr_threshold,
            'bonferroni_mask': bonferroni_mask,
            'bonferroni_threshold': bonferroni_threshold,
            'clusters': clusters,
            'mean_difference': mean_diff,
            'cohens_d': cohens_d,
            'diagnostics': diagnostics.__dict__,
        }

    # ------------------------------------------------------------------
    # Multiple comparison correction
    # ------------------------------------------------------------------
    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray, alpha: float) -> (np.ndarray, float):
        p_values = np.asarray(p_values)
        n = len(p_values)
        order = np.argsort(p_values)
        sorted_p = p_values[order]
        thresholds = alpha * (np.arange(1, n + 1) / n)
        below = sorted_p <= thresholds
        if not np.any(below):
            return np.zeros_like(p_values, dtype=bool), thresholds[-1]
        max_idx = np.max(np.where(below)[0])
        critical_p = sorted_p[max_idx]
        mask = p_values <= critical_p
        return mask, critical_p

    # ------------------------------------------------------------------
    # Cluster identification
    # ------------------------------------------------------------------
    @staticmethod
    def _find_clusters(significant: np.ndarray, t_stats: np.ndarray, p_values: np.ndarray) -> List[Dict[str, float]]:
        clusters: List[Dict[str, float]] = []
        in_cluster = False
        start = 0
        for idx, flag in enumerate(significant):
            if flag and not in_cluster:
                in_cluster = True
                start = idx
            elif not flag and in_cluster:
                clusters.append(SPMAnalyzer._cluster_info(start, idx - 1, t_stats, p_values))
                in_cluster = False
        if in_cluster:
            clusters.append(SPMAnalyzer._cluster_info(start, len(significant) - 1, t_stats, p_values))
        return clusters

    @staticmethod
    def _cluster_info(start: int, end: int, t_stats: np.ndarray, p_values: np.ndarray) -> Dict[str, float]:
        indices = slice(start, end + 1)
        cluster_t = t_stats[indices]
        peak_idx = start + int(np.argmax(np.abs(cluster_t)))
        return {
            'start_gait_cycle': float(start),
            'end_gait_cycle': float(end),
            'duration': float(end - start + 1),
            'peak_t': float(t_stats[peak_idx]),
            'peak_p': float(p_values[peak_idx]),
            'peak_gait_cycle': float(peak_idx),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _diagnostics(differences: np.ndarray) -> StatisticalDiagnostics:
        residuals = differences
        flattened = residuals.reshape(-1)
        flattened = flattened[~np.isnan(flattened)]
        if len(flattened) < 3:
            return StatisticalDiagnostics(shapiro_pvalue=float('nan'), durbin_watson=float('nan'), notes='Insufficient data')
        shapiro_p = stats.shapiro(flattened).pvalue
        diff = np.diff(flattened)
        denominator = np.sum(flattened ** 2)
        dw = np.sum(diff ** 2) / denominator if denominator > 0 else float('nan')
        notes = []
        if shapiro_p < 0.05:
            notes.append('Non-normal residuals')
        if dw < 1.5 or dw > 2.5:
            notes.append('Autocorrelation detected')
        return StatisticalDiagnostics(shapiro_pvalue=float(shapiro_p), durbin_watson=float(dw), notes=', '.join(notes) if notes else 'OK')

    # ------------------------------------------------------------------
    # Summary and persistence
    # ------------------------------------------------------------------
    def spm_summary(self, spm_result: Dict[str, object]) -> Dict[str, object]:
        significant = spm_result['fdr_mask']
        n_points = len(significant)
        n_significant = int(np.sum(significant))
        pct_significant = n_significant / n_points * 100 if n_points else 0.0
        summary = {
            'total_points': n_points,
            'significant_points_fdr': n_significant,
            'percent_significant_fdr': pct_significant,
            'num_clusters': len(spm_result['clusters']),
            'max_absolute_t': float(np.nanmax(np.abs(spm_result['t_stats']))),
            'mean_cohens_d': float(np.nanmean(np.abs(spm_result['cohens_d']))),
        }
        if pct_significant < 10:
            summary['interpretation'] = 'Excellent agreement (<10% significant)'
        elif pct_significant < 30:
            summary['interpretation'] = 'Good agreement (<30% significant)'
        elif pct_significant < 50:
            summary['interpretation'] = 'Fair agreement (<50% significant)'
        else:
            summary['interpretation'] = 'Poor agreement (>50% significant)'
        return summary

    def visualize_spm(self, spm_result: Dict[str, object], mp_mean: np.ndarray, hosp_mean: np.ndarray, save_path: Optional[Path] = None):
        """Generate a three-panel SPM plot highlighting significant regions."""
        import matplotlib.pyplot as plt

        gait_cycle = np.arange(len(spm_result['t_stats']))
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        ax = axes[0]
        ax.plot(gait_cycle, mp_mean, label='MediaPipe', color='tab:blue', linewidth=2)
        ax.plot(gait_cycle, hosp_mean, label='Hospital', color='tab:red', linewidth=2)
        if spm_result['clusters']:
            for idx, cluster in enumerate(spm_result['clusters']):
                ax.axvspan(cluster['start_gait_cycle'], cluster['end_gait_cycle'], color='gold', alpha=0.3, label='FDR significant' if idx == 0 else None)
        ax.set_ylabel('Angle (degrees)')
        ax.set_xlabel('Gait Cycle (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(gait_cycle, spm_result['t_stats'], color='black', linewidth=2)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(spm_result['fdr_threshold'], color='tab:green', linestyle='--', label=f"FDR threshold {spm_result['fdr_threshold']:.3f}")
        ax.axhline(-spm_result['fdr_threshold'], color='tab:green', linestyle='--')
        ax.axhline(spm_result['bonferroni_threshold'], color='tab:orange', linestyle=':', label=f"Bonferroni {spm_result['bonferroni_threshold']:.4f}")
        ax.axhline(-spm_result['bonferroni_threshold'], color='tab:orange', linestyle=':')
        if spm_result['clusters']:
            for cluster in spm_result['clusters']:
                ax.axvspan(cluster['start_gait_cycle'], cluster['end_gait_cycle'], color='gold', alpha=0.3)
        ax.set_ylabel('t-statistic')
        ax.set_xlabel('Gait Cycle (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[2]
        ax.semilogy(gait_cycle, spm_result['p_values'], color='black', linewidth=2)
        ax.axhline(self.alpha, color='tab:red', linestyle='--', label=f"alpha = {self.alpha}")
        ax.set_ylabel('p-value (log scale)')
        ax.set_xlabel('Gait Cycle (%)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def save_result(self, spm_result: Dict[str, object], path: Path) -> None:
        np.savez_compressed(path, **{k: v for k, v in spm_result.items() if isinstance(v, np.ndarray)})

    # ------------------------------------------------------------------
    # Permutation test (non-parametric alternative)
    # ------------------------------------------------------------------
    def permutation_spm(self, group1: np.ndarray, group2: np.ndarray, n_permutations: int = 10000) -> Dict[str, object]:
        """
        Permutation-based SPM analysis (non-parametric).

        Uses sign-flipping for paired data to generate null distribution.
        Family-wise error rate is controlled by comparing observed max |t| to null distribution.

        Args:
            group1: Array of shape (n_subjects, n_points)
            group2: Array of shape (n_subjects, n_points)
            n_permutations: Number of permutations

        Returns:
            Dictionary with test results
        """
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)

        if group1.shape != group2.shape:
            raise ValueError("Input arrays must have identical shape")

        differences = group1 - group2
        if self.smooth_sigma is not None:
            differences = gaussian_filter1d(differences, sigma=self.smooth_sigma, axis=1)

        n_subjects, n_points = differences.shape

        # Observed t-statistics
        mean_diff = np.mean(differences, axis=0)
        std_diff = np.std(differences, axis=0, ddof=1)
        se_diff = std_diff / np.sqrt(n_subjects)

        with np.errstate(divide='ignore', invalid='ignore'):
            t_obs = np.where(se_diff > 0, mean_diff / se_diff, 0.0)

        # Permutation distribution
        print(f"  Running {n_permutations} permutations...")
        null_max_t = np.zeros(n_permutations)

        for perm_idx in range(n_permutations):
            # Random sign flips for each subject (preserves pairing)
            signs = np.random.choice([-1, 1], size=n_subjects)
            diff_perm = signs[:, np.newaxis] * differences

            mean_perm = np.mean(diff_perm, axis=0)
            std_perm = np.std(diff_perm, axis=0, ddof=1)
            se_perm = std_perm / np.sqrt(n_subjects)

            with np.errstate(divide='ignore', invalid='ignore'):
                t_perm = np.where(se_perm > 0, mean_perm / se_perm, 0.0)

            # Store max absolute t-statistic (for family-wise error control)
            null_max_t[perm_idx] = np.max(np.abs(t_perm))

        # Critical threshold (family-wise alpha)
        threshold = np.percentile(null_max_t, (1 - self.alpha) * 100)

        # Significant points
        perm_mask = np.abs(t_obs) > threshold

        # Empirical p-values
        p_values_perm = np.zeros(n_points)
        for i in range(n_points):
            p_values_perm[i] = np.mean(null_max_t >= np.abs(t_obs[i]))

        # Clusters
        clusters = self._find_clusters(perm_mask, t_obs, p_values_perm)

        # Diagnostics
        diagnostics = self._diagnostics(differences)

        # Effect size
        cohens_d = np.divide(mean_diff, std_diff, out=np.zeros_like(mean_diff), where=std_diff > 0)

        print(f"  Permutation complete. Threshold: {threshold:.3f}")

        return {
            't_stats': t_obs,
            'p_values': p_values_perm,
            'permutation_mask': perm_mask,
            'permutation_threshold': threshold,
            'null_distribution': null_max_t,
            'clusters': clusters,
            'mean_difference': mean_diff,
            'cohens_d': cohens_d,
            'diagnostics': diagnostics.__dict__,
            'method': 'permutation',
            'n_permutations': n_permutations,
        }


def main() -> None:
    print("SPMAnalyzer is a library module; use run_improved_validation.py for execution context.")


if __name__ == "__main__":
    main()
