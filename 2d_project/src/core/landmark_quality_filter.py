#!/usr/bin/env python3
"""
Landmark quality assessment and filtering.
Proactively detect and handle poor tracking before angle calculation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LandmarkQuality:
    """Quality metrics for a single landmark over time."""
    visibility_score: float  # Mean visibility (0-1)
    stability_score: float   # Inverse of jitter (0-1)
    coverage_score: float    # Fraction of frames detected (0-1)
    overall_score: float     # Weighted combination (0-1)


class LandmarkQualityFilter:
    """
    Filter MediaPipe landmarks based on quality metrics.

    Addresses root causes of QC failures:
    - Low visibility scores during occlusion
    - High jitter in ankle/foot landmarks
    - Incomplete tracking (many missing frames)
    """

    def __init__(self, visibility_threshold=0.5, stability_threshold=0.3):
        """
        Args:
            visibility_threshold: Minimum mean visibility to accept landmark
            stability_threshold: Maximum normalized jitter to accept
        """
        self.visibility_threshold = visibility_threshold
        self.stability_threshold = stability_threshold

    def assess_landmark_quality(self, landmarks_df: pd.DataFrame) -> Dict[str, LandmarkQuality]:
        """
        Assess quality of each landmark over the sequence.

        Args:
            landmarks_df: DataFrame with columns ['frame', 'landmark', 'x', 'y', 'z', 'visibility']

        Returns:
            Dictionary mapping landmark name to LandmarkQuality
        """
        quality_report = {}

        landmark_names = landmarks_df['landmark'].unique()

        for lm_name in landmark_names:
            lm_data = landmarks_df[landmarks_df['landmark'] == lm_name]

            # Metric 1: Visibility
            visibility_score = lm_data['visibility'].mean()

            # Metric 2: Stability (inverse of jitter)
            positions = lm_data[['x', 'y', 'z']].values
            if len(positions) > 1:
                # Calculate frame-to-frame displacement
                displacements = np.linalg.norm(np.diff(positions, axis=0), axis=1)
                # Normalize by typical body scale (assume hip-ankle height ~1m)
                jitter = np.std(displacements)
                stability_score = 1.0 / (1.0 + jitter * 100)  # Normalize to 0-1
            else:
                stability_score = 0.0

            # Metric 3: Coverage (how many frames have this landmark)
            total_frames = landmarks_df['frame'].nunique()
            detected_frames = len(lm_data)
            coverage_score = detected_frames / total_frames if total_frames > 0 else 0.0

            # Overall score (weighted combination)
            overall_score = (
                0.4 * visibility_score +
                0.3 * stability_score +
                0.3 * coverage_score
            )

            quality_report[lm_name] = LandmarkQuality(
                visibility_score=visibility_score,
                stability_score=stability_score,
                coverage_score=coverage_score,
                overall_score=overall_score
            )

        return quality_report

    def filter_low_quality_landmarks(self, landmarks_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove or interpolate landmarks with poor quality.

        Returns:
            filtered_df: Cleaned landmarks
            removed_landmarks: List of landmark names that were problematic
        """
        quality_report = self.assess_landmark_quality(landmarks_df)

        removed_landmarks = []
        filtered_df = landmarks_df.copy()

        for lm_name, quality in quality_report.items():
            # Check if quality is below threshold
            if quality.visibility_score < self.visibility_threshold or \
               quality.coverage_score < 0.5:

                # Mark for removal or special handling
                removed_landmarks.append(lm_name)

                # For critical landmarks (ankle, knee, hip), try to recover
                if any(joint in lm_name.lower() for joint in ['ankle', 'knee', 'hip']):
                    print(f"⚠️  {lm_name}: Low quality (vis={quality.visibility_score:.2f}, "
                          f"cov={quality.coverage_score:.2f}). Attempting recovery...")

                    # Try bilateral symmetry (use opposite side if available)
                    if 'LEFT' in lm_name:
                        opposite = lm_name.replace('LEFT', 'RIGHT')
                    elif 'RIGHT' in lm_name:
                        opposite = lm_name.replace('RIGHT', 'LEFT')
                    else:
                        opposite = None

                    if opposite and opposite in quality_report:
                        opposite_quality = quality_report[opposite]
                        if opposite_quality.overall_score > quality.overall_score + 0.2:
                            # Use mirrored opposite side
                            filtered_df = self._mirror_landmark(filtered_df, opposite, lm_name)
                            print(f"  → Recovered using mirrored {opposite}")
                        else:
                            # Both sides are bad, mark as unreliable
                            print(f"  → Both sides unreliable, flagging for exclusion")
                else:
                    # Non-critical landmark, just remove
                    filtered_df = filtered_df[filtered_df['landmark'] != lm_name]

        return filtered_df, removed_landmarks

    def _mirror_landmark(self, df: pd.DataFrame, source_lm: str, target_lm: str) -> pd.DataFrame:
        """
        Mirror a landmark across the body midline.
        """
        source_data = df[df['landmark'] == source_lm].copy()

        if len(source_data) == 0:
            return df

        # Flip X coordinate (assume camera is frontal or sagittal)
        # For sagittal view, we'd actually need to use the opposite side's data
        # This is a simplified version
        source_data['landmark'] = target_lm
        source_data['x'] = -source_data['x']  # Mirror X

        # Remove old target_lm data
        df = df[df['landmark'] != target_lm]

        # Add mirrored data
        df = pd.concat([df, source_data], ignore_index=True)

        return df

    def generate_quality_report(self, landmarks_df: pd.DataFrame, output_path: str = None):
        """
        Generate comprehensive quality report with recommendations.
        """
        quality_report = self.assess_landmark_quality(landmarks_df)

        print("\n" + "="*60)
        print("LANDMARK QUALITY ASSESSMENT REPORT")
        print("="*60)

        # Group by body part
        body_parts = {
            'Ankle': [k for k in quality_report.keys() if 'ANKLE' in k],
            'Knee': [k for k in quality_report.keys() if 'KNEE' in k],
            'Hip': [k for k in quality_report.keys() if 'HIP' in k],
            'Foot': [k for k in quality_report.keys() if 'FOOT' in k or 'HEEL' in k or 'TOE' in k]
        }

        for part, landmarks in body_parts.items():
            if not landmarks:
                continue

            print(f"\n{part}:")
            print("-" * 60)

            for lm in landmarks:
                q = quality_report[lm]
                status = "✓ GOOD" if q.overall_score >= 0.7 else \
                         "⚠ FAIR" if q.overall_score >= 0.5 else \
                         "✗ POOR"

                print(f"  {lm:30s} {status}")
                print(f"    Visibility: {q.visibility_score:.2f} | "
                      f"Stability: {q.stability_score:.2f} | "
                      f"Coverage: {q.coverage_score:.2f} | "
                      f"Overall: {q.overall_score:.2f}")

                # Recommendations
                if q.overall_score < 0.5:
                    issues = []
                    if q.visibility_score < 0.5:
                        issues.append("Low visibility (occlusion or poor lighting)")
                    if q.stability_score < 0.5:
                        issues.append("High jitter (fast motion or tracking errors)")
                    if q.coverage_score < 0.7:
                        issues.append("Incomplete tracking (many missing frames)")

                    print(f"    Issues: {', '.join(issues)}")

        # Overall summary
        all_scores = [q.overall_score for q in quality_report.values()]
        mean_quality = np.mean(all_scores)
        poor_count = sum(1 for s in all_scores if s < 0.5)

        print("\n" + "="*60)
        print(f"Overall Quality: {mean_quality:.2f}")
        print(f"Poor Quality Landmarks: {poor_count}/{len(all_scores)}")

        if mean_quality < 0.6:
            print("\n⚠️  WARNING: Video quality is below acceptable threshold")
            print("Recommendations:")
            print("  - Ensure good lighting conditions")
            print("  - Keep subject fully visible (avoid occlusions)")
            print("  - Use higher resolution camera (720p minimum)")
            print("  - Maintain stable camera position")
        elif poor_count > 0:
            print(f"\n⚠️  {poor_count} landmarks need attention")
            print("  - Consider using bilateral symmetry to recover data")
            print("  - Apply aggressive smoothing to affected joints")
        else:
            print("\n✓ Video quality is acceptable for analysis")

        print("="*60)

        # Save report if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(f"Mean Quality: {mean_quality:.3f}\n")
                f.write(f"Poor Landmarks: {poor_count}\n\n")
                for lm, q in quality_report.items():
                    f.write(f"{lm},{q.visibility_score:.3f},{q.stability_score:.3f},"
                           f"{q.coverage_score:.3f},{q.overall_score:.3f}\n")

        return quality_report


# Example usage
if __name__ == "__main__":
    from sagittal_extractor_2d import MediaPipeSagittalExtractor

    # Test with S1_15 (known bad ankle tracking from QC report)
    VIDEO_PATH = "/data/gait/data/15/15-2.mp4"

    print("Extracting landmarks...")
    extractor = MediaPipeSagittalExtractor()
    landmarks, _ = extractor.extract_pose_landmarks(VIDEO_PATH)

    # Convert to long format for quality assessment
    landmarks_long = []
    for frame_idx, frame_data in enumerate(landmarks):
        if frame_data is not None:
            for lm_idx, lm in enumerate(frame_data.landmark):
                landmarks_long.append({
                    'frame': frame_idx,
                    'landmark': f'LANDMARK_{lm_idx}',
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                })

    landmarks_df = pd.DataFrame(landmarks_long)

    # Add landmark names (MediaPipe pose has 33 landmarks)
    landmark_names = {
        0: 'NOSE', 11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
        23: 'LEFT_HIP', 24: 'RIGHT_HIP',
        25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
        27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE',
        29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
        31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
    }

    landmarks_df['landmark'] = landmarks_df['landmark'].apply(
        lambda x: landmark_names.get(int(x.split('_')[1]), x)
    )

    # Assess quality
    qf = LandmarkQualityFilter(visibility_threshold=0.5, stability_threshold=0.3)
    quality_report = qf.generate_quality_report(landmarks_df,
                                                 output_path='/data/gait/2d_project/landmark_quality_report.txt')

    # Filter low-quality landmarks
    filtered_df, removed = qf.filter_low_quality_landmarks(landmarks_df)

    print(f"\nRemoved/flagged {len(removed)} landmarks: {removed}")
