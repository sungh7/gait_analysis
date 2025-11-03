#!/usr/bin/env python3
"""
Extract ALL GAVD Views (Front, Back, Left Side, Right Side)
===========================================================

Extracts features from all available MediaPipe pose CSVs including:
- Right side (152 files)
- Left side (103 files)
- Front (133 files)
- Back (if available)

Total: 388+ pose CSVs → Maximum dataset size

Author: Gait Analysis System
Date: 2025-10-30
"""

import json
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import interpolate
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class GAVDAllViewsExtractor:
    """Extract from all available GAVD pose CSVs"""

    def __init__(self):
        self.gavd_root = Path("/data/datasets/GAVD")
        self.pose_dir = self.gavd_root / "mediapipe_pose"
        self.annotations = self._load_annotations()

    def _load_annotations(self) -> pd.DataFrame:
        """Load GAVD annotations"""
        dfs = []
        for i in range(1, 6):
            csv_file = self.gavd_root / f"data/GAVD_Clinical_Annotations_{i}.csv"
            if csv_file.exists():
                df = pd.read_csv(csv_file, low_memory=False)
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _classify_gait(self, gait_pat: str) -> str:
        """Classify gait into normal/pathological"""
        if pd.isna(gait_pat):
            return None
        gait_pat = str(gait_pat).lower()
        if gait_pat == 'normal':
            return 'normal'
        if gait_pat in ['exercise', 'prosthetic', 'style']:
            return None
        return 'pathological'

    def _extract_pattern(self, csv_file: Path, view: str) -> Dict:
        """Extract pattern from one CSV"""
        try:
            df = pd.read_csv(csv_file)
            if len(df) < 30:
                return None

            # Get seq_id
            filename = csv_file.stem
            seq_id = filename.split('_')[0]

            # Get gait class
            seq_info = self.annotations[self.annotations['seq'] == seq_id]
            if len(seq_info) == 0:
                return None

            gait_pat = seq_info.iloc[0]['gait_pat']
            gait_class = self._classify_gait(gait_pat)
            if gait_class is None:
                return None

            # Extract heel heights
            left_heel = df[df['position'] == 'left_heel'].sort_values('frame')
            right_heel = df[df['position'] == 'right_heel'].sort_values('frame')

            if len(left_heel) < 10 or len(right_heel) < 10:
                return None

            heel_height_left = left_heel['y'].values
            heel_height_right = right_heel['y'].values

            # NaN interpolation
            for heel_data in [heel_height_left, heel_height_right]:
                if np.any(np.isnan(heel_data)):
                    nan_pct = np.sum(np.isnan(heel_data)) / len(heel_data)
                    if nan_pct >= 0.5:
                        return None
                    valid_idx = ~np.isnan(heel_data)
                    if np.sum(valid_idx) <= 1:
                        return None

            # Interpolate NaN
            if np.any(np.isnan(heel_height_left)):
                valid_idx = ~np.isnan(heel_height_left)
                f = interpolate.interp1d(np.where(valid_idx)[0], heel_height_left[valid_idx],
                                        kind='linear', fill_value='extrapolate')
                heel_height_left = f(np.arange(len(heel_height_left)))

            if np.any(np.isnan(heel_height_right)):
                valid_idx = ~np.isnan(heel_height_right)
                f = interpolate.interp1d(np.where(valid_idx)[0], heel_height_right[valid_idx],
                                        kind='linear', fill_value='extrapolate')
                heel_height_right = f(np.arange(len(heel_height_right)))

            # 3D stride length
            left_hip = df[df['position'] == 'left_hip'].sort_values('frame')
            right_hip = df[df['position'] == 'right_hip'].sort_values('frame')

            stride_3d = 0
            if len(left_hip) > 10 and len(right_hip) > 10:
                min_len = min(len(left_hip), len(right_hip))
                hip_x = (left_hip['x'].values[:min_len] + right_hip['x'].values[:min_len]) / 2
                hip_z = (left_hip['z'].values[:min_len] + right_hip['z'].values[:min_len]) / 2

                peaks_left, _ = find_peaks(heel_height_left, height=np.mean(heel_height_left), distance=5)
                if len(peaks_left) > 1:
                    dists = []
                    for i in range(len(peaks_left)-1):
                        idx1 = peaks_left[i]
                        idx2 = peaks_left[i+1]
                        if idx1 < len(hip_x) and idx2 < len(hip_x):
                            dx = hip_x[idx2] - hip_x[idx1]
                            dz = hip_z[idx2] - hip_z[idx1]
                            dists.append(np.sqrt(dx**2 + dz**2))
                    stride_3d = np.mean(dists) if len(dists) > 0 else 0

            # Trunk sway
            left_shoulder = df[df['position'] == 'left_shoulder'].sort_values('frame')
            right_shoulder = df[df['position'] == 'right_shoulder'].sort_values('frame')

            trunk_sway = 0
            if len(left_shoulder) > 10 and len(right_shoulder) > 10:
                min_len = min(len(left_shoulder), len(right_shoulder))
                shoulder_x = (left_shoulder['x'].values[:min_len] + right_shoulder['x'].values[:min_len]) / 2
                trunk_sway = np.std(shoulder_x)

            pattern = {
                'seq_id': seq_id,
                'csv_file': str(csv_file),
                'camera_view': view,
                'gait_class': gait_class,
                'gait_pathology': str(gait_pat),
                'heel_height_left': heel_height_left.tolist(),
                'heel_height_right': heel_height_right.tolist(),
                'stride_length_3d': float(stride_3d),
                'trunk_sway': float(trunk_sway),
                'n_frames': len(df['frame'].unique()),
                'fps': 30
            }

            return pattern

        except Exception as e:
            return None

    def extract_all(self) -> List[Dict]:
        """Extract from all views"""

        print("="*80)
        print("Extracting ALL GAVD Views")
        print("="*80)
        print()

        patterns = []
        stats = {'extracted': 0, 'failed': 0, 'excluded': 0}

        for view in ['right_side', 'left_side', 'front', 'back']:
            view_dir = self.pose_dir / view
            if not view_dir.exists():
                continue

            csv_files = list(view_dir.glob("*.csv"))
            print(f"{view}: {len(csv_files)} CSV files")

            view_extracted = 0
            for csv_file in csv_files:
                pattern = self._extract_pattern(csv_file, view)

                if pattern is not None:
                    patterns.append(pattern)
                    view_extracted += 1
                    stats['extracted'] += 1
                else:
                    if self._classify_gait(self.annotations[
                        self.annotations['seq'] == csv_file.stem.split('_')[0]
                    ].iloc[0]['gait_pat'] if len(self.annotations[
                        self.annotations['seq'] == csv_file.stem.split('_')[0]
                    ]) > 0 else None) is None:
                        stats['excluded'] += 1
                    else:
                        stats['failed'] += 1

            print(f"  → Extracted {view_extracted} patterns")

        print()
        print("="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"  Total extracted: {stats['extracted']}")
        print(f"  Failed validation: {stats['failed']}")
        print(f"  Excluded (prosthetic/exercise/style): {stats['excluded']}")
        print()

        # Summary by class
        normal = [p for p in patterns if p['gait_class'] == 'normal']
        pathological = [p for p in patterns if p['gait_class'] == 'pathological']

        print("Dataset Summary:")
        print(f"  Normal: {len(normal)}")
        print(f"  Pathological: {len(pathological)}")
        print(f"  Total: {len(patterns)}")
        print()

        # Summary by view
        print("By Camera View:")
        for view in ['right_side', 'left_side', 'front', 'back']:
            view_patterns = [p for p in patterns if p['camera_view'] == view]
            if len(view_patterns) > 0:
                print(f"  {view}: {len(view_patterns)}")
        print()

        # Pathology distribution
        from collections import Counter
        pathology_counts = Counter([p['gait_pathology'] for p in pathological])
        print("Pathology Distribution:")
        for pathology, count in sorted(pathology_counts.items(), key=lambda x: -x[1]):
            print(f"  {pathology:20s}: {count:3d} cases")
        print()

        return patterns


def main():
    """Main extraction"""

    extractor = GAVDAllViewsExtractor()
    patterns = extractor.extract_all()

    # Save
    output_file = "gavd_all_views_patterns.json"
    with open(output_file, 'w') as f:
        json.dump(patterns, f, indent=2)

    print(f"Saved {len(patterns)} patterns to: {output_file}")
    print()


if __name__ == "__main__":
    main()
