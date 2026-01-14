"""
Gait Analysis Data Parser
Extracts subject information and joint angle measurements from hospital gait analysis Excel files.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


class GaitDataParser:
    """Parser for gait analysis Excel files with subject info and joint angle measurements."""

    # Joint position codes
    JOINTS = [
        'r.an.angle', 'r.kn.angle', 'r.hi.angle', 'r.ga.angle', 'r.pe.angle', 'r.to.angle',
        'l.an.angle', 'l.kn.angle', 'l.hi.angle', 'l.ga.angle', 'l.pe.angle', 'l.to.angle',
        'r.sh.angle', 'r.el.angle', 'l.sh.angle', 'l.el.angle'
    ]

    # Column mappings (0-indexed)
    COLS = {
        'joint': 0,
        'gait_cycle': 1,
        # Condition 1 (Subject measurement)
        'condition1_xyz': (3, 4, 5),  # D, E, F
        'condition1_upper_sd_xyz': (6, 7, 8),  # G, H, I
        'condition1_lower_sd_xyz': (9, 10, 11),  # J, K, L
        'condition1_sd_xyz': (12, 13, 14),  # M, N, O
        # Normal reference
        'normal_xyz': (27, 28, 29),  # AB, AC, AD
        'normal_upper_sd_xyz': (30, 31, 32),  # AE, AF, AG
        'normal_lower_sd_xyz': (33, 34, 35),  # AH, AI, AJ
        'normal_sd_xyz': (36, 37, 38),  # AK, AL, AM
        'normal_sdx2_xyz': (39, 40, 41),  # AN, AO, AP
        # Subject demographics
        'demographics_col': 43,  # AR column
    }

    # Demographics row mappings
    DEMO_ROWS = {
        'name': 2,
        'hospital_id': 3,
        'date': 4,
        'age': 5,
        'comment': 6,
        'height': 7,
        'examiner': 8,
        'normal_age_comparison': 9,
        'right_strides': 10,
        'left_strides': 11,
        'weight': 12,
        'right_ids': 17,
        'right_ss': 18,
        'right_ids_ss': 19,
        'right_sls': 20,
        'left_ids': 21,
        'left_ss': 22,
        'left_ids_ss': 23,
        'left_sls': 24,
        'cadence_right': 26,
        'cadence_left': 27,
        'cadence_avg': 28,
        'cadence_avg2': 29,
    }

    def __init__(self, file_path: str):
        """
        Initialize parser with Excel file path.

        Args:
            file_path: Path to the Excel file
        """
        self.file_path = file_path
        self.df = None
        self._load_data()

    def _load_data(self):
        """Load Excel file with proper settings."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.df = pd.read_excel(self.file_path, sheet_name=0, header=None)

    def extract_subject_info(self) -> Dict:
        """
        Extract subject demographics and clinical information.

        Returns:
            Dictionary with subject information
        """
        col = self.COLS['demographics_col']

        info = {
            'name': self._get_demo_value('name', col),
            'hospital_id': self._get_demo_value('hospital_id', col),
            'date': self._get_demo_value('date', col),
            'age': self._get_demo_value('age', col),
            'comment': self._get_demo_value('comment', col),
            'height_cm': self._get_demo_value('height', col),
            'weight_kg': self._get_demo_value('weight', col),
            'examiner': self._get_demo_value('examiner', col),
            'normal_age_comparison': self._get_demo_value('normal_age_comparison', col),
            'strides': {
                'right': self._get_demo_value('right_strides', col),
                'left': self._get_demo_value('left_strides', col),
            },
            'gait_cycle_timing': {
                'right': {
                    'ids': self._get_demo_value('right_ids', col),
                    'ss': self._get_demo_value('right_ss', col),
                    'ids_ss': self._get_demo_value('right_ids_ss', col),
                    'sls': self._get_demo_value('right_sls', col),
                },
                'left': {
                    'ids': self._get_demo_value('left_ids', col),
                    'ss': self._get_demo_value('left_ss', col),
                    'ids_ss': self._get_demo_value('left_ids_ss', col),
                    'sls': self._get_demo_value('left_sls', col),
                }
            },
            'cadence': {
                'right': self._get_demo_value('cadence_right', col),
                'left': self._get_demo_value('cadence_left', col),
                'average': self._get_demo_value('cadence_avg', col),
            }
        }

        return info

    def _get_demo_value(self, key: str, col: int):
        """Get demographic value from specific row and column."""
        row = self.DEMO_ROWS.get(key)
        if row is None:
            return None
        value = self.df.iloc[row, col]
        return None if pd.isna(value) else value

    def extract_gait_data_long(self, subject_id: Optional[str] = None) -> pd.DataFrame:
        """
        Extract gait angle data in long format.

        Args:
            subject_id: Optional subject identifier to include in output

        Returns:
            DataFrame with columns:
            - subject_id (if provided)
            - joint
            - gait_cycle (0-100)
            - plane (x, y, z)
            - condition1_avg
            - condition1_upper_sd
            - condition1_lower_sd
            - condition1_sd
            - normal_avg
            - normal_upper_sd
            - normal_lower_sd
            - normal_sd
            - normal_sdx2
        """
        # Extract valid rows (0-1617 in 0-indexed, rows 2-1617 contain data)
        gait_df = self.df.iloc[2:1618].copy()

        rows = []
        for _, row in gait_df.iterrows():
            joint = row[self.COLS['joint']]
            cycle = row[self.COLS['gait_cycle']]

            if pd.isna(joint) or pd.isna(cycle):
                continue

            # Extract data for each plane (x, y, z)
            for plane_idx, plane in enumerate(['x', 'y', 'z']):
                data_row = {
                    'joint': joint,
                    'gait_cycle': int(cycle),
                    'plane': plane,
                    'condition1_avg': row[self.COLS['condition1_xyz'][plane_idx]],
                    'condition1_upper_sd': row[self.COLS['condition1_upper_sd_xyz'][plane_idx]],
                    'condition1_lower_sd': row[self.COLS['condition1_lower_sd_xyz'][plane_idx]],
                    'condition1_sd': row[self.COLS['condition1_sd_xyz'][plane_idx]],
                    'normal_avg': row[self.COLS['normal_xyz'][plane_idx]],
                    'normal_upper_sd': row[self.COLS['normal_upper_sd_xyz'][plane_idx]],
                    'normal_lower_sd': row[self.COLS['normal_lower_sd_xyz'][plane_idx]],
                    'normal_sd': row[self.COLS['normal_sd_xyz'][plane_idx]],
                    'normal_sdx2': row[self.COLS['normal_sdx2_xyz'][plane_idx]],
                }

                if subject_id:
                    data_row['subject_id'] = subject_id

                rows.append(data_row)

        df_long = pd.DataFrame(rows)

        # Reorder columns
        cols = ['subject_id'] if subject_id else []
        cols.extend(['joint', 'gait_cycle', 'plane',
                     'condition1_avg', 'condition1_upper_sd', 'condition1_lower_sd', 'condition1_sd',
                     'normal_avg', 'normal_upper_sd', 'normal_lower_sd', 'normal_sd', 'normal_sdx2'])

        return df_long[cols]

    def validate_data(self) -> Tuple[bool, str]:
        """
        Validate data structure and completeness.

        Returns:
            Tuple of (is_valid, message)
        """
        issues = []

        # Check data shape
        if self.df.shape[0] < 1618:
            issues.append(f"Expected at least 1618 rows, got {self.df.shape[0]}")

        # Check for expected joints
        gait_df = self.df.iloc[2:1618]
        unique_joints = gait_df[self.COLS['joint']].dropna().unique()
        missing_joints = set(self.JOINTS) - set(unique_joints)
        if missing_joints:
            issues.append(f"Missing joints: {missing_joints}")

        # Check gait cycle range (should be 0-100, 101 points)
        for joint in self.JOINTS:
            joint_data = gait_df[gait_df[self.COLS['joint']] == joint]
            cycles = joint_data[self.COLS['gait_cycle']].dropna().unique()
            if len(cycles) != 101:
                issues.append(f"Joint {joint} has {len(cycles)} cycle points (expected 101)")

        # Check subject info
        demo_col = self.COLS['demographics_col']
        if pd.isna(self.df.iloc[self.DEMO_ROWS['name'], demo_col]):
            issues.append("Subject name is missing")

        if issues:
            return False, "; ".join(issues)
        return True, "Data validation passed"

    def get_joint_data(self, joint: str, plane: str = 'y') -> pd.DataFrame:
        """
        Extract data for a specific joint and plane.

        Args:
            joint: Joint code (e.g., 'r.kn.angle')
            plane: Plane ('x', 'y', or 'z'). Default 'y' (sagittal)

        Returns:
            DataFrame with gait cycle and measurements
        """
        long_df = self.extract_gait_data_long()
        return long_df[(long_df['joint'] == joint) & (long_df['plane'] == plane)].copy()
