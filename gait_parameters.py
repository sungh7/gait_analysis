import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class GaitParameterCalculator:
    def __init__(self, fps: float = 30.0, height_cm: float = 170.0):
        """
        Args:
            fps: Frames per second of the video
            height_cm: Subject height in cm (used for spatial estimation)
        """
        self.fps = fps
        self.height_cm = height_cm

    def calculate_temporal_parameters(self, heel_strikes: Dict[str, List[int]]) -> Dict[str, Dict]:
        """
        Calculate temporal parameters from heel strike frames.
        
        Args:
            heel_strikes: Dictionary with 'left' and 'right' lists of heel strike frame numbers.
            
        Returns:
            Dictionary with parameters for left and right sides.
        """
        results = {}
        
        # Combine all strikes to calculate step times
        all_strikes = []
        for side in ['left', 'right']:
            for frame in heel_strikes.get(side, []):
                all_strikes.append({'frame': frame, 'side': side})
        
        all_strikes.sort(key=lambda x: x['frame'])
        
        for side in ['left', 'right']:
            side_strikes = sorted(heel_strikes.get(side, []))
            opp_side = 'right' if side == 'left' else 'left'
            
            if len(side_strikes) < 2:
                results[side] = {
                    'stride_time_s': np.nan,
                    'cadence_spm': np.nan,
                    'step_time_s': np.nan
                }
                continue
                
            # 1. Stride Time (s): Time between consecutive heel strikes of the SAME foot
            stride_frames = np.diff(side_strikes)
            stride_time_s = np.mean(stride_frames) / self.fps
            
            # 2. Cadence (steps/min): 60 / Step Time, or 120 / Stride Time
            cadence_spm = 120.0 / stride_time_s if stride_time_s > 0 else 0
            
            # 3. Step Time (s): Time from Opposite HS to Current HS
            # Find preceding opposite strike for each current strike
            step_times = []
            for curr_frame in side_strikes:
                # Find the closest preceding opposite strike
                prec_opp = [s for s in all_strikes if s['side'] == opp_side and s['frame'] < curr_frame]
                if prec_opp:
                    prev_frame = prec_opp[-1]['frame']
                    # Check if it's within reasonable range (e.g., < 1.5 stride)
                    if (curr_frame - prev_frame) < (stride_frames.mean() * 1.5):
                        step_times.append(curr_frame - prev_frame)
            
            step_time_s = (np.mean(step_times) / self.fps) if step_times else (stride_time_s / 2.0)
            
            results[side] = {
                'stride_time_s': float(stride_time_s),
                'cadence_spm': float(cadence_spm),
                'step_time_s': float(step_time_s)
            }
            
        return results

    def estimate_spatial_parameters(self, temporal_params: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Estimate spatial parameters using anthropometric models.
        Note: This is a heuristic estimation for 2D video without depth.
        
        Model: Step Length approx proportional to Height and Cadence (or Stride Frequency)
        Simple Model: Step Length = 0.4 * Height (Rough approximation for normal walking)
        Better Model (Hof, 1996): Relates Stride Length to Leg Length and Froude number.
        
        Here we use a simplified estimation:
        Stride Length (m) ≈ Velocity * Stride Time
        Velocity (m/s) ≈ 0.25 * g^0.5 * LegLength^0.5 * (StrideFreq)^? -> Too complex without calibration.
        
        We will use: Step Length (m) ≈ 0.41 * Height (m)
        Stride Length (m) = 2 * Step Length
        Velocity (m/s) = Stride Length / Stride Time
        """
        results = {}
        height_m = self.height_cm / 100.0
        
        # Heuristic Step Length Ratio (0.41 is typical for adults)
        # Can be adjusted if we have calibration
        step_length_ratio = 0.41
        
        for side, params in temporal_params.items():
            stride_time = params.get('stride_time_s')
            
            if stride_time is None or np.isnan(stride_time):
                results[side] = {
                    'step_length_m': np.nan,
                    'stride_length_m': np.nan,
                    'velocity_mps': np.nan
                }
                continue
                
            # Estimate Step Length
            step_length_m = height_m * step_length_ratio
            stride_length_m = step_length_m * 2.0
            
            # Calculate Velocity
            velocity_mps = stride_length_m / stride_time if stride_time > 0 else 0
            
            results[side] = {
                'step_length_m': float(step_length_m),
                'stride_length_m': float(stride_length_m),
                'velocity_mps': float(velocity_mps)
            }
            
        return results
