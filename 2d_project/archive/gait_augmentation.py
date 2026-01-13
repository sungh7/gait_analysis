
import numpy as np
from scipy.interpolate import interp1d

class GaitAugmentor:
    """
    Provides methods to simulate:
    1. Realistic 2D Camera Distortions (for Robust PCA Training)
    2. Pathological Gait Anomalies (for Anomaly Detection Testing)
    """
    
    @staticmethod
    def augment_amplitude_flattening(cycle, magnitude_range=(0.6, 0.95)):
        """
        Simulates perspective flattening (2D projection of 3D motion).
        Randomly scales the signal amplitude.
        Scale factor is sampled from Uniform(magnitude_range).
        """
        scale = np.random.uniform(*magnitude_range)
        # Scale around the mean or 0? 
        # Perspective usually compresses the ROM.
        # Assuming cycle is roughly centered or we want to compress ROM about the center of ROM.
        center = (np.max(cycle) + np.min(cycle)) / 2
        return (cycle - center) * scale + center

    @staticmethod
    def augment_phase_shift(cycle, shift_range=(-0.1, 0.1)):
        """
        Simulates temporal misalignment or view-dependent phase delays.
        Shift is fraction of cycle length.
        """
        n = len(cycle)
        shift_frac = np.random.uniform(*shift_range)
        shift_idx = int(n * shift_frac)
        # Circular shift for continuous cycle, or roll with edge handling?
        # Gait cycles are periodic. Roll is appropriate.
        return np.roll(cycle, shift_idx)

    @staticmethod
    def augment_vertical_offset(cycle, offset_std=5.0):
        """
        Simulates systematic keypoint bias (e.g. hip detected too low).
        Adds a constant offset.
        """
        offset = np.random.normal(0, offset_std)
        return cycle + offset
        
    @staticmethod
    def apply_camera_distortion(cycle):
        """
        Applies a random combination of Flattening, Phase Shift, and Offset
        to create a "Synthetic 2D MediaPipe Cycle" from a Vicon Cycle.
        """
        y = cycle.copy()
        y = GaitAugmentor.augment_amplitude_flattening(y)
        y = GaitAugmentor.augment_phase_shift(y)
        y = GaitAugmentor.augment_vertical_offset(y)
        # Add random noise
        noise = np.random.normal(0, 2.0, len(y)) # 2 degrees noise
        return y + noise

    # --- Pathology Simulators (for Anomaly Detection) ---

    @staticmethod
    def simulate_stiff_knee(cycle, reduction_factor=0.5):
        """
        Simulates reduced ROM (Stiff Knee).
        """
        # Similar to flattening, but clinically severe
        center = np.min(cycle) # Stiff knee usually fails to flex, so sticks near extension
        # Or just compress ROM.
        return (cycle - np.mean(cycle)) * reduction_factor + np.mean(cycle)

    @staticmethod
    def simulate_double_bump(cycle):
        """
        Simulates 'Camel Back' or Orthogonal distortion (e.g. at mid-swing).
        Adds a Gaussian bump in a region where it shouldn't be.
        """
        y = cycle.copy()
        x = np.linspace(0, 100, len(y))
        # Add bump at 70% cycle (Swing phase)
        bump = 15.0 * np.exp(-0.5 * ((x - 70) / 5)**2)
        return y + bump

    @staticmethod
    def simulate_asymmetry(cycle_l, cycle_r):
        """
        Makes one leg significantly different from the other.
        """
        cycle_l_mod = cycle_l.copy()
        cycle_r_mod = GaitAugmentor.augment_amplitude_flattening(cycle_r, (0.4, 0.5)) # Severe reduction
        return cycle_l_mod, cycle_r_mod
