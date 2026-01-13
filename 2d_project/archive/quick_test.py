#!/usr/bin/env python3
"""Quick test of improvements on a small sample."""

import numpy as np
import pandas as pd
from improved_signal_processing import ImprovedSignalProcessor
from kinematic_constraints import KinematicConstraintEnforcer

print("="*60)
print("QUICK TEST: Signal Processing + Kinematic Constraints")
print("="*60)

# Create synthetic noisy knee angle signal (simulating MediaPipe output)
np.random.seed(42)
frames = 200  # ~6.7 seconds at 30fps
t = np.linspace(0, frames/30, frames)

# Simulate gait: sinusoidal knee flexion (0-60 degrees) with noise
clean_signal = 30 + 30 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz gait

# Add realistic noise and artifacts
noise = np.random.normal(0, 5, frames)  # 5-degree random noise
spikes = np.zeros(frames)
spikes[[50, 120, 170]] = [40, -30, 50]  # Tracking errors

# Add some anatomically impossible values
raw_signal = clean_signal + noise + spikes
raw_signal[100:105] = -20  # Impossible hyperextension
raw_signal[150:153] = 150  # Impossible flexion

print("\n1. Testing Signal Processing...")
processor = ImprovedSignalProcessor(fps=30)
processed_signal, quality = processor.process_joint_angle(raw_signal, 'knee')

print(f"   SNR: {quality['snr']:.1f} dB")
print(f"   Smoothness: {quality['smoothness']:.3f}")
print(f"   Overall Quality: {quality['overall_quality']:.3f}")

# Check improvement
raw_jerk = np.std(np.diff(raw_signal, n=2))
processed_jerk = np.std(np.diff(processed_signal, n=2))
jerk_reduction = (raw_jerk - processed_jerk) / raw_jerk * 100

print(f"   Jerk Reduction: {jerk_reduction:.1f}%")

print("\n2. Testing Kinematic Constraints...")

# Create DataFrame for constraint enforcer
angles_df = pd.DataFrame({
    'right_knee_angle': raw_signal
})

enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)

# Check violations before
violations_before = enforcer.validate_angles(angles_df)
total_before = (violations_before['right_knee_angle']['angle_violations'] +
                violations_before['right_knee_angle']['velocity_violations'] +
                violations_before['right_knee_angle']['acceleration_violations'])

print(f"   Violations BEFORE: {total_before}")

# Apply constraints
angles_constrained = enforcer.enforce_joint_angle_constraints(angles_df)

# Check violations after
violations_after = enforcer.validate_angles(angles_constrained)
total_after = (violations_after['right_knee_angle']['angle_violations'] +
               violations_after['right_knee_angle']['velocity_violations'] +
               violations_after['right_knee_angle']['acceleration_violations'])

print(f"   Violations AFTER: {total_after}")
print(f"   Fixed: {total_before - total_after} violations")

print("\n3. Visual Comparison...")

# Quick ASCII visualization
print("\n   Raw signal (first 50 frames):")
mini_raw = raw_signal[:50]
print(f"   Range: {mini_raw.min():.1f} to {mini_raw.max():.1f} degrees")
print(f"   Mean: {mini_raw.mean():.1f}, Std: {mini_raw.std():.1f}")

print("\n   Processed signal (first 50 frames):")
mini_proc = processed_signal[:50]
print(f"   Range: {mini_proc.min():.1f} to {mini_proc.max():.1f} degrees")
print(f"   Mean: {mini_proc.mean():.1f}, Std: {mini_proc.std():.1f}")

print("\n   Constrained signal (first 50 frames):")
mini_const = angles_constrained['right_knee_angle'].values[:50]
print(f"   Range: {mini_const.min():.1f} to {mini_const.max():.1f} degrees")
print(f"   Mean: {mini_const.mean():.1f}, Std: {mini_const.std():.1f}")

print("\n" + "="*60)
print("✓ QUICK TEST COMPLETE")
print("="*60)

print("\nKey Results:")
print(f"  - Signal processing reduced jerk by {jerk_reduction:.1f}%")
print(f"  - Kinematic constraints fixed {total_before - total_after} violations")
print(f"  - Overall quality score: {quality['overall_quality']:.2f}/1.0")

if quality['overall_quality'] > 0.6 and (total_before - total_after) > 0:
    print("\n✓ Improvements are working as expected!")
else:
    print("\n⚠ Check parameters - results may need tuning")

# Create simple plot
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.plot(raw_signal, 'r-', alpha=0.3, linewidth=1, label='Raw (noisy + artifacts)')
    ax.plot(processed_signal, 'b-', linewidth=1.5, label='After signal processing')
    ax.plot(angles_constrained['right_knee_angle'].values, 'g-', linewidth=2, label='After constraints')

    # Show physiological limits
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Physiological limits')
    ax.axhline(140, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Knee Angle (degrees)')
    ax.set_title('Quick Test: Improvement Pipeline on Synthetic Data')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/data/gait/2d_project/quick_test_result.png', dpi=150)
    print("\n✓ Saved plot to: quick_test_result.png")

except Exception as e:
    print(f"\n⚠ Could not create plot: {e}")
