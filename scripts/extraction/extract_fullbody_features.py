import pandas as pd
import numpy as np
import scipy.signal
import argparse
import os

def calculate_temporal_features(df, fps=30):
    """
    Calculates temporal features based on ankle vertical movement.
    """
    # Use left ankle y-coordinate (vertical) to detect steps
    # In MediaPipe, y increases downwards. 
    # Heel strike is typically a local maximum in y (lowest point visually) or minimum depending on perspective.
    # Let's assume cyclic movement.
    
    # Filter signal
    y_signal = df['y_27'].values # Left ankle
    
    # Find peaks (heel strikes)
    peaks, _ = scipy.signal.find_peaks(-y_signal, distance=fps*0.5) # Min distance 0.5s
    
    num_steps = len(peaks)
    duration_s = len(df) / fps
    
    cadence = (num_steps / duration_s) * 60 if duration_s > 0 else 0
    
    # Step times
    step_times = np.diff(peaks) / fps
    mean_step_time = np.mean(step_times) if len(step_times) > 0 else 0
    std_step_time = np.std(step_times) if len(step_times) > 0 else 0
    
    stride_time_cv = (std_step_time / mean_step_time) * 100 if mean_step_time > 0 else 0
    
    return {
        'cadence_spm': cadence,
        'mean_step_time_s': mean_step_time,
        'stride_time_cv': stride_time_cv
    }

def calculate_asymmetry_features(df):
    """
    Calculates asymmetry features using relative ranges of motion.
    """
    # Arm Swing (Shoulder Z-movement or total displacement)
    # Using x-coordinate (forward/backward) for arm swing proxy in side view
    left_arm_rom = df['x_11'].max() - df['x_11'].min()
    right_arm_rom = df['x_12'].max() - df['x_12'].min()
    
    arm_swing_asymmetry = abs(left_arm_rom - right_arm_rom)
    
    # Step Length Proxy (Ankle X-excursion)
    left_step_rom = df['x_27'].max() - df['x_27'].min()
    right_step_rom = df['x_28'].max() - df['x_28'].min()
    
    step_length_ratio = left_step_rom / right_step_rom if right_step_rom > 0 else 1.0
    
    return {
        'arm_swing_asymmetry': arm_swing_asymmetry,
        'step_length_ratio': step_length_ratio,
        'left_arm_rom': left_arm_rom,
        'right_arm_rom': right_arm_rom
    }

def calculate_coordination_features(df):
    """
    Calculates phase coordination between arm and leg.
    """
    # Cross-correlation between Left Arm (11) and Right Leg (28)
    # They should be in phase (move forward together) or anti-phase depending on coord system
    
    s1 = df['x_11'].values # Left Shoulder
    s2 = df['x_28'].values # Right Ankle
    
    # Normalize
    s1 = (s1 - np.mean(s1)) / (np.std(s1) + 1e-6)
    s2 = (s2 - np.mean(s2)) / (np.std(s2) + 1e-6)
    
    correlation = np.corrcoef(s1, s2)[0, 1]
    
    return {
        'arm_leg_coordination': correlation
    }

def extract_features(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Basic check for required columns
        required_cols = ['y_27', 'x_11', 'x_12', 'x_27', 'x_28']
        for col in required_cols:
            if col not in df.columns:
                # Try to handle case where columns might be named differently (e.g. just numbers)
                # But for now assume standard format from our mock generator
                print(f"Missing column {col} in {csv_path}")
                return None

        temporal = calculate_temporal_features(df)
        asymmetry = calculate_asymmetry_features(df)
        coordination = calculate_coordination_features(df)
        
        features = {**temporal, **asymmetry, **coordination}
        features['filename'] = os.path.basename(csv_path)
        
        return features
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to single input CSV')
    parser.add_argument('--input_dir', type=str, help='Path to directory of CSVs')
    parser.add_argument('--output', type=str, default='calibration_free_features.csv')
    args = parser.parse_args()
    
    results = []
    
    if args.input_file:
        feat = extract_features(args.input_file)
        if feat:
            results.append(feat)
            
    elif args.input_dir:
        for fname in os.listdir(args.input_dir):
            if fname.endswith('.csv') and 'pose_landmarks' in fname:
                path = os.path.join(args.input_dir, fname)
                feat = extract_features(path)
                if feat:
                    results.append(feat)
    
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(args.output, index=False)
        print(f"Features extracted to {args.output}")
        print(out_df.head())
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
