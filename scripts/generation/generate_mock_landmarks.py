import pandas as pd
import numpy as np
import os

def generate_mock_landmarks(output_path, num_frames=300, gait_type='normal'):
    """
    Generates synthetic MediaPipe landmark data simulating a walking gait.
    Types: 'normal', 'hemiplegic' (asymmetric), 'parkinsonian' (small steps, no arm swing).
    """
    
    # Time vector
    t = np.linspace(0, 10, num_frames)
    
    # Gait cycle frequency (approx 1 Hz)
    freq = 1.0
    if gait_type == 'parkinsonian':
        freq = 1.5 # Shuffling, faster cadence often
    
    # Generate data dictionary
    data = {
        'timestamp': t,
    }
    
    # Define landmarks to simulate (x, y, z, visibility)
    # 11, 12: Shoulders
    # 23, 24: Hips
    # 25, 26: Knees
    # 27, 28: Ankles
    
    landmarks = [
        (11, 'left_shoulder'), (12, 'right_shoulder'),
        (23, 'left_hip'), (24, 'right_hip'),
        (25, 'left_knee'), (26, 'right_knee'),
        (27, 'left_ankle'), (28, 'right_ankle')
    ]
    
    for lm_id, name in landmarks:
        # Base position
        base_x = 0.5
        base_y = 0.5
        base_z = 0.0
        
        # Phase shift for left/right (180 degrees out of phase)
        phase = 0 if 'left' in name else np.pi
        
        # Default Amplitudes (Normal)
        amp_x = 0.0 # Forward/Back (Arm swing / Step length)
        amp_y = 0.05 # Vertical bobbing
        amp_z = 0.1 # Depth
        
        # Adjust based on joint
        if 'ankle' in name:
            amp_y = 0.1
            amp_z = 0.3 # Step length proxy in Z (or X depending on view)
            # Let's use X for forward/back motion in side view
            amp_x = 0.2 
        elif 'knee' in name:
            amp_y = 0.08
            amp_x = 0.1
        elif 'shoulder' in name:
            amp_x = 0.15 # Arm swing
            
        # --- PATHOLOGY MODIFICATIONS ---
        
        if gait_type == 'hemiplegic':
            # Simulate right side impairment
            if 'right' in name:
                amp_x *= 0.2 # Reduced ROM
                amp_y *= 0.5 # Reduced foot clearance
                if 'shoulder' in name:
                    amp_x = 0.01 # No arm swing on affected side
                    
        elif gait_type == 'parkinsonian':
            # Global reduction in amplitude
            amp_x *= 0.3 # Small shuffling steps, reduced arm swing
            amp_y *= 0.2 # Shuffling (low foot clearance)
            if 'shoulder' in name:
                amp_x *= 0.1 # Rigid upper body
        
        # Simulate movement
        # Side view: X is forward/back, Y is up/down
        data[f'x_{lm_id}'] = base_x + amp_x * np.sin(2 * np.pi * freq * t + phase)
        data[f'y_{lm_id}'] = base_y + amp_y * np.cos(2 * np.pi * freq * t + phase)
        data[f'z_{lm_id}'] = base_z + amp_z * np.sin(2 * np.pi * freq * t + phase)
        data[f'v_{lm_id}'] = np.ones(num_frames)
        
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    # print(f"Generated {gait_type} mock data at {output_path}")
    return df

if __name__ == "__main__":
    # Generate one of each for testing
    generate_mock_landmarks('/data/gait/mock_normal.csv', gait_type='normal')
    generate_mock_landmarks('/data/gait/mock_hemiplegic.csv', gait_type='hemiplegic')
    generate_mock_landmarks('/data/gait/mock_parkinsonian.csv', gait_type='parkinsonian')
    print("Generated mock datasets for Normal, Hemiplegic, and Parkinsonian gait.")
