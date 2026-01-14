import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append("/data/gait")
from mediapipe_csv_processor import MediaPipeCSVProcessor

def predict_omcs(mp_csv_path, output_dir=None):
    mp_csv_path = Path(mp_csv_path)
    if output_dir is None:
        output_dir = mp_csv_path.parent
    else:
        output_dir = Path(output_dir)
        
    print(f"Processing {mp_csv_path}...")
    
    # Load Model
    model_path = "/data/gait/omcs_correction_model.joblib"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Please train it first.")
        return
    
    model = joblib.load(model_path)
    print("Model loaded.")
    
    # Process MP
    processor = MediaPipeCSVProcessor(
        conversion_params_path="/data/gait/angle_conversion_params.json"
    )
    
    mp_results = processor.process_csv_file(mp_csv_path)
    
    predictions = {}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for s_idx, side in enumerate(['left', 'right']):
        mp_cycle = mp_results[side]['averaged_cycle']
        if mp_cycle is None:
            print(f"No cycle found for {side} side.")
            continue
            
        prefix = side[0]
        
        for j_idx, joint in enumerate(['hip', 'knee', 'ankle']):
            joint_code = f"{prefix}.{joint[:2]}.angle"
            
            # Get MP angles
            mp_joint_rows = mp_cycle[mp_cycle['joint'] == joint_code]
            if len(mp_joint_rows) == 0:
                continue
                
            # Handle duplicates
            if len(mp_joint_rows) > 101:
                mp_joint_rows = mp_joint_rows.drop_duplicates(subset=['gait_cycle'])
                
            if len(mp_joint_rows) != 101:
                print(f"Skipping {side} {joint}: rows {len(mp_joint_rows)} != 101")
                continue
                
            mp_angles = mp_joint_rows['angle_mean'].values
            
            # Prepare Features
            # MP Cycle (101) + Joint One-Hot (3) + Side One-Hot (2)
            is_hip = 1 if joint == 'hip' else 0
            is_knee = 1 if joint == 'knee' else 0
            is_ankle = 1 if joint == 'ankle' else 0
            
            is_left = 1 if side == 'left' else 0
            is_right = 1 if side == 'right' else 0
            
            features = np.concatenate([
                mp_angles,
                [is_hip, is_knee, is_ankle],
                [is_left, is_right]
            ]).reshape(1, -1)
            
            # Predict
            pred_angles = model.predict(features)[0]
            
            predictions[f"{side}_{joint}"] = pred_angles
            
            # Plot
            ax = axes[s_idx, j_idx]
            ax.plot(mp_angles, label='MP (Raw)', linestyle='--', color='blue', alpha=0.6)
            ax.plot(pred_angles, label='Predicted OMCS', color='green', linewidth=2)
            ax.set_title(f"{side.capitalize()} {joint.capitalize()}")
            ax.legend()
            ax.grid(True)
            
    plt.tight_layout()
    plot_path = output_dir / f"{mp_csv_path.stem}_prediction.png"
    plt.savefig(plot_path)
    print(f"Prediction plot saved to {plot_path}")
    
    # Save Predictions to CSV
    # Create a DataFrame with columns: gait_cycle, l_hip, l_knee, ...
    data = {'gait_cycle': np.arange(101)}
    for key, val in predictions.items():
        data[key] = val
        
    df_pred = pd.DataFrame(data)
    csv_path = output_dir / f"{mp_csv_path.stem}_predicted_omcs.csv"
    df_pred.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_omcs.py <mp_csv_path>")
    else:
        predict_omcs(sys.argv[1])
