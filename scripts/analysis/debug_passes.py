import pandas as pd
import numpy as np
import glob
from scipy.signal import find_peaks

def detect_passes_debug(df):
    print(f"Data Length: {len(df)}")
    if 'LEFT_HIP_x' not in df.columns:
        print("Missing HIP columns")
        return []
    
    # 1. Get Hip X Trajectory
    hip_x = (df['LEFT_HIP_x'] + df['RIGHT_HIP_x']) / 2
    print(f"Hip X Range: {hip_x.min():.3f} to {hip_x.max():.3f}")
    
    # 2. Smooth
    hip_x_smooth = hip_x.rolling(window=45, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # 3. Find Peaks/Valleys
    # Relax parameters for debugging
    peaks, _ = find_peaks(hip_x_smooth, distance=60, prominence=0.05) 
    valleys, _ = find_peaks(-hip_x_smooth, distance=60, prominence=0.05)
    
    print(f"Peaks (Indices): {peaks}")
    print(f"Valleys (Indices): {valleys}")
    
    turn_points = sorted(list(peaks) + list(valleys))
    print(f"Turn Points: {turn_points}")
    
    boundaries = [0] + turn_points + [len(df)]
    boundaries = sorted(list(set(boundaries)))
    
    passes = []
    min_len = 60
    min_disp = 0.2 # Relaxed
    
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        duration = end - start
        
        disp = hip_x_smooth.iloc[end-1] - hip_x_smooth.iloc[start]
        print(f"Segment {i}: {start}-{end} (Dur: {duration}), Disp: {disp:.3f}")
        
        if duration < min_len:
            print("  -> Too short")
            continue
        
        if abs(disp) < min_disp:
            print("  -> Too small displacement")
            continue
            
        direction = 'right' if disp > 0 else 'left'
        passes.append({'start': start, 'end': end, 'direction': direction})
        print(f"  -> Added Pass ({direction})")
        
    return passes

def main():
    csv_files = glob.glob(f"/data/gait/data/2/*_side_pose_fps*.csv")
    if not csv_files:
        print("No CSV found")
        return
    
    csv_path = csv_files[0]
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'position' in df.columns:
         df = df.pivot(index='frame', columns='position', values=['x', 'y', 'z', 'visibility'])
         df.columns = [f'{pos.upper()}_{col}' for col, pos in df.columns]
         df = df.reset_index()
         
    passes = detect_passes_debug(df)
    print(f"Total Passes Found: {len(passes)}")

if __name__ == "__main__":
    main()
