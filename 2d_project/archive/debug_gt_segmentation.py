
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def parse_vector_gt_csv(csv_path):
    # Same as cycle_comparison.py
    try:
        df = pd.read_csv(csv_path, header=None, skiprows=4)
        target_params = ['r.kn.angle']
        data = {}
        for param in target_params:
            rows = df[df[1] == param]
            if rows.empty: continue
            rows = rows.sort_values(by=0)
            vals_x = pd.to_numeric(rows[4], errors='coerce').values
            vals_y = pd.to_numeric(rows[5], errors='coerce').values
            vals_z = pd.to_numeric(rows[6], errors='coerce').values
            data[param] = {'x': vals_x, 'y': vals_y, 'z': vals_z}
        return data
    except Exception as e:
        print(e)
        return None

def debug_gt():
    path = "/data/gait/data/1/excel/S1_01_edited.csv"
    data = parse_vector_gt_csv(path)
    if not data:
        print("Parse failed")
        return
        
    sx = pd.Series(data['r.kn.angle']['x']).interpolate().values
    sy = pd.Series(data['r.kn.angle']['y']).interpolate().values
    sz = pd.Series(data['r.kn.angle']['z']).interpolate().values
    
    print(f"X range: {sx.min():.2f} to {sx.max():.2f}")
    print(f"Y range: {sy.min():.2f} to {sy.max():.2f}")
    print(f"Z range: {sz.min():.2f} to {sz.max():.2f}")
    
    # Test segmentation on Y (Flexion)
    signal = sy
    inverted = -signal
    # Tune parameters here if needed
    peaks, props = find_peaks(inverted, distance=20, prominence=1)
    print(f"DEBUG: Found {len(peaks)} peaks on Y-axis with dist=20, prom=1")
    
    plt.figure(figsize=(12, 6))
    plt.plot(signal, label='Y (Flexion)')
    plt.plot(peaks, signal[peaks], 'rx', label='Detected Extension')
    plt.title(f"GT Knee Angle Components")
    plt.legend()
    plt.savefig("/data/gait/2d_project/gt_components.png")

if __name__ == "__main__":
    debug_gt()
