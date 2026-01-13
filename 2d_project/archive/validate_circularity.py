
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate, resample
from fastdtw import fastdtw
import sys
import os
from tqdm import tqdm

# --- IMPORTS ---
sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template

OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# Use Subject 1 as the primary validation testbed
VIDEO_PATH = "/data/gait/data/1/1-2.mp4"

def segment_cycle_dtw(signal, template, stride=4, start_idx=0, end_idx=None):
    """Scan signal with template using Euclidean distance proxy"""
    if end_idx is None: end_idx = len(signal)
    
    dtw_profile = []
    T = 35 # Assumed window size
    
    scan_range = range(start_idx, end_idx - T, stride)
    if not scan_range: return []
    
    for i in scan_range:
        window = signal[i:i+T]
        w_norm = resample(window, 101) # Match template size
        dist = np.linalg.norm(w_norm - template)
        dtw_profile.append(dist)
        
    dtw_profile = np.array(dtw_profile)
    # Find peaks in -distance (minima)
    inverted = -dtw_profile
    peaks, _ = find_peaks(inverted, distance=int(T/stride * 0.7))
    return peaks * stride + start_idx

def run_loco_experiment():
    print("Running LOCO (Leave-One-Cycle-Out) Experiment on S1...")
    
    # 1. Load Signal
    extractor = MediaPipeSagittalExtractor()
    lm, _ = extractor.extract_pose_landmarks(VIDEO_PATH)
    angles = extractor.calculate_joint_angles(lm)
    signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    # 2. Get Initial Candidates (Rough Segmentation)
    # Re-use logic from derive_self_template to get the pool of cycles
    full_template, candidates = derive_self_template(signal)
    
    print(f"Total Candidates Found: {len(candidates)}")
    
    # 3. LOCO Loop
    # For each candidate cycle (roughly identified), we generate a NEW template WITHOUT it
    # And check if we can DETECT it accurately using the LOCO template.
    
    results = []
    
    # Since candidates are just waveforms, we need their original indices to check detection?
    # Actually derive_self_template doesn't return indices.
    # Let's rebuild the logic here to keep track of indices.
    
    # Detect Standard Cycles first (Target)
    starts_standard = segment_cycle_dtw(signal, full_template)
    print(f"Standard Detection Count: {len(starts_standard)}")
    
    detected_count_loco = 0
    total_loco_trials = 0
    
    # To truly do LOCO on the *Template*, we have to iterate through the *detection list*?
    # Or iterate through the *Candidates Pool* used to MAKE the template.
    
    # Let's say we have N candidates used to make the Self-Template.
    # We remove candidate k. Make Template_k.
    # Then we scan the WHOLE signal with Template_k.
    # We check if the cycle corresponding to candidate k is still detected at the same spot.
    
    # Re-extract candidates with indices
    sig_detrend = signal - np.mean(signal)
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:]
    peaks, _ = find_peaks(corr, distance=15)
    period = peaks[1] if len(peaks)>1 else 35
    
    inverted = -signal
    rough_peaks, _ = find_peaks(inverted, distance=int(period*0.7), prominence=5)
    
    candidate_list = [] # (start, end, waveform)
    for i in range(len(rough_peaks)-1):
        s, e = rough_peaks[i], rough_peaks[i+1]
        length = e - s
        if abs(length - period) < period * 0.4:
            seg = signal[s:e]
            seg_norm = resample(seg, 101)
            candidate_list.append({'s':s, 'e':e, 'wave': seg_norm})
            
    print(f"Candidate Pool for Template: {len(candidate_list)}")
    
    # Main LOCO Loop
    matches = 0
    errors = []
    
    for i in tqdm(range(len(candidate_list))):
        target_cycle = candidate_list[i]
        
        # 1. Build LOCO Template (Median of all except i)
        others = [c['wave'] for j, c in enumerate(candidate_list) if j != i]
        if not others: continue
        loco_template = np.median(others, axis=0)
        
        # 2. Phase Align LOCO template
        min_idx = np.argmin(loco_template)
        loco_template = np.roll(loco_template, -min_idx)
        
        # 3. Detect using LOCO template
        # Scan ONLY the neighborhood of the target cycle to be efficient?
        # Or scan full? Let's scan full to be rigorous (check false neg).
        # Scanning full is slow ~1 min per cycle x 40 cycles.
        # Let's scan neighborhood +/- 50 frames.
        
        search_start = max(0, target_cycle['s'] - 50)
        search_end = min(len(signal), target_cycle['e'] + 50)
        
        detected_starts = segment_cycle_dtw(signal, loco_template, start_idx=search_start, end_idx=search_end)
        
        # 4. Check Match
        # We look for a detection near target_cycle['s'] (approx)
        # Note: rough_peaks are EXTENSION peaks (~Heel Strike). dtw matches start (~Heel Strike).
        # They should be close.
        
        target_s = target_cycle['s']
        
        # Find closest detection
        if len(detected_starts) == 0:
            match_dist = 999
        else:
            match_dist = np.min(np.abs(detected_starts - target_s))
            
        if match_dist < 20: # 20 frames tolerance (approx 600ms? No 20 frames is huge at 30fps. 20 frames = 0.6s)
            # Actually tolerance should be smaller. +/- 5 frames?
            # rough pre-segmentation is purely Peak Detection.
            # DTW detection refines it.
            # The question is: Does LOCO template find the SAME spot as Standard Template?
            pass
            
        # Better Comparison: Compare LOCO detection vs SELF detection
        # Run Standard Template detection on this window
        self_starts = segment_cycle_dtw(signal, full_template, start_idx=search_start, end_idx=search_end)
        if len(self_starts) == 0: continue # Should not happen
        self_s = self_starts[np.argmin(np.abs(self_starts - target_s))]
        
        # Now find LOCO detection
        if len(detected_starts) == 0:
            print(f"Cycle {i}: LOCO miss")
            continue
            
        loco_s = detected_starts[np.argmin(np.abs(detected_starts - target_s))]
        
        diff = abs(loco_s - self_s)
        errors.append(diff)
        if diff <= 5: # 5 frames tolerance (approx 150ms? at 30fps = 166ms. It's okay)
            matches += 1
            
    print(f"\n--- LOCO Results ---")
    print(f"Total Cycles Tested: {len(candidate_list)}")
    print(f"Successful Matches (Error <= 5 frames): {matches}")
    print(f"Robustness Rate: {matches/len(candidate_list)*100:.1f}%")
    print(f"Mean Detection Shift: {np.mean(errors):.2f} frames")
    
    # Save chart
    plt.figure()
    plt.hist(errors, bins=10)
    plt.title("Detection Shift (Self-Template vs LOCO-Template)")
    plt.xlabel("Frame Error")
    plt.savefig(f"{OUTPUT_DIR}/loco_error_hist.png")
    
    # Save CSV
    pd.DataFrame({'Cycle': range(len(errors)), 'Shift_Frames': errors}).to_csv(f"{OUTPUT_DIR}/loco_results.csv", index=False)

if __name__ == "__main__":
    run_loco_experiment()
