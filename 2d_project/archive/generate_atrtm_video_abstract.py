import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks, correlate, resample
import sys
import os

# --- INLINE LOGIC FOR AT-RTM ---

def derive_self_template(signal, verbose=False):
    # 1. Autocorrelation to find approximate period
    sig_detrend = signal - np.mean(signal)
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:] 
    
    peaks, _ = find_peaks(corr, distance=15) # min 0.5s
    
    if len(peaks) < 2:
        period = 35 # fallback
    else:
        period = peaks[1]
    
    if verbose: print(f"Estimated Period: {period}")
    
    # 2. Gather candidates (using extension peaks)
    inverted = -signal
    rough_peaks, _ = find_peaks(inverted, distance=int(period*0.7), prominence=5)
    
    candidates = []
    for i in range(len(rough_peaks)-1):
        start, end = rough_peaks[i], rough_peaks[i+1]
        length = end - start
        if abs(length - period) < period * 0.4:
            seg = signal[start:end]
            seg_norm = resample(seg, 101)
            candidates.append(seg_norm)
            
    if not candidates:
        return None
        
    candidates = np.array(candidates)
    self_template = np.median(candidates, axis=0) # Median template
    
    # Align to Min (Heel Strike proxy)
    min_idx = np.argmin(self_template)
    self_template = np.roll(self_template, -min_idx)
    
    return self_template, period

def scan_signal_euclidean(signal, template, window_size, step=2):
    dists = []
    indices = []
    
    # Pre-calculate full profile for plotting context
    for i in range(0, len(signal) - window_size, 1): # Fine scan for profile
        window = signal[i:i+window_size]
        window_norm = resample(window, len(template))
        # Normalized Euclidean distance
        d = np.linalg.norm(window_norm - template) / np.sqrt(len(template))
        dists.append(d)
        indices.append(i)
        
    dists = np.array(dists)
    
    # Normalize 0-1 for visualization
    if np.max(dists) > 0:
        dists_norm = (dists - np.min(dists)) / (np.max(dists) - np.min(dists))
    else:
        dists_norm = dists
        
    # Find peaks (valleys in distance)
    peaks, _ = find_peaks(-dists_norm, distance=int(window_size*0.7), prominence=0.1)
    
    return indices, dists_norm, peaks

# --- ANIMATION SCRIPT ---

def create_video_abstract(output_file='at_rtm_demo.mp4'):
    print("🎬 Starting Video Abstract Generation...")
    
    # 1. Load Data
    csv_path = '/data/gait/2d_project/batch_results/S1_01_improved_angles.csv'
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # Use 'right_knee_angle' (or 'r.kn.angle' if relying on raw column not present in this file)
    # The file has 'right_knee_angle' which is processed.
    if 'right_knee_angle' in df.columns:
        signal = df['right_knee_angle'].values
    else:
        print("❌ 'right_knee_angle' column not found.")
        return

    # Trim to a nice active walking section (e.g., frames 200 to 500)
    # Subject 1 usually starts walking around frame 50-100.
    start_frame = 200
    end_frame = 550
    signal = signal[start_frame:end_frame]
    
    # Normalize for clean plotting (center around 0 is not needed, but nice range)
    # Leave as degrees for realism
    
    # 2. Run AT-RTM Logic
    print("running AT-RTM logic...")
    template, period = derive_self_template(signal, verbose=True)
    if template is None:
        print("❌ Template derivation failed.")
        return
        
    window_size = int(period) # Scan window ~ period
    indices, dist_profile, peaks = scan_signal_euclidean(signal, template, window_size)
    
    # 3. Setup Animation
    scan_step = 2
    num_frames = (len(signal) - window_size) // scan_step
    
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    # Use dark background for modern look? No, white is better for paper.
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])
    
    # Top: Signal Stream
    ax_signal = fig.add_subplot(gs[0, :])
    ax_signal.set_title("1. Real-time Input Signal (Knee Flexion)", fontsize=14, fontweight='bold')
    ax_signal.set_xlim(0, len(signal))
    # Y-lim with padding
    y_min, y_max = np.min(signal), np.max(signal)
    ax_signal.set_ylim(y_min - 10, y_max + 10)
    ax_signal.set_ylabel("Angle (°)")
    
    # Draw static signal faintly
    ax_signal.plot(signal, color='gray', alpha=0.3, label='Full Signal')
    
    # Dynamic lines
    line_past, = ax_signal.plot([], [], 'k-', linewidth=2, label='Processed')
    rect_window = Rectangle((0, y_min), window_size, y_max-y_min, 
                           linewidth=2, edgecolor='#E63946', facecolor='none', label='Scanning Window')
    ax_signal.add_patch(rect_window)
    ax_signal.legend(loc='upper right')
    
    # Bottom Left: Template Matching Zoom-in
    ax_match = fig.add_subplot(gs[1:, 0])
    ax_match.set_title("2. Matching: Window vs. Self-Template", fontsize=14, fontweight='bold')
    ax_match.set_ylim(np.min(template)-10, np.max(template)+10)
    ax_match.set_yticks([]) # Hide Y ticks for cleanliness
    
    # Resample template to window size for visual comparison if needed, 
    # but theoretically we compare resampled window to template.
    # So we plot Template (fixed) and Resampled Window (dynamic).
    x_templ = np.linspace(0, 100, len(template))
    line_template, = ax_match.plot(x_templ, template, 'b--', linewidth=2, label='Self-Derived Template')
    line_window_resampled, = ax_match.plot([], [], 'r-', linewidth=3, alpha=0.8, label='Current Window (Resampled)')
    
    text_match = ax_match.text(0.5, 0.9, '', transform=ax_match.transAxes, ha='center', fontsize=16, fontweight='bold', color='green')
    ax_match.legend(loc='lower right')
    
    # Bottom Right: Distance Profile
    ax_dist = fig.add_subplot(gs[1:, 1])
    ax_dist.set_title("3. Similarity Score (Local Minima = Hit)", fontsize=14, fontweight='bold')
    ax_dist.set_xlim(0, len(signal))
    ax_dist.set_ylim(0, 1.1)
    ax_dist.set_xlabel("Frame")
    ax_dist.set_ylabel("Diff. Score")
    
    line_dist, = ax_dist.plot([], [], color='#457B9D', linewidth=2)
    scat_peaks = ax_dist.scatter([], [], c='#E63946', s=150, marker='*', zorder=5, label='Cycle Detected')
    
    # Highlight current position
    line_curr_pos = ax_dist.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    def update(frame):
        idx = frame * scan_step
        if idx >= len(signal) - window_size:
            return line_past,
            
        # 1. Update Signal
        line_past.set_data(np.arange(idx), signal[:idx])
        rect_window.set_x(idx)
        
        # 2. Update Match Logic
        current_window = signal[idx:idx+window_size]
        curr_win_resampled = resample(current_window, len(template))
        
        line_window_resampled.set_data(x_templ, curr_win_resampled)
        
        # Check match
        # Peaks are indices in the 'indices' array which maps to signal index 1:1 here because step=1 in scan
        # We used step=1 for dist_profile generation
        
        # Check if idx is close to any peak
        # Peaks are indices in signal
        is_hit = False
        dist_to_peak = np.abs(peaks - idx)
        if np.any(dist_to_peak < 3): # Within 3 frames
            text_match.set_text("MATCH FOUND!")
            rect_window.set_edgecolor('#2A9D8F') # Green
            is_hit = True
        else:
            text_match.set_text("")
            rect_window.set_edgecolor('#E63946') # Red
            
        # 3. Update Distance
        line_dist.set_data(np.arange(idx), dist_profile[:idx])
        
        # Update peaks shown
        shown_peaks = peaks[peaks <= idx]
        if len(shown_peaks) > 0:
            scat_peaks.set_offsets(np.c_[shown_peaks, dist_profile[shown_peaks]])
        
        line_curr_pos.set_xdata([idx])
        
        return line_past, rect_window, line_window_resampled, line_dist, scat_peaks, text_match, line_curr_pos

    print(f"🎥 Rendering {num_frames} frames...")
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)
    
    # Save as GIF first (safer)
    try:
        # Prefer MP4 if ffmpeg available
        if os.system("which ffmpeg > /dev/null") == 0:
            ani.save(output_file, writer='ffmpeg', fps=30, dpi=100)
            print(f"✅ Video saved: {output_file}")
        else:
            gif_path = output_file.replace('.mp4', '.gif')
            ani.save(gif_path, writer='pillow', fps=30)
            print(f"✅ GIF saved: {gif_path}")
            
    except Exception as e:
        print(f"⚠️ Error saving: {e}")

if __name__ == "__main__":
    create_video_abstract()
