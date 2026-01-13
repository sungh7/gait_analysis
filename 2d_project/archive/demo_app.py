import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate, resample
import io

# --- AT-RTM Logic (Inline for portability) ---
def analyze_signal(signal, fs=30):
    # 1. Period Estimation
    sig_detrend = signal - np.mean(signal)
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:]
    peaks, _ = find_peaks(corr, distance=15)
    
    if len(peaks) < 2:
        period = 35
        estimated_cycle_time = 1.0
    else:
        period = peaks[1]
        estimated_cycle_time = period / fs
        
    # 2. Template Extraction
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
        return None, {}, None
        
    self_template = np.median(candidates, axis=0)
    min_idx = np.argmin(self_template)
    self_template = np.roll(self_template, -min_idx)
    
    # 3. Pattern Matching
    dists = []
    window_monitor = int(period)
    matches = []
    
    # Fast matching
    for i in range(0, len(signal)-window_monitor, 2):
        window = signal[i:i+window_monitor]
        window_norm = resample(window, 101)
        d = np.linalg.norm(window_norm - self_template)
        dists.append(d)
        
    dists = np.array(dists)
    # Find minima in distance
    peak_indices, _ = find_peaks(-dists, distance=int(period*0.7 * 0.5)) # scaled
    
    onset_indices = peak_indices * 2 # compensate for step=2
    
    # Metrics
    cycle_times = np.diff(onset_indices) / fs
    avg_stride_time = np.mean(cycle_times)
    cadence = 60 / avg_stride_time
    cv = np.std(cycle_times) / avg_stride_time * 100
    
    metrics = {
        "num_cycles": len(onset_indices),
        "avg_stride_time": avg_stride_time,
        "cadence": cadence,
        "cv": cv,
        "period_frames": period
    }
    
    return onset_indices, metrics, self_template

# --- UI ---
st.set_page_config(page_title="AT-RTM Gait Analysis", page_icon="🚶", layout="wide")

st.title("🚶 Auto-Template Resampled Template Matching (AT-RTM)")
st.caption("Robust Gait Cycle Segmentation from Monocular 2D Video")

st.markdown("""
This demo showcases the **AT-RTM algorithm** described in our research paper.
Upload a CSV file containing knee angle data to automatically extract gait parameters without manual parameter tuning.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload CSV (Time series)", type=["csv"])
    
    use_sample = st.checkbox("Use Sample Data (Subject 1)", value=True)
    
    data = None
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    elif use_sample:
        # Generate synthetic if file not found, or load local S1
        # Try local S1
        try:
            sample_path = '/data/gait/2d_project/batch_results/S1_01_improved_angles.csv'
            data = pd.read_csv(sample_path)
            st.success("Loaded Sample Subject 1")
        except:
             t = np.linspace(0, 10, 300)
             sig = 60 * np.sin(2 * np.pi * 1.0 * t) + 130 + np.random.normal(0, 2, 300)
             data = pd.DataFrame({'right_knee_angle': sig})
             st.warning("Loaded Synthetic Data")

    if data is not None:
        # Column selection
        cols = data.columns.tolist()
        # Auto-select likely column
        default_idx = 0
        for i, c in enumerate(cols):
            if 'knee' in c.lower() and 'angle' in c.lower():
                default_idx = i
                break
        
        target_col = st.selectbox("Select Joint Angle Column", cols, index=default_idx)
        signal = data[target_col].values
        
        # Trim options
        start_f, end_f = st.slider("Select Frame Range", 0, len(signal), (200, min(600, len(signal))))
        signal = signal[start_f:end_f]
        
        if st.button("Run Analysis"):
            with st.spinner("Analyzing gait patterns..."):
                onsets, metrics, template = analyze_signal(signal)
                
                if metrics:
                    st.session_state['results'] = (onsets, metrics, template, signal)
                else:
                    st.error("Analysis failed. Signal might be too short or irregular.")

with col2:
    if 'results' in st.session_state:
        onsets, metrics, template, signal = st.session_state['results']
        
        st.header("2. Analysis Report")
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Detected Cycles", f"{metrics['num_cycles']}")
        m2.metric("Avg Stride Time", f"{metrics['avg_stride_time']:.3f} s")
        m3.metric("Cadence", f"{metrics['cadence']:.1f} str/min")
        m4.metric("Variability (CV)", f"{metrics['cv']:.1f}%")
        
        # Visuals
        st.subheader("Segmentation Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(signal, label='Raw Signal', color='gray', alpha=0.7)
        for o in onsets:
            ax.axvline(o, color='r', linestyle='--', alpha=0.6)
        ax.set_title("Detected Cycle Boundaries (Heel Strikes)")
        ax.legend()
        st.pyplot(fig)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Self-Derived Template")
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.plot(template, 'b-', linewidth=2)
            ax2.set_title("Subject-Specific Pattern")
            ax2.grid(True)
            st.pyplot(fig2)
            
        with c2:
            st.markdown("### Clinical Interpretation")
            if metrics['cv'] < 3.0:
                st.success(f"✅ Low Variability ({metrics['cv']:.1f}%): Healthy, rhythmic gait.")
            elif metrics['cv'] < 6.0:
                st.warning(f"⚠️ Moderate Variability ({metrics['cv']:.1f}%): Potential instability.")
            else:
                st.error(f"❌ High Variability ({metrics['cv']:.1f}%): Risk of falls or pathology.")

st.markdown("---")
st.markdown("© 2026 AT-RTM Research Team")
