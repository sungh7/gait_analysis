import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def plot_bland_altman(ax, data1, data2, title, unit="deg"):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # MP - GT
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    
    ax.scatter(mean, diff, alpha=0.1, s=2, color='blue')
    ax.axhline(md, color='red', linestyle='-')
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--')
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
    # Add text labels
    ax.text(np.max(mean), md + 1.96*sd, f'+1.96 SD: {md + 1.96*sd:.2f}', va='bottom', ha='right', fontsize=8)
    ax.text(np.max(mean), md - 1.96*sd, f'-1.96 SD: {md - 1.96*sd:.2f}', va='top', ha='right', fontsize=8)
    ax.text(np.max(mean), md, f'Mean: {md:.2f}', va='bottom', ha='right', fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel(f"Mean ({unit})")
    ax.set_ylabel(f"Difference (MP - GT) ({unit})")

def generate_bland_altman_figure(data, output_path):
    hip_mp_raw = []
    hip_mp_cal = []
    hip_gt = []
    
    ankle_mp_raw = []
    ankle_mp_cal = []
    ankle_gt = []
    
    for subj_id, subj_data in data.items():
        if 'hip' in subj_data:
            hip_mp_cal.extend(subj_data['hip']['mp_waveform'])
            hip_mp_raw.extend(subj_data['hip']['raw_mp_waveform'])
            hip_gt.extend(subj_data['hip']['gt_waveform'])
        if 'ankle' in subj_data:
            ankle_mp_cal.extend(subj_data['ankle']['mp_waveform'])
            ankle_mp_raw.extend(subj_data['ankle']['raw_mp_waveform'])
            ankle_gt.extend(subj_data['ankle']['gt_waveform'])
            
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Row 1: Hip
    plot_bland_altman(axes[0, 0], hip_mp_raw, hip_gt, "Hip: Before Calibration", unit="deg")
    plot_bland_altman(axes[0, 1], hip_mp_cal, hip_gt, "Hip: After Calibration", unit="deg")
    
    # Row 2: Ankle
    plot_bland_altman(axes[1, 0], ankle_mp_raw, ankle_gt, "Ankle: Before Calibration", unit="deg")
    plot_bland_altman(axes[1, 1], ankle_mp_cal, ankle_gt, "Ankle: After Calibration", unit="deg")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Bland-Altman plot to {output_path}")

def generate_parameter_distribution_figure(data, output_path):
    hip_slopes = []
    hip_intercepts = []
    ankle_slopes = []
    ankle_intercepts = []
    
    for subj_id, subj_data in data.items():
        if 'hip' in subj_data:
            hip_slopes.append(subj_data['hip']['slope'])
            hip_intercepts.append(subj_data['hip']['intercept'])
        if 'ankle' in subj_data:
            ankle_slopes.append(subj_data['ankle']['slope'])
            ankle_intercepts.append(subj_data['ankle']['intercept'])
            
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Hip Slope
    axes[0, 0].hist(hip_slopes, bins=10, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Hip Slope Distribution")
    axes[0, 0].set_xlabel("Slope")
    axes[0, 0].set_ylabel("Count")
    
    # Hip Intercept
    axes[0, 1].hist(hip_intercepts, bins=10, color='skyblue', edgecolor='black')
    axes[0, 1].set_title("Hip Intercept Distribution")
    axes[0, 1].set_xlabel("Intercept (deg)")
    
    # Ankle Slope
    axes[1, 0].hist(ankle_slopes, bins=10, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title("Ankle Slope Distribution")
    axes[1, 0].set_xlabel("Slope")
    axes[1, 0].set_ylabel("Count")
    
    # Ankle Intercept
    axes[1, 1].hist(ankle_intercepts, bins=10, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title("Ankle Intercept Distribution")
    axes[1, 1].set_xlabel("Intercept (deg)")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Parameter Distribution plot to {output_path}")

def draw_box(ax, x, y, width, height, text, color='white'):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=1, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return x + width/2, y, x + width/2, y + height

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.5))

def generate_schematic_figure(output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define layout
    box_w = 2.5
    box_h = 1.0
    x_center = 5
    y_start = 9
    y_gap = 1.5
    
    # 1. Input
    cx1, cy1_bot, cx1_top, cy1_top = draw_box(ax, x_center - box_w/2, y_start - box_h, box_w, box_h, "Raw Video Input\n(Smartphone)", color='#E0E0E0')
    
    # 2. Pose Extraction
    y_curr = y_start - y_gap - box_h
    cx2, cy2_bot, cx2_top, cy2_top = draw_box(ax, x_center - box_w/2, y_curr, box_w, box_h, "MediaPipe Pose\n(2D Landmarks)", color='#D1C4E9')
    draw_arrow(ax, cx1, cy1_bot, cx2_top, cy2_top)
    
    # 3. Sign Correction
    y_curr -= y_gap
    cx3, cy3_bot, cx3_top, cy3_top = draw_box(ax, x_center - box_w/2, y_curr, box_w, box_h, "Sign Correction\n(Correlation Check)", color='#BBDEFB')
    draw_arrow(ax, cx2, cy2_bot, cx3_top, cy3_top)
    
    # 4. DTW Alignment
    y_curr -= y_gap
    cx4, cy4_bot, cx4_top, cy4_top = draw_box(ax, x_center - box_w/2, y_curr, box_w, box_h, "DTW Alignment\n(Temporal Warping)", color='#C8E6C9')
    draw_arrow(ax, cx3, cy3_bot, cx4_top, cy4_top)
    
    # 5. Deming Regression (Fix Amplitude/Offset)
    y_curr -= y_gap
    cx5, cy5_bot, cx5_top, cy5_top = draw_box(ax, x_center - box_w/2, y_curr, box_w, box_h, "Per-Subject\nDeming Regression\n(Fix Amp/Offset)", color='#FFECB3')
    draw_arrow(ax, cx4, cy4_bot, cx5_top, cy5_top)
    
    # 6. Output
    y_curr -= y_gap
    cx6, cy6_bot, cx6_top, cy6_top = draw_box(ax, x_center - box_w/2, y_curr, box_w, box_h, "Calibrated\nJoint Angles", color='#FFCCBC')
    draw_arrow(ax, cx5, cy5_bot, cx6_top, cy6_top)
    
    # Side annotations
    ax.text(x_center + box_w/2 + 0.5, y_start - box_h/2, "60 FPS", va='center', fontsize=9, color='gray')
    ax.text(x_center + box_w/2 + 0.5, y_start - y_gap - box_h - box_h/2, "X, Y Coordinates", va='center', fontsize=9, color='gray')
    ax.text(x_center + box_w/2 + 0.5, y_start - 2*y_gap - 2*box_h - box_h/2, "Fix Inversions", va='center', fontsize=9, color='gray')
    ax.text(x_center + box_w/2 + 0.5, y_start - 3*y_gap - 3*box_h - box_h/2, "Match Gait Events", va='center', fontsize=9, color='gray')
    # Removed dangling annotation, integrated into box 5
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Schematic to {output_path}")

def main():
    json_path = "/data/gait/final_validation_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return
        
    data = load_data(json_path)
    
    output_dir = "/data/gait/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_bland_altman_figure(data, os.path.join(output_dir, "Figure5_BlandAltman.png"))
    generate_schematic_figure(os.path.join(output_dir, "Figure6_Pipeline.png"))
    generate_parameter_distribution_figure(data, os.path.join(output_dir, "Figure7_Parameters.png"))

if __name__ == "__main__":
    main()
