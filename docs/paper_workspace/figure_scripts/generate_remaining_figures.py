import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
import seaborn as sns

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_metrics(mp, gt):
    min_len = min(len(mp), len(gt))
    mp = np.array(mp[:min_len])
    gt = np.array(gt[:min_len])
    rmse = np.sqrt(np.mean((mp - gt)**2))
    if np.std(mp) == 0 or np.std(gt) == 0:
        corr = 0
    else:
        corr, _ = pearsonr(mp, gt)
    return corr, rmse

def get_calibration_params(mp, gt):
    # Simple linear regression to get slope/intercept
    min_len = min(len(mp), len(gt))
    mp = np.array(mp[:min_len])
    gt = np.array(gt[:min_len])
    
    slope, intercept, r_value, p_value, std_err = linregress(mp, gt)
    return slope, intercept

def plot_figure11_parameters(data, output_path):
    # Figure 11: Subject-Specific Calibration Parameters
    # Panel A: Hip Slope vs Intercept
    # Panel B: Ankle Slope vs Intercept
    # Panel C: Slope Distribution (Both)
    
    hip_params = {'slope': [], 'intercept': [], 'id': []}
    ankle_params = {'slope': [], 'intercept': [], 'id': []}
    
    for subj_id, subj_data in data.items():
        # Hip
        if 'hip' in subj_data:
            raw_mp = subj_data['hip']['raw_mp_waveform']
            gt = subj_data['hip']['gt_waveform']
            s, i = get_calibration_params(raw_mp, gt)
            hip_params['slope'].append(s)
            hip_params['intercept'].append(i)
            hip_params['id'].append(subj_id)
            
        # Ankle (Use Stage 2 aligned data if possible, or just raw for illustration of need)
        # To show "Why per-subject is needed", we should compare Raw MP to GT.
        # But Ankle raw has sign issues. Let's assume sign is corrected (Stage 1).
        if 'ankle' in subj_data:
            raw_mp = np.array(subj_data['ankle']['raw_mp_waveform'])
            # Auto-sign correct for fair slope estimation
            gt = np.array(subj_data['ankle']['gt_waveform'])
            
            # Check correlation to decide sign
            r_pos, _ = pearsonr(raw_mp[:min(len(raw_mp), len(gt))], gt[:min(len(raw_mp), len(gt))])
            if r_pos < 0:
                raw_mp = -raw_mp
                
            s, i = get_calibration_params(raw_mp, gt)
            ankle_params['slope'].append(s)
            ankle_params['intercept'].append(i)
            ankle_params['id'].append(subj_id)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3)
    
    # Panel A: Hip Scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(hip_params['slope'], hip_params['intercept'], c='blue', alpha=0.6, s=50)
    ax1.set_title('(A) Hip Calibration Parameters', fontweight='bold')
    ax1.set_xlabel('Slope (Gain)')
    ax1.set_ylabel('Intercept (Offset)')
    ax1.grid(True, alpha=0.3)
    # Add mean point
    ax1.scatter(np.mean(hip_params['slope']), np.mean(hip_params['intercept']), c='red', marker='X', s=100, label='Mean')
    ax1.legend()
    
    # Panel B: Ankle Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(ankle_params['slope'], ankle_params['intercept'], c='green', alpha=0.6, s=50)
    ax2.set_title('(B) Ankle Calibration Parameters', fontweight='bold')
    ax2.set_xlabel('Slope (Gain)')
    ax2.set_ylabel('Intercept (Offset)')
    ax2.grid(True, alpha=0.3)
    ax2.scatter(np.mean(ankle_params['slope']), np.mean(ankle_params['intercept']), c='red', marker='X', s=100, label='Mean')
    ax2.legend()
    
    # Panel C: Slope Distributions
    ax3 = fig.add_subplot(gs[0, 2])
    sns.kdeplot(hip_params['slope'], ax=ax3, color='blue', fill=True, alpha=0.2, label='Hip Slope')
    sns.kdeplot(ankle_params['slope'], ax=ax3, color='green', fill=True, alpha=0.2, label='Ankle Slope')
    ax3.set_title('(C) Parameter Variability', fontweight='bold')
    ax3.set_xlabel('Slope Value')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Figure 11 to {output_path}")

def plot_figure12_errors(data, output_path):
    # Figure 12: Individual Error Distribution
    # Ranked bar chart of RMSE for Hip and Ankle
    
    subjects = []
    hip_rmse = []
    ankle_rmse = []
    
    for subj_id, subj_data in data.items():
        subjects.append(subj_id)
        
        # Hip
        if 'hip' in subj_data:
            _, rmse = calculate_metrics(subj_data['hip']['mp_waveform'], subj_data['hip']['gt_waveform'])
            hip_rmse.append(rmse)
        else:
            hip_rmse.append(0)
            
        # Ankle
        if 'ankle' in subj_data:
            _, rmse = calculate_metrics(subj_data['ankle']['mp_waveform'], subj_data['ankle']['gt_waveform'])
            ankle_rmse.append(rmse)
        else:
            ankle_rmse.append(0)
            
    # Sort by Ankle RMSE (since it's the harder joint)
    sorted_indices = np.argsort(ankle_rmse)
    subjects = np.array(subjects)[sorted_indices]
    hip_rmse = np.array(hip_rmse)[sorted_indices]
    ankle_rmse = np.array(ankle_rmse)[sorted_indices]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(subjects))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, hip_rmse, width, label='Hip RMSE', color='blue', alpha=0.7)
    rects2 = ax.bar(x + width/2, ankle_rmse, width, label='Ankle RMSE', color='green', alpha=0.7)
    
    ax.set_ylabel('RMSE (degrees)')
    ax.set_title('Individual Subject Performance (Ranked by Ankle Error)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add mean lines
    ax.axhline(np.mean(hip_rmse), color='blue', linestyle='--', alpha=0.5, label=f'Mean Hip: {np.mean(hip_rmse):.1f}')
    ax.axhline(np.mean(ankle_rmse), color='green', linestyle='--', alpha=0.5, label=f'Mean Ankle: {np.mean(ankle_rmse):.1f}')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved Figure 12 to {output_path}")

def main():
    json_path = "/data/gait/final_validation_results.json"
    output_dir = "/data/gait/figures"
    
    data = load_data(json_path)
    
    plot_figure11_parameters(data, f"{output_dir}/Figure11_Parameters.png")
    plot_figure12_errors(data, f"{output_dir}/Figure12_ErrorDist.png")

if __name__ == "__main__":
    main()
