"""
Paper Assets Generator
Generates high-quality figures and statistical tables for the research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy import stats

def generate_demographics_table():
    """Generate Table 1: Participant Demographics"""
    print("\nGenerating Table 1 (Demographics)...")
    df = pd.read_csv('/data/gait/body_measurements.csv')
    
    # Calculate BMI
    df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    
    stats_df = df[['age', 'height_cm', 'weight_kg', 'bmi']].describe().T
    stats_df['mean_sd'] = stats_df.apply(lambda x: f"{x['mean']:.1f} ± {x['std']:.1f}", axis=1)
    
    print(stats_df[['mean_sd']])
    return df

def plot_bland_altman(df_results):
    """Generate Bland-Altman Plot for ROM"""
    print("\nGenerating Bland-Altman Plot...")
    
    joints = ['hip', 'knee', 'ankle']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, joint in enumerate(joints):
        joint_data = df_results[df_results['joint'] == joint]
        
        mp_rom = joint_data['mp_rom']
        gt_rom = joint_data['gt_rom']
        
        mean = (mp_rom + gt_rom) / 2
        diff = mp_rom - gt_rom
        md = np.mean(diff)
        sd = np.std(diff, axis=0)
        
        ax = axes[i]
        ax.scatter(mean, diff, alpha=0.5)
        ax.axhline(md, color='gray', linestyle='--')
        ax.axhline(md + 1.96*sd, color='gray', linestyle=':')
        ax.axhline(md - 1.96*sd, color='gray', linestyle=':')
        
        ax.set_title(f'{joint.capitalize()} ROM')
        ax.set_xlabel('Mean ROM (°)')
        ax.set_ylabel('Difference (MP - Vicon) (°)')
        ax.text(mean.max(), md + 1.96*sd, f'+1.96 SD: {md + 1.96*sd:.1f}', va='bottom', ha='right')
        ax.text(mean.max(), md - 1.96*sd, f'-1.96 SD: {md - 1.96*sd:.1f}', va='top', ha='right')
        ax.text(mean.max(), md, f'Mean: {md:.1f}', va='bottom', ha='right')
        
    plt.tight_layout()
    plt.savefig('/data/gait/bland_altman_rom.png', dpi=300)
    print("✓ Saved bland_altman_rom.png")

def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(group1) - np.mean(group2)) / pooled_se
    return d

def analyze_gavd_stats():
    """Analyze GAVD Statistics and generate ROC"""
    print("\nAnalyzing GAVD Statistics...")
    df = pd.read_csv('/data/gait/gavd_extracted_features_full.csv')
    
    # Define groups strictly
    healthy_patterns = ['normal']
    patho_patterns = ['myopathic', 'cerebral palsy', 'parkinsons', 'antalgic', 'stroke', 'abnormal']
    
    healthy = df[df['gait_pattern'].isin(healthy_patterns)].copy()
    patho = df[df['gait_pattern'].isin(patho_patterns)].copy()
    
    print(f"Healthy samples: {len(healthy)}")
    print(f"Pathological samples: {len(patho)}")
    
    # Cohen's d
    features = ['velocity_mps', 'stride_length_m', 'knee_rom']
    print("\nEffect Sizes (Cohen's d):")
    for feat in features:
        # Drop NaNs for specific feature
        g1 = healthy[feat].dropna()
        g2 = patho[feat].dropna()
        
        if len(g1) > 0 and len(g2) > 0:
            d = calculate_cohens_d(g1, g2)
            print(f"{feat}: d = {d:.2f} (Healthy: {g1.mean():.2f}, Patho: {g2.mean():.2f})")
        else:
            print(f"{feat}: Insufficient data")
        
    # ROC Curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.impute import SimpleImputer
    
    # Combine and prepare for ML
    data = pd.concat([healthy, patho])
    X = data[features].values
    y = np.concatenate([np.zeros(len(healthy)), np.ones(len(patho))])
    
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    y_probas = cross_val_predict(clf, X, y, cv=5, method='predict_proba')
    
    fpr, tpr, _ = roc_curve(y, y_probas[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Pathological Detection)')
    plt.legend(loc="lower right")
    plt.savefig('/data/gait/roc_curve.png', dpi=300)
    print("✓ Saved roc_curve.png")

def main():
    # 1. Demographics
    generate_demographics_table()
    
    # 2. Bland-Altman
    df_results = pd.read_csv('/data/gait/comprehensive_analysis_results.csv')
    plot_bland_altman(df_results)
    
    # 3. GAVD Stats & ROC
    analyze_gavd_stats()

if __name__ == "__main__":
    main()
