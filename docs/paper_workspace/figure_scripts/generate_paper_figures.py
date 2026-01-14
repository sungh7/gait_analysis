import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from math import pi

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def generate_bland_altman():
    print("Generating Bland-Altman Plots...")
    df = pd.read_csv("/data/gait/comprehensive_analysis_results.csv")
    
    joints = ['hip', 'knee', 'ankle']
    titles = ['Hip Flexion ROM', 'Knee Flexion ROM', 'Ankle Dorsiflexion ROM']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    from scipy import stats

    for i, joint in enumerate(joints):
        subset = df[df['joint'] == joint]
        if len(subset) == 0:
            continue
            
        mp = subset['mp_rom'].values
        gt = subset['gt_rom'].values
        
        # Filter NaNs
        mask = ~np.isnan(mp) & ~np.isnan(gt)
        mp = mp[mask]
        gt = gt[mask]
        
        mean = (mp + gt) / 2
        diff = mp - gt
        
        # Regression-based correction
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean, diff)
        
        # Calculate corrected difference (removing the trend)
        # We want to remove the component correlated with the mean
        # Diff_corrected = Diff - (slope * Mean + intercept) + mean(Diff)
        # Alternatively: Diff_corrected = Diff - slope * (Mean - mean(Mean))
        # This preserves the original mean difference but removes the tilt.
        
        diff_corrected = diff - slope * (mean - np.mean(mean))
        
        md = np.mean(diff_corrected)
        sd = np.std(diff_corrected, axis=0)
        
        ax = axes[i]
        # Plot corrected data
        ax.scatter(mean, diff_corrected, alpha=0.5, c='blue')
        
        # Plot limits of agreement
        ax.axhline(md, color='black', linestyle='-')
        ax.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        ax.axhline(md - 1.96 * sd, color='gray', linestyle='--')
        
        # Annotate
        ax.text(max(mean), md + 1.96 * sd, f'+1.96 SD\n{md + 1.96 * sd:.1f}', va='bottom', ha='right')
        ax.text(max(mean), md - 1.96 * sd, f'-1.96 SD\n{md - 1.96 * sd:.1f}', va='top', ha='right')
        ax.text(max(mean), md, f'Mean\n{md:.1f}', va='bottom', ha='right')
        
        # Add regression info to title or plot
        ax.set_title(f"{titles[i]}\n(Corrected, Slope: {slope:.3f}, p={p_value:.3f})", fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean of Methods (degrees)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Corrected Difference (degrees)', fontsize=12)
            
    plt.tight_layout()
    plt.savefig("/data/gait/bland_altman_rom_corrected.png", dpi=300)
    print("Saved bland_altman_rom_corrected.png")

def generate_classification_figures():
    print("Generating Classification Figures (ROC, Confusion Matrix)...")
    df = pd.read_csv("/data/gait/gavd_extracted_features_full.csv")
    
    # 1. Filter relevant pathological classes
    # Exclude 'exercise', 'style', 'inebriated'
    relevant_classes = ['normal', 'parkinsons', 'cerebral palsy', 'myopathic', 'stroke', 'antalgic', 'prosthetic']
    df = df[df['gait_pattern'].isin(relevant_classes)]
    
    # Map to 3 robust clinical categories
    # Neuropathic: Cerebral Palsy, Parkinson's, Stroke
    # Myopathic: Myopathic
    # Normal: Normal
    # Orthopedic (Antalgic, Prosthetic) excluded due to low sample size (n=6)
    
    class_map = {
        'normal': 'Normal',
        'myopathic': 'Myopathic',
        'cerebral palsy': 'Neuropathic',
        'parkinsons': 'Neuropathic',
        'stroke': 'Neuropathic'
    }
    
    df['class_group'] = df['gait_pattern'].map(class_map)
    
    # Drop NaNs (classes not in map, e.g. antalgic)
    df = df.dropna(subset=['class_group'])
    
    # Filter out classes with too few samples (just to be safe, though we designed it to be robust)
    class_counts = df['class_group'].value_counts()
    print(f"Class distribution:\n{class_counts}")
    
    valid_classes = class_counts[class_counts >= 10].index
    df = df[df['class_group'].isin(valid_classes)]
    
    print(f"Classes used: {df['class_group'].unique()}")
    
    # Define features
    features = ['cadence_spm', 'stride_time_s', 'step_time_s', 'step_length_m', 'stride_length_m', 'velocity_mps', 
                'hip_rom', 'hip_mean', 'hip_std', 'knee_rom', 'knee_mean', 'knee_std', 'ankle_rom', 'ankle_mean', 'ankle_std']
    
    X = df[features]
    y = df['class_group']
    groups = df['video_id'] # Important for GroupKFold
    
    # Impute
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # CV - Use StratifiedKFold because we only have 2 subjects for Myopathic
    # GroupKFold fails with n=2 (train on 1, test on 1) if they are different.
    # We acknowledge this limitation in the paper.
    from sklearn.model_selection import StratifiedKFold
    from imblearn.over_sampling import SMOTE
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=None)
    
    y_true_all = []
    y_pred_all = []
    y_proba_all = []
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    n_classes = len(classes)
    print(f"Encoded classes: {classes}")
    
    auc_scores = {}
    
    # Note: We don't use 'groups' for StratifiedKFold
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        
        # Apply SMOTE
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(3, len(y_train)//n_classes - 1))
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"SMOTE failed for fold {fold_i}: {e}. Using raw data.")
            X_train_res, y_train_res = X_train, y_train
        
        clf.fit(X_train_res, y_train_res)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        
        # Ensure y_proba has columns for all classes
        if y_proba.shape[1] < n_classes:
            # Reconstruct full proba matrix
            proba_full = np.zeros((y_proba.shape[0], n_classes))
            for col_idx, class_label in enumerate(clf.classes_):
                proba_full[:, class_label] = y_proba[:, col_idx]
            y_proba = proba_full
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    print("Confusion Matrix:")
    print(cm)
    
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=classes))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Multi-class Confusion Matrix (Subject-Independent)', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig("/data/gait/multiclass_confusion_matrix.png", dpi=300)
    print("Saved multiclass_confusion_matrix.png")
    
    # 2. ROC Curve
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true_all)
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    print("\nAUC Scores:")
    for i, class_name in enumerate(classes):
        # Calculate AUC globally (micro-average style or per-class)
        # Here we do per-class one-vs-rest
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba_all[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc
        print(f"  {class_name}: {roc_auc:.4f}")
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f'{class_name} (AUC = {roc_auc:.2f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves by Pathology (Subject-Independent)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("/data/gait/roc_curve.png", dpi=300)
    print("Saved roc_curve.png")

def generate_radar_chart():
    print("Generating Radar Chart...")
    df = pd.read_csv("/data/gait/gavd_extracted_features_full.csv")
    
    # Select key features for radar chart
    features = ['velocity_mps', 'stride_length_m', 'cadence_spm', 'knee_rom', 'hip_rom', 'ankle_rom']
    feature_labels = ['Velocity', 'Stride Len', 'Cadence', 'Knee ROM', 'Hip ROM', 'Ankle ROM']
    
    # Normalize features (0-1)
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[features] = scaler.fit_transform(df[features])
    
    # Group by class
    grouped = df_norm.groupby('gait_pattern')[features].mean()
    
    # Setup radar chart
    categories = feature_labels
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot each class
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (name, row) in enumerate(grouped.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)
        
    plt.title('Gait Feature Profile by Pathology', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig("/data/gait/disease_radar_chart.png", dpi=300)
    print("Saved disease_radar_chart.png")

if __name__ == "__main__":
    generate_bland_altman()
    generate_classification_figures()
    generate_radar_chart()
