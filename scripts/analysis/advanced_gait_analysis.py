"""
Advanced Pathological Gait Analysis
Performs data refinement, advanced classification, and clinical statistical analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

def load_and_refine_data(csv_path):
    """Load and filter data for clinical relevance"""
    df = pd.read_csv(csv_path)
    
    print(f"Original dataset size: {len(df)}")
    
    # Define groups
    healthy_patterns = ['normal']
    pathological_patterns = [
        'myopathic', 'cerebral palsy', 'parkinsons', 
        'antalgic', 'stroke', 'prosthetic', 'abnormal'
    ]
    excluded_patterns = ['style', 'exercise', 'inebriated']
    
    # Filter data
    df_healthy = df[df['gait_pattern'].isin(healthy_patterns)].copy()
    df_patho = df[df['gait_pattern'].isin(pathological_patterns)].copy()
    
    df_healthy['group'] = 'Healthy'
    df_patho['group'] = 'Pathological'
    
    # Combine
    df_refined = pd.concat([df_healthy, df_patho])
    
    print(f"Refined dataset size: {len(df_refined)}")
    print(f"  - Healthy: {len(df_healthy)}")
    print(f"  - Pathological: {len(df_patho)}")
    print(f"  - Excluded: {len(df) - len(df_refined)} (Style, Exercise, etc.)")
    
    return df_refined

def train_binary_classifier(df, feature_cols):
    """Train Healthy vs Pathological classifier"""
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION (Healthy vs Pathological)")
    print("="*60)
    
    X = df[feature_cols].values
    y = (df['group'] == 'Pathological').astype(int) # 1 for Pathological
    groups = df['video_id'].values
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost (Gradient Boosting)': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, clf in models.items():
        logo = LeaveOneGroupOut()
        y_pred = cross_val_predict(clf, X_scaled, y, groups=groups, cv=logo)
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(classification_report(y, y_pred, target_names=['Healthy', 'Pathological']))
        
        results[name] = {'accuracy': acc, 'model': clf, 'y_pred': y_pred}
        
        # Save confusion matrix for best model
        if name == 'Random Forest': # Usually robust
            cm = confusion_matrix(y, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Healthy', 'Pathological'],
                       yticklabels=['Healthy', 'Pathological'])
            plt.title('Confusion Matrix: Healthy vs Pathological')
            plt.tight_layout()
            plt.savefig('/data/gait/binary_confusion_matrix.png')
            
            # Feature Importance
            clf.fit(X_scaled, y)
            feat_imp = pd.DataFrame({
                'feature': feature_cols,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            print("\nTop 10 Important Features:")
            print(feat_imp.head(10))
            
    return results

def analyze_clinical_features(df, feature_cols):
    """Perform statistical analysis and visualization"""
    print("\n" + "="*60)
    print("CLINICAL FEATURE ANALYSIS")
    print("="*60)
    
    # 1. T-test (Healthy vs Pathological)
    print("\nStatistical Significance (T-test):")
    significant_features = []
    
    for col in feature_cols:
        healthy_vals = df[df['group'] == 'Healthy'][col].dropna()
        patho_vals = df[df['group'] == 'Pathological'][col].dropna()
        
        t_stat, p_val = stats.ttest_ind(healthy_vals, patho_vals, equal_var=False)
        
        if p_val < 0.05:
            star = "*" if p_val < 0.05 else ""
            star += "*" if p_val < 0.01 else ""
            star += "*" if p_val < 0.001 else ""
            
            print(f"{col:20s}: p={p_val:.4f} {star} (Healthy {healthy_vals.mean():.2f} vs Patho {patho_vals.mean():.2f})")
            significant_features.append(col)
            
    # 2. Box Plots for top features
    top_features = ['cadence_spm', 'stride_time_s', 'velocity_mps', 'knee_rom', 'hip_rom']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(top_features):
        if col in df.columns:
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='group', y=col, data=df, palette="Set2")
            plt.title(col)
            
    plt.tight_layout()
    plt.savefig('/data/gait/clinical_boxplots.png')
    print("\n✓ Saved clinical boxplots")
    
    # 3. Radar Chart for Disease Profiling
    # Normalize features to 0-1 range for radar chart
    scaler = StandardScaler()
    df_norm = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(df[feature_cols].mean()))
    
    # Calculate mean profile for each disease
    diseases = ['normal', 'myopathic', 'cerebral palsy', 'parkinsons']
    radar_features = ['velocity_mps', 'cadence_spm', 'stride_length_m', 'hip_rom', 'knee_rom', 'ankle_rom']
    
    # Check if we have enough data for these diseases
    available_diseases = [d for d in diseases if d in df['gait_pattern'].unique()]
    
    if len(available_diseases) > 1:
        # Create Radar Chart
        categories = radar_features
        N = len(categories)
        
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        for disease in available_diseases:
            values = df_norm[df_norm['gait_pattern'] == disease][radar_features].mean().values.flatten().tolist()
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=disease)
            ax.fill(angles, values, alpha=0.1)
            
        plt.xticks(angles[:-1], categories)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title("Gait Profile by Disease (Z-score normalized)")
        plt.savefig('/data/gait/disease_radar_chart.png')
        print("\n✓ Saved disease radar chart")

def main():
    # Load data
    csv_path = '/data/gait/gavd_extracted_features_full.csv'
    df = load_and_refine_data(csv_path)
    
    # Define features
    feature_cols = [
        'cadence_spm', 'stride_time_s', 'velocity_mps', 'step_length_m', 'stride_length_m',
        'hip_rom', 'hip_mean', 'hip_std',
        'knee_rom', 'knee_mean', 'knee_std',
        'ankle_rom', 'ankle_mean', 'ankle_std'
    ]
    
    # Run analysis
    train_binary_classifier(df, feature_cols)
    analyze_clinical_features(df, feature_cols)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
