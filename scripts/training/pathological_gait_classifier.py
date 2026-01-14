"""
Pathological Gait Binary Classifier
Classifies gait as Normal vs Abnormal using extracted features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(csv_path):
    """Load features and prepare for classification"""
    df = pd.read_csv(csv_path)
    
    # Create binary target: Normal vs Abnormal
    # Consider 'normal' gait_pattern as Normal (0), others as Abnormal (1)
    df['label'] = df['gait_pattern'].apply(lambda x: 0 if x == 'normal' else 1)
    df['label_name'] = df['label'].map({0: 'Normal', 1: 'Abnormal'})
    
    # Select features
    feature_cols = [
        'cadence_spm', 'stride_time_s', 'velocity_mps',
        'hip_rom', 'hip_mean', 'hip_std',
        'knee_rom', 'knee_mean', 'knee_std',
        'ankle_rom', 'ankle_mean', 'ankle_std'
    ]
    
    # Remove samples with missing features
    df_clean = df[['seq', 'video_id', 'side', 'gait_pattern', 'label', 'label_name'] + feature_cols].dropna()
    
    print(f"Loaded {len(df_clean)} samples with complete features")
    print(f"\nClass distribution:")
    print(df_clean['label_name'].value_counts())
    
    return df_clean, feature_cols

def train_and_evaluate(df, feature_cols):
    """Train classifier with Leave-One-Video-Out CV"""
    
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['video_id'].values  # Group by video to avoid data leakage
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    
    # Leave-One-Group-Out Cross-Validation (one video at a time)
    logo = LeaveOneGroupOut()
    y_pred = cross_val_predict(clf, X_scaled, y, groups=groups, cv=logo)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    print("\n" + "="*60)
    print("BINARY CLASSIFICATION RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Normal', 'Abnormal']))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('/data/gait/gavd_confusion_matrix.png', dpi=150)
    print("\n✓ Confusion matrix saved to gavd_confusion_matrix.png")
    
    # Feature Importance (train on all data for interpretation)
    clf.fit(X_scaled, y)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance for Pathological Gait Detection')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('/data/gait/gavd_feature_importance.png', dpi=150)
    print("\n✓ Feature importance plot saved to gavd_feature_importance.png")
    
    return clf, scaler, feature_importance

def main():
    print("="*60)
    print("Pathological Gait Binary Classification")
    print("="*60)
    
    # Load data
    df, feature_cols = load_and_prepare_data('/data/gait/gavd_extracted_features_test.csv')
    
    # Train and evaluate
    clf, scaler, feature_importance = train_and_evaluate(df, feature_cols)
    
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
