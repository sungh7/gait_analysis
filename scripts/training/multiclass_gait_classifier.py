"""
Multi-class Pathological Gait Classifier
Classifies gait into 4 categories: Normal, Myopathic, Cerebral Palsy, Other Pathological
Uses SMOTE for handling class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(csv_path):
    """Load and prepare data for 4-class classification"""
    df = pd.read_csv(csv_path)
    
    # Filter out non-clinical patterns
    excluded_patterns = ['style', 'exercise', 'inebriated']
    df = df[~df['gait_pattern'].isin(excluded_patterns)].copy()
    
    # Define 4 classes
    def map_class(pattern):
        if pattern == 'normal':
            return 'Normal'
        elif pattern == 'myopathic':
            return 'Myopathic'
        elif pattern == 'cerebral palsy':
            return 'Cerebral Palsy'
        else:
            return 'Other Pathological'
            
    df['class_label'] = df['gait_pattern'].apply(map_class)
    
    print("Class Distribution:")
    print(df['class_label'].value_counts())
    
    return df

def train_multiclass_model(df, feature_cols):
    """Train model with SMOTE and Stratified K-Fold"""
    
    X = df[feature_cols].values
    y = df['class_label'].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Storage for results
    y_true_all = []
    y_pred_all = []
    feature_importances = []
    
    print("\nStarting 5-Fold Cross-Validation with SMOTE...")
    
    fold = 1
    for train_index, test_index in skf.split(X_scaled, y_encoded):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Apply SMOTE only to training data
        # k_neighbors must be smaller than the smallest class size (minus 1)
        # Smallest class is CP (n=10), so in 4/5 split, n=8. k=5 is safe.
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Train Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train_resampled, y_train_resampled)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        feature_importances.append(clf.feature_importances_)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"Fold {fold}: Accuracy = {acc:.3f}")
        fold += 1
        
    # Overall Results
    overall_acc = accuracy_score(y_true_all, y_pred_all)
    print("\n" + "="*60)
    print(f"OVERALL ACCURACY: {overall_acc:.3f}")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Multi-class Confusion Matrix (SMOTE Augmented)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('/data/gait/multiclass_confusion_matrix.png')
    print("✓ Saved confusion matrix")
    
    # Average Feature Importance
    avg_importance = np.mean(feature_importances, axis=0)
    feat_imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features for Multi-class Classification:")
    print(feat_imp_df.head(10))
    
    # Save feature importance plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feat_imp_df.head(15))
    plt.title('Feature Importance for Disease Classification')
    plt.tight_layout()
    plt.savefig('/data/gait/multiclass_feature_importance.png')
    print("✓ Saved feature importance plot")
    
    return feat_imp_df

def main():
    csv_path = '/data/gait/gavd_extracted_features_full.csv'
    df = load_and_prepare_data(csv_path)
    
    feature_cols = [
        'cadence_spm', 'stride_time_s', 'velocity_mps', 'step_length_m', 'stride_length_m',
        'hip_rom', 'hip_mean', 'hip_std',
        'knee_rom', 'knee_mean', 'knee_std',
        'ankle_rom', 'ankle_mean', 'ankle_std'
    ]
    
    train_multiclass_model(df, feature_cols)

if __name__ == "__main__":
    main()
