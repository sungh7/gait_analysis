"""
Train and Save Model
Trains the final multi-class Random Forest model and saves it for the web app.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

def train_and_save():
    print("Loading data...")
    df = pd.read_csv('/data/gait/gavd_extracted_features_full.csv')
    
    # 1. Filter and Label
    excluded_patterns = ['style', 'exercise', 'inebriated']
    df = df[~df['gait_pattern'].isin(excluded_patterns)].copy()
    
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
    
    feature_cols = [
        'cadence_spm', 'stride_time_s', 'velocity_mps', 'step_length_m', 'stride_length_m',
        'hip_rom', 'hip_mean', 'hip_std',
        'knee_rom', 'knee_mean', 'knee_std',
        'ankle_rom', 'ankle_mean', 'ankle_std'
    ]
    
    X = df[feature_cols].values
    y = df['class_label'].values
    
    # 2. Encode Labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Handle Missing Values (Imputation) BEFORE SMOTE
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # 4. Apply SMOTE
    print(f"Original shape: {X_imputed.shape}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_imputed, y_encoded)
    print(f"Resampled shape: {X_resampled.shape}")
    
    # 5. Create and Train Pipeline
    # Note: We don't need Imputer in pipeline if we assume input features might have NaNs, 
    # but SMOTE required clean data. For inference, we need an Imputer in the pipeline 
    # to handle potential NaNs in new data.
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training model...")
    pipeline.fit(X_resampled, y_resampled)
    
    # 6. Save Artifacts
    artifacts = {
        'pipeline': pipeline,
        'label_encoder': le,
        'features': feature_cols
    }
    
    joblib.dump(artifacts, '/data/gait/gait_classifier.joblib')
    print("✓ Model saved to /data/gait/gait_classifier.joblib")

if __name__ == "__main__":
    train_and_save()
