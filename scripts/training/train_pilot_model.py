import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os
import sys

# Import feature extraction
sys.path.append('/data/gait')
from extract_fullbody_features import extract_features

def create_dataset():
    """
    Generates a synthetic dataset of 50 samples per class.
    """
    data = []
    labels = []
    
    classes = ['normal', 'hemiplegic', 'parkinsonian']
    
    print("Generating synthetic dataset...")
    
    for label in classes:
        for i in range(50):
            # Generate mock file
            filename = f'/data/gait/temp_mock_{label}_{i}.csv'
            
            # Import generator dynamically or run script
            # Here we'll just run the script command for simplicity or import if possible
            # Let's use the script we just wrote, importing it would be cleaner but let's stick to file I/O for robustness
            
            # Actually, let's import the function since we are in python
            from generate_mock_landmarks import generate_mock_landmarks
            
            # Add some random noise to duration/amplitude to make it realistic
            # We need to modify generate_mock_landmarks to accept noise or just rely on its internal logic
            # For this pilot, exact duplicates are fine, but let's vary frame count slightly
            frames = np.random.randint(250, 350)
            
            generate_mock_landmarks(filename, num_frames=frames, gait_type=label)
            
            # Extract features
            features = extract_features(filename)
            if features:
                # Add label
                features['label'] = label
                data.append(features)
                
            # Cleanup
            if os.path.exists(filename):
                os.remove(filename)
                
    return pd.DataFrame(data)

def train_pilot_model(df):
    """
    Trains a simple RF classifier.
    """
    print(f"\nDataset Shape: {df.shape}")
    print("Class Distribution:")
    print(df['label'].value_counts())
    
    # Prepare X, y
    feature_cols = [c for c in df.columns if c not in ['label', 'filename']]
    X = df[feature_cols]
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nPilot Model Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Feature Importances:")
    print(importances)
    
    return clf, importances

if __name__ == "__main__":
    df = create_dataset()
    if not df.empty:
        train_pilot_model(df)
    else:
        print("Failed to create dataset.")
