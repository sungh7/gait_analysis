
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import cv2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
from scipy import odr

CSV_PATH = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"
DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def get_video_duration(video_path):
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0: return frame_count / fps
    return None

def deming_regression(x, y):
    # Orthogonal Distance Regression (Error in both X and Y)
    def f(B, x):
        return B[0]*x + B[1]
    linear = odr.Model(f)
    mydata = odr.Data(x, y)
    myodr = odr.ODR(mydata, linear, beta0=[1., 0.])
    myoutput = myodr.run()
    return myoutput.beta[0], myoutput.beta[1]

def run_experiment():
    print("Loading data and computing scalar features...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Feature Engineering
    cadence_list = []
    height_list = []
    weight_list = []
    age_list = []
    
    for idx, row in df.iterrows():
        sid = str(row['subject'])
        
        # Normalize to integer string (e.g. 1.0 -> "1", 1 -> "1")
        try:
            sid_int = int(float(sid))
            sid_dir = str(sid_int)
            sid_str = f"{sid_int:02d}"
        except:
            continue # Skip invalid IDs
        
        # A. Calculate MP Cadence (Steps / Minute)
        # Try multiple path patterns
        video_candidates = [
            f"{DATA_DIR}/{sid_dir}/{sid_dir}-2.mp4",
            f"{DATA_DIR}/{sid_dir}/{sid_dir}.mp4",
            f"{DATA_DIR}/{sid_dir}/video.mp4"
        ]
        video_path = None
        for p in video_candidates:
            if os.path.exists(p):
                video_path = p
                break

        duration = get_video_duration(video_path)
        if duration:
            # Cadence = (Cycles * 2 [for steps? no, stride is same side] ) / Duration_min
            # Wait, cadence usually steps/min. 1 stride = 2 steps.
            # But MP counts 'cycles' (strides).
            steps = row['mp_count'] * 2 
            cadence = steps / (duration / 60)
        else:
            cadence = np.nan
        cadence_list.append(cadence)
        
        # B. Metadata
        json_path = f"{DATA_DIR}/processed_new/S1_{sid_str}_info.json"
        h, w, a = np.nan, np.nan, np.nan
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    info = json.load(f)
                    demo = info.get('demographics', {})
                    h = demo.get('height_cm', np.nan)
                    w = demo.get('weight_kg', demo.get('weight_cm', np.nan))
                    a = demo.get('age', np.nan)
            except: pass
        height_list.append(h)
        weight_list.append(w)
        age_list.append(a)
        
    df['mp_cadence'] = cadence_list
    df['height'] = height_list
    df['weight'] = weight_list
    df['age'] = age_list
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Impute missing scalar features with Mean (instead of dropping rows)
    for col in ['mp_cadence', 'height', 'weight', 'age', 'bmi']:
        df[col] = df[col].fillna(df[col].mean())

    # Filter valid subjects (Relaxed correlation > 0.3 to get N= ~10-15)
    # df_clean = df[df['correlation'] > 0.5].dropna().reset_index(drop=True)
    df_clean = df.dropna(subset=['gt_rom']).reset_index(drop=True) # Use ALL data for now to test "Global" even on bad tracking?
    # No, bad tracking (shape mismatch) is unfixable by scale.
    # Let's stick to r > 0.5 but debug confirmed N=6.
    # User wants to know "Is there a regression?". We should use ALL N=21 valid subjects from QC.
    # The 'correlation' in CSV is from comparison against Vicon.
    # Wait, 'final_benchmarks.csv' ALREADY filtered for quality (N=21 out of 26).
    # So we should use ALL rows in the CSV.
    df_clean = df.reset_index(drop=True)
    print(f"Valid Subjects for Regression (N={len(df_clean)})")
    
    X = df_clean[['mp_rom', 'mp_cadence', 'height', 'weight', 'age', 'bmi']].values
    y = df_clean['gt_rom'].values
    
    # Models
    models = {
        "Linear": LinearRegression(),
        "Poly (Deg 2)": "poly2",
        "SVR (RBF)": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=1),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "RANSAC": RANSACRegressor(random_state=42),
        "GaussianProcess": GaussianProcessRegressor(kernel=1.0 * RBF(length_scale=1.0) + WhiteKernel(), n_restarts_optimizer=5),
        "Deming": "deming"
    }
    
    results = {}
    loo = LeaveOneOut()
    
    print("\nStarting LOOCV Tournament...")
    
    for name, model in models.items():
        y_true_all = []
        y_pred_all = []
        
        # Standardization (Important for SVM/Poly)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if name == "Deming":
            # Deming only works for 1D X (MP_ROM vs GT_ROM) usually.
            # We will run it on univariate MP_ROM only.
            slope, intercept = deming_regression(df_clean['mp_rom'], df_clean['gt_rom'])
            y_pred = df_clean['mp_rom'] * slope + intercept
            y_true_all = df_clean['gt_rom']
            y_pred_all = y_pred
            
        else:
            for train_index, test_index in loo.split(X_scaled):
                X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                if isinstance(model, str):
                    if model == "poly2":
                        poly = PolynomialFeatures(degree=2)
                        X_train_poly = poly.fit_transform(X_train)
                        X_test_poly = poly.fit_transform(X_test) # Use fit_transform on test? No, transform! 
                        # Wait, standardized X is already passed.
                        # Correct logic:
                        X_test_poly = poly.transform(X_test)
                        m = LinearRegression()
                        m.fit(X_train_poly, y_train)
                        pred = m.predict(X_test_poly)
                    else: # Deming handled above
                        continue
                else:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                y_true_all.append(y_test[0])
                y_pred_all.append(pred[0])
        
        # Calc Metrics
        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        results[name] = {'R2': r2, 'MAE': mae}
        
    # Report
    print("\n=== Model Tournament Results (LOOCV) ===")
    res_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
    print(res_df)
    
    res_df.to_csv(f"{OUTPUT_DIR}/regression_tournament_results.csv")
    
    # Plot Best Model vs Linear
    best_model_name = res_df.index[0]
    print(f"\nBest Model: {best_model_name}")
    
    # (Simplified plotting for visual check - just Scatter)
    # Re-predict all using best model trained on all
    # ... (Omitted for brevity, will rely on printed metrics)

if __name__ == "__main__":
    run_experiment()
