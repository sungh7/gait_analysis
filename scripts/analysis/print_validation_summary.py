import json
import numpy as np

with open('/data/gait/dtw_validation_results.json', 'r') as f:
    results = json.load(f)

print("Validation Summary:")
print(f"{'Joint':<10} | {'RMSE':<10} | {'Corr':<10} | {'ICC':<10}")
print("-" * 46)

for joint in ['hip', 'knee', 'ankle']:
    data = results['summary'][joint]['after_dtw']
    print(f"{joint.capitalize():<10} | {data['rmse_mean']:.2f} ± {data['rmse_std']:.2f} | {data['correlation_mean']:.2f} ± {data['correlation_std']:.2f} | {data['icc_mean']:.2f} ± {data['icc_std']:.2f}")
