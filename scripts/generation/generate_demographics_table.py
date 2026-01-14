import pandas as pd
import numpy as np

def generate_demographics_table():
    df = pd.read_csv('/data/gait/body_measurements.csv')
    
    # Select relevant columns
    cols = ['subject_id', 'age', 'height_cm', 'weight_kg', 'leg_length_cm']
    df_clean = df[cols].copy()
    
    # Calculate stats
    stats = df_clean[['age', 'height_cm', 'weight_kg', 'leg_length_cm']].agg(['mean', 'std'])
    
    # Format table
    print("| Subject ID | Age (years) | Height (cm) | Weight (kg) | Leg Length (cm) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for _, row in df_clean.iterrows():
        print(f"| {int(row['subject_id'])} | {int(row['age'])} | {row['height_cm']} | {row['weight_kg']} | {row['leg_length_cm']:.2f} |")
        
    print(f"| **Mean ± SD** | **{stats.loc['mean', 'age']:.1f} ± {stats.loc['std', 'age']:.1f}** | **{stats.loc['mean', 'height_cm']:.1f} ± {stats.loc['std', 'height_cm']:.1f}** | **{stats.loc['mean', 'weight_kg']:.1f} ± {stats.loc['std', 'weight_kg']:.1f}** | **{stats.loc['mean', 'leg_length_cm']:.1f} ± {stats.loc['std', 'leg_length_cm']:.1f}** |")

if __name__ == "__main__":
    generate_demographics_table()
