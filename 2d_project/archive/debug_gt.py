import pandas as pd
import numpy as np

csv_path = "/data/gait/data/1/excel/S1_01_edited.csv"

print(f"Loading {csv_path}...")
try:
    # Try header=3 (row 4)
    df = pd.read_csv(csv_path, header=3)
    print("Columns (Header=3):")
    print(df.columns.tolist()[:10])
    
    if 'r.kn.angle' in df.columns:
        print("\n'r.kn.angle' found!")
        col = df['r.kn.angle']
        print("Type:", type(col))
        if isinstance(col, pd.DataFrame):
            print("It is a DataFrame (duplicated columns). Head:")
            print(col.head())
            # Check values
            val = col.iloc[0, 0]
            print(f"Value [0,0]: {val} (Type: {type(val)})")
        else:
            print("It is a Series. Head:")
            print(col.head())
            val = col.iloc[0]
            print(f"Value [0]: {val} (Type: {type(val)})")
            
    else:
        print("\n'r.kn.angle' NOT found with header=3.")
        # Print first few rows to debug header
        print("\nFirst 5 rows raw:")
        df_raw = pd.read_csv(csv_path, header=None, nrows=5)
        print(df_raw)

except Exception as e:
    print(f"Error: {e}")
