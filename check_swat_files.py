
import pandas as pd
import os

files = ['swat2.csv', 'swat_train2.csv', 'swat_train.csv']
base_path = r'f:\GDN\GDN\data\SWaT'

for file in files:
    path = os.path.join(base_path, file)
    if os.path.exists(path):
        print(f"--- Analyzing {file} ---")
        try:
            # Read only the last column to be fast, and header
            df = pd.read_csv(path, nrows=5)
            cols = df.columns.tolist()
            if 'Normal/Attack' in cols:
                # Read only last column
                df_labels = pd.read_csv(path, usecols=['Normal/Attack'])
                print(df_labels['Normal/Attack'].value_counts())
            else:
                print("Column 'Normal/Attack' not found.")
        except Exception as e:
            print(f"Error reading {file}: {e}")
