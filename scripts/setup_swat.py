import pandas as pd
import numpy as np
import os

def setup_swat():
    data_dir = r'f:\GDN\GDN\data\SWaT'
    train_file = os.path.join(data_dir, 'swat_train2.csv')
    test_file = os.path.join(data_dir, 'swat2.csv')
    
    output_train = os.path.join(data_dir, 'train.csv')
    output_test = os.path.join(data_dir, 'test.csv')
    output_list = os.path.join(data_dir, 'list.txt')
    
    print("Processing Training Data...")
    # swa_train2.csv likely has no index column, first column is FIT101
    # We need to read it, add an index, and rename Normal/Attack
    df_train = pd.read_csv(train_file)
    
    # Rename Normal/Attack to attack
    if 'Normal/Attack' in df_train.columns:
        df_train.rename(columns={'Normal/Attack': 'attack'}, inplace=True)
        
    # Check if 'attack' column exists now
    if 'attack' not in df_train.columns:
        print("Warning: 'attack' column not found in train file via 'Normal/Attack' rename.")
    
    # Add dummy index since main.py expects index_col=0
    # We insert it at position 0
    df_train.insert(0, 'timestamp', range(len(df_train)))
    
    print(f"Saving train.csv with shape {df_train.shape}...")
    df_train.to_csv(output_train, index=False)
    
    # Generate list.txt from feature columns
    # Features are all columns except 'timestamp' (index) and 'attack'
    features = [c for c in df_train.columns if c not in ['timestamp', 'attack']]
    print(f"Found {len(features)} features.")
    
    with open(output_list, 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    print("Saved list.txt")

    print("Processing Test Data...")
    # swat2.csv has Timestamp as first column, which is good for index_col=0
    # But checking to be sure.
    # Also rename Normal/Attack to attack
    # Note: swat2.csv might have extra spaces in header based on previous cat output?
    # " Timestamp,FIT101,..." vs "FIT101,..."
    
    # Let's read with pandas default
    df_test = pd.read_csv(test_file)
    
    # Cleanup column names (strip whitespace)
    df_test.columns = df_test.columns.str.strip()
    
    # Rename Normal/Attack to attack
    if 'Normal/Attack' in df_test.columns:
        df_test.rename(columns={'Normal/Attack': 'attack'}, inplace=True)
        
    # Ensure there is an 'attack' column
    if 'attack' not in df_test.columns:
        print("CRITICAL: 'attack' column missing in test file.")

    # Check if we need to add index
    # Based on observation, swat2.csv starts with FIT101, so it needs an index
    if 'Timestamp' not in df_test.columns and 'timestamp' not in df_test.columns:
         print("Adding dummy timestamp to test data...")
         df_test.insert(0, 'timestamp', range(len(df_test)))
    
    # Check if first column is timestamp-like or if we need to ensure index logic
    # main.py does: pd.read_csv(..., index_col=0)
    # df_test columns[0] should be the index.
    print(f"Test columns: {df_test.columns.tolist()[:5]} ...")
    
    print(f"Saving test.csv with shape {df_test.shape}...")
    df_test.to_csv(output_test, index=False)
    
    print("SWaT Setup Complete.")

if __name__ == "__main__":
    setup_swat()
