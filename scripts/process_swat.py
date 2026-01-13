# process_swat.py
# Convert SWaT CSV (Timestamp + sensors + Normal/Attack) to GDN-like train/test CSV with 'attack' label
# Author: you
# Notes:
# - Assumes the input has a 'Timestamp' column (string) and 'Normal/Attack' label column (Normal/Attack).
# - Output train.csv/test.csv will contain normalized features + 'attack' (0/1).
# - Downsample by median over windows; label is max over window (any anomaly -> 1).

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TRAIN_PATH = "../data/swat/swat_train.csv"
TEST_PATH  = "../data/swat/swat_test.csv"

TIME_COL  = "Timestamp"
LABEL_COL = "Normal/Attack"   # SWaT original label column
OUT_LABEL = "attack"          # output label name (consistent with many TSAD codebases)

DOWNSAMPLE_LEN = 10
TRAIN_TRIM_START = 2160        # keep your original trimming logic; set to 0 to disable

def norm(train_arr: np.ndarray, test_arr: np.ndarray):
    """MinMax normalize to [0,1] using train statistics."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_arr)
    return scaler.transform(train_arr), scaler.transform(test_arr)

def downsample(data: np.ndarray, labels: np.ndarray, down_len: int):
    """
    Downsample multivariate time series:
      - features: median within each window
      - labels: max within each window (any 1 -> 1)
    """
    np_data = np.asarray(data, dtype=float)
    np_labels = np.asarray(labels, dtype=float)

    orig_len, col_num = np_data.shape
    down_time_len = orig_len // down_len

    # truncate to multiple of down_len
    np_data = np_data[: down_time_len * down_len]
    np_labels = np_labels[: down_time_len * down_len]

    # features: (T, F) -> (T//k, k, F) -> median over k
    d_data = np_data.reshape(down_time_len, down_len, col_num)
    d_data = np.median(d_data, axis=1)

    # labels: (T,) -> (T//k, k) -> max over k
    d_labels = np_labels.reshape(down_time_len, down_len).max(axis=1)

    return d_data, d_labels

def map_label(series: pd.Series) -> pd.Series:
    s_raw = series.astype(str)

    # 1) 去 BOM + 去首尾空格 + 小写
    s = s_raw.str.replace("\ufeff", "", regex=False).str.strip().str.lower()

    # 2) 只保留字母 a-z（删除所有空格/不可见字符/标点/斜杠等）
    s = s.str.replace(r"[^a-z]", "", regex=True)

    mapping = {"normal": 0, "attack": 1}
    y = s.map(mapping)

    if y.isna().any():
        bad_clean = s[y.isna()].unique()
        bad_raw = s_raw[y.isna()].unique()
        raise ValueError(
            f"Unmapped label values in {LABEL_COL}: cleaned={bad_clean}, raw={list(bad_raw)[:20]}"
        )

    return y.astype(int)



def load_and_prepare(path: str) -> pd.DataFrame:
    """Load CSV and strip column names."""
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    return df

def main():
    # 1) Load
    train = load_and_prepare(TRAIN_PATH)
    test  = load_and_prepare(TEST_PATH)

    # 2) Check required columns
    for name, df in [("train", train), ("test", test)]:
        if LABEL_COL not in df.columns:
            raise KeyError(f"Missing '{LABEL_COL}' in {name}. columns={list(df.columns)}")
        if TIME_COL not in df.columns:
            # not fatal, but SWaT usually has it
            print(f"[WARN] '{TIME_COL}' not found in {name}. Proceeding without dropping it.")

    # 3) Drop time column (do NOT include string time in features)
    if TIME_COL in train.columns:
        train = train.drop(columns=[TIME_COL])
    if TIME_COL in test.columns:
        test = test.drop(columns=[TIME_COL])

    # 4) Extract and map labels
    y_train = map_label(train[LABEL_COL]).to_numpy()
    print("Unique raw labels in test:", test[LABEL_COL].astype(str).unique()[:20])

    y_test  = map_label(test[LABEL_COL]).to_numpy()

    # 5) Extract features
    X_train = train.drop(columns=[LABEL_COL])
    X_test  = test.drop(columns=[LABEL_COL])

    # 6) Force numeric for features; coerce errors to NaN then fill
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test  = X_test.apply(pd.to_numeric, errors="coerce")

    X_train = X_train.fillna(X_train.mean()).fillna(0)
    X_test  = X_test.fillna(X_test.mean()).fillna(0)

    # 7) Normalize using train stats
    x_train, x_test = norm(X_train.to_numpy(), X_test.to_numpy())
    X_train = pd.DataFrame(x_train, columns=X_train.columns)
    X_test  = pd.DataFrame(x_test, columns=X_test.columns)

    # 8) Downsample
    d_train_x, d_train_y = downsample(X_train.to_numpy(), y_train, DOWNSAMPLE_LEN)
    d_test_x, d_test_y   = downsample(X_test.to_numpy(), y_test, DOWNSAMPLE_LEN)

    # 9) Build output frames
    train_df = pd.DataFrame(d_train_x, columns=X_train.columns)
    test_df  = pd.DataFrame(d_test_x, columns=X_test.columns)

    train_df[OUT_LABEL] = d_train_y.astype(int)
    test_df[OUT_LABEL]  = d_test_y.astype(int)

    # 10) Optional trim (keep original behavior)
    if TRAIN_TRIM_START and TRAIN_TRIM_START > 0:
        if TRAIN_TRIM_START < len(train_df):
            train_df = train_df.iloc[TRAIN_TRIM_START:].reset_index(drop=True)
        else:
            raise ValueError(
                f"TRAIN_TRIM_START={TRAIN_TRIM_START} >= len(train_df)={len(train_df)}; trimming would empty the set."
            )

    # 11) Save
    train_df.to_csv("./train.csv", index=False)
    test_df.to_csv("./test.csv", index=False)

    with open("./list.txt", "w", encoding="utf-8") as f:
        for col in X_train.columns:
            f.write(col + "\n")

    print("[OK] Saved: ./train.csv, ./test.csv, ./list.txt")
    print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
    print("Train label counts:\n", train_df[OUT_LABEL].value_counts().sort_index())
    print("Test label counts:\n", test_df[OUT_LABEL].value_counts().sort_index())

if __name__ == "__main__":
    main()
