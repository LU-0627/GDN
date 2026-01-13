# process_swat.py
# Convert SWaT CSV (Timestamp + sensors + Normal/Attack) to GDN-like train/test CSV with 'attack' label
# Output: train.csv/test.csv includes Timestamp + normalized features + attack(0/1)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TRAIN_PATH = "../data/swat/swat_train.csv"
TEST_PATH  = "../data/swat/swat_test.csv"

TIME_COL  = "Timestamp"
LABEL_COL = "Normal/Attack"   # SWaT original label column
OUT_LABEL = "attack"          # output label name

DOWNSAMPLE_LEN = 10
TRAIN_TRIM_START = 2160       # set to 0 to disable

def norm(train_arr: np.ndarray, test_arr: np.ndarray):
    """MinMax normalize to [0,1] using train statistics."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_arr)
    return scaler.transform(train_arr), scaler.transform(test_arr)

def downsample_features(data: np.ndarray, down_len: int):
    """Downsample features by median over window."""
    np_data = np.asarray(data, dtype=float)
    orig_len, col_num = np_data.shape
    down_time_len = orig_len // down_len

    np_data = np_data[: down_time_len * down_len]
    d_data = np_data.reshape(down_time_len, down_len, col_num)
    d_data = np.median(d_data, axis=1)
    return d_data, down_time_len

def downsample_labels(labels: np.ndarray, down_len: int, down_time_len: int):
    """Downsample labels by max over window (any anomaly -> 1)."""
    np_labels = np.asarray(labels, dtype=float)
    np_labels = np_labels[: down_time_len * down_len]
    d_labels = np_labels.reshape(down_time_len, down_len).max(axis=1)
    return d_labels

def downsample_timestamps(ts: pd.Series, down_len: int, down_time_len: int):
    """
    Keep Timestamp column after downsampling.
    Policy: use the first timestamp of each window (index 0 of the window).
    """
    ts = ts.astype(str).to_numpy()
    ts = ts[: down_time_len * down_len]
    ts = ts.reshape(down_time_len, down_len)[:, 0]
    return ts

def map_label(series: pd.Series) -> pd.Series:
    """
    Robust label mapping:
      - lowercase
      - remove BOM
      - remove any non-letter chars (space, invisible chars, etc.)
    So 'a ttack' -> 'attack'
    """
    s = series.astype(str)
    s = s.str.replace("\ufeff", "", regex=False).str.strip().str.lower()
    s = s.str.replace(r"[^a-z]", "", regex=True)  # keep only letters

    mapping = {"normal": 0, "attack": 1}
    y = s.map(mapping)

    if y.isna().any():
        bad_clean = s[y.isna()].unique()
        bad_raw = series[y.isna()].astype(str).unique()
        raise ValueError(
            f"Unmapped label values in {LABEL_COL}: cleaned={bad_clean}, raw={list(bad_raw)[:20]}"
        )

    return y.astype(int)

def load_and_prepare(path: str) -> pd.DataFrame:
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
            raise KeyError(f"Missing '{TIME_COL}' in {name}. columns={list(df.columns)}")

    # 3) Extract Timestamp (keep it)
    ts_train = train[TIME_COL]
    ts_test  = test[TIME_COL]

    # 4) Extract and map labels
    y_train = map_label(train[LABEL_COL]).to_numpy()
    y_test  = map_label(test[LABEL_COL]).to_numpy()

    # 5) Extract features (drop time + label)
    X_train = train.drop(columns=[TIME_COL, LABEL_COL])
    X_test  = test.drop(columns=[TIME_COL, LABEL_COL])

    # 6) Force numeric for features; coerce errors to NaN then fill
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test  = X_test.apply(pd.to_numeric, errors="coerce")

    X_train = X_train.fillna(X_train.mean()).fillna(0)
    X_test  = X_test.fillna(X_test.mean()).fillna(0)

    # 7) Normalize using train stats
    x_train, x_test = norm(X_train.to_numpy(), X_test.to_numpy())
    X_train = pd.DataFrame(x_train, columns=X_train.columns)
    X_test  = pd.DataFrame(x_test, columns=X_test.columns)

    # 8) Downsample (features + labels + timestamps must align)
    d_train_x, train_down_len = downsample_features(X_train.to_numpy(), DOWNSAMPLE_LEN)
    d_test_x,  test_down_len  = downsample_features(X_test.to_numpy(), DOWNSAMPLE_LEN)

    d_train_y = downsample_labels(y_train, DOWNSAMPLE_LEN, train_down_len)
    d_test_y  = downsample_labels(y_test,  DOWNSAMPLE_LEN, test_down_len)

    d_train_ts = downsample_timestamps(ts_train, DOWNSAMPLE_LEN, train_down_len)
    d_test_ts  = downsample_timestamps(ts_test,  DOWNSAMPLE_LEN, test_down_len)

    # 9) Build output frames (Timestamp first column)
    train_df = pd.DataFrame(d_train_x, columns=X_train.columns)
    test_df  = pd.DataFrame(d_test_x,  columns=X_test.columns)

    train_df.insert(0, TIME_COL, d_train_ts)
    test_df.insert(0, TIME_COL, d_test_ts)

    train_df[OUT_LABEL] = d_train_y.astype(int)
    test_df[OUT_LABEL]  = d_test_y.astype(int)

    # 10) Optional trim (keep original behavior; affects Timestamp too)
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

    # list.txt should list feature names only (not Timestamp, not label)
    with open("./list.txt", "w", encoding="utf-8") as f:
        for col in X_train.columns:
            f.write(col + "\n")

    print("[OK] Saved: ./train.csv, ./test.csv, ./list.txt")
    print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
    print("Train label counts:\n", train_df[OUT_LABEL].value_counts().sort_index())
    print("Test label counts:\n", test_df[OUT_LABEL].value_counts().sort_index())

if __name__ == "__main__":
    main()
