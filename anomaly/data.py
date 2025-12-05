"""Data loader and preprocessing for CMAPSS-like files.

Functions:
- load_cmapss(fd): read train/test/RUL for a given FD id (e.g. 'FD001')
- build_windows: helper to build sequence windows for recurrent models
"""
from pathlib import Path
import pandas as pd
import numpy as np

COL_NAMES = ["unit","time"] + [f"os{i}" for i in range(1,4)] + [f"s{i}" for i in range(1,22)]

def read_dataframe(path):
    # files are whitespace separated with no header
    return pd.read_csv(path, delim_whitespace=True, header=None, names=COL_NAMES)

def load_cmapss(data_dir, fd_id="FD001"):
    """Load train/test and RUL into dictionaries keyed by unit id.

    Returns: train_df, test_df, rul_series
    """
    base = Path(data_dir)
    train = read_dataframe(base / f"train_{fd_id}.txt")
    test = read_dataframe(base / f"test_{fd_id}.txt")
    rul = pd.read_csv(base / f"RUL_{fd_id}.txt", header=None).iloc[:,0]

    # add RUL to test: test units are ordered by unit id; RUL file correspond to final remaining cycles
    # compute max cycle per unit in test to create true remaining life per row when needed
    return train, test, rul

def add_rul(df):
    """Compute Remaining Useful Life (RUL) per row in df assuming it is a training trajectory.
    df must contain `unit` and `time` (cycle). Returns df with `RUL` column.
    """
    df = df.copy()
    max_cycle = df.groupby('unit')['time'].transform('max')
    df['RUL'] = max_cycle - df['time']
    return df

def compute_scaler(train_df, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in train_df.columns if c.startswith('os') or c.startswith('s')]
    mu = train_df[feature_cols].mean()
    sigma = train_df[feature_cols].std().replace(0,1.0)
    return mu, sigma, feature_cols

def scale_df(df, mu, sigma, feature_cols):
    out = df.copy()
    out[feature_cols] = (out[feature_cols] - mu) / sigma
    return out

def build_sequence_windows(df, unit_id, feature_cols, window_size=30, step=1):
    """Return array of shape (n_windows, window_size, n_features) for a given unit time series."""
    u = df[df['unit']==unit_id].sort_values('time')
    arr = u[feature_cols].values
    if arr.shape[0] < window_size:
        return np.empty((0, window_size, arr.shape[1]))
    windows = []
    for i in range(0, arr.shape[0]-window_size+1, step):
        windows.append(arr[i:i+window_size])
    return np.array(windows)
