"""Train and evaluate the requested anomaly detection methods.

This script provides orchestration to:
- load data
- train models on FD001 (or chosen dataset)
- compute anomaly scores on test sequences
- determine detection point per unit using thresholding on training normal scores
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from .data import load_cmapss, add_rul, compute_scaler, scale_df, build_sequence_windows
from .models import (
    z_score_scores, pca_reconstruction_scores, isolation_forest_score, oneclass_svm_score,
    dense_autoencoder_train_predict, lstm_autoencoder_train_predict, tcn_vae_train_predict
)

def threshold_from_train(scores, factor=3.0):
    mu = np.mean(scores)
    sigma = np.std(scores)
    return mu + factor*sigma

def detect_point(scores, threshold, min_consec=1):
    # returns first index (1-based cycle) where score>threshold and remains for min_consec
    over = scores > threshold
    for i in range(len(over)):
        if over[i:i+min_consec].all():
            return i+1  # 1-based
    return None

def evaluate_methods(data_dir, fd_id="FD001", window_size=30):
    data_dir = Path(data_dir)
    train_df, test_df, rul = load_cmapss(data_dir, fd_id)
    train_df = add_rul(train_df)
    # features
    feature_cols = [c for c in train_df.columns if c.startswith('os') or c.startswith('s')]

    mu, sigma, feature_cols = compute_scaler(train_df, feature_cols)
    train_scaled = scale_df(train_df, mu, sigma, feature_cols)
    test_scaled = scale_df(test_df, mu, sigma, feature_cols)

    results = {}

    # Z-scores
    z_scores, z_per_feature = z_score_scores(train_scaled, feature_cols)
    # for thresholding we can use train set baseline
    z_thr = threshold_from_train(z_scores)
    results['z'] = {'train_scores': z_scores, 'threshold': z_thr}

    # PCA
    pca_train_err, _ = pca_reconstruction_scores(train_scaled, train_scaled, feature_cols)
    pca_thr = threshold_from_train(pca_train_err)
    results['pca'] = {'train_scores': pca_train_err, 'threshold': pca_thr}

    # IsolationForest
    iso_train, _ = isolation_forest_score(train_scaled, train_scaled, feature_cols)
    iso_thr = threshold_from_train(iso_train)
    results['isolation'] = {'train_scores': iso_train, 'threshold': iso_thr}

    # OneClassSVM
    ocsvm_train, _ = oneclass_svm_score(train_scaled, train_scaled, feature_cols)
    ocsvm_thr = threshold_from_train(ocsvm_train)
    results['oneclass'] = {'train_scores': ocsvm_train, 'threshold': ocsvm_thr}

    # Dense Autoencoder (per-row)
    ae_train_err, _, ae_model = dense_autoencoder_train_predict(train_scaled, train_scaled, feature_cols, epochs=10)
    ae_thr = threshold_from_train(ae_train_err)
    results['ae'] = {'train_scores': ae_train_err, 'threshold': ae_thr, 'model': ae_model}

    # For sequence models build windows from train units
    # create windows by concatenating windows from all train units
    train_windows = []
    for unit in train_scaled['unit'].unique():
        w = build_sequence_windows(train_scaled, unit, feature_cols, window_size=window_size)
        if w.size:
            train_windows.append(w)
    if len(train_windows):
        train_windows = np.vstack(train_windows)
    else:
        train_windows = np.empty((0, window_size, len(feature_cols)))

    # Build sliding windows for test per-unit later when scoring

    # LSTM-AE
    if train_windows.shape[0] > 0:
        lstm_train_err, _, lstm_model = lstm_autoencoder_train_predict(train_windows, train_windows, epochs=8)
        lstm_thr = threshold_from_train(lstm_train_err)
        results['lstm'] = {'train_scores': lstm_train_err, 'threshold': lstm_thr, 'model': lstm_model}
    else:
        results['lstm'] = {'train_scores': np.array([]), 'threshold': np.inf}

    # TCN-VAE
    if train_windows.shape[0] > 0:
        tcn_train_err, _, tcn_model = tcn_vae_train_predict(train_windows, train_windows, epochs=8)
        tcn_thr = threshold_from_train(tcn_train_err)
        results['tcn'] = {'train_scores': tcn_train_err, 'threshold': tcn_thr, 'model': tcn_model}
    else:
        results['tcn'] = {'train_scores': np.array([]), 'threshold': np.inf}

    # Now score test units and determine detection points
    units = sorted(test_scaled['unit'].unique())
    per_unit_detection = {m: {} for m in results.keys()}

    # helper to score per method per row or per window
    # row-based methods: z, pca, isolation, oneclass, ae
    # sequence-based: lstm, tcn

    # precompute row-wise scores for test
    # Z
    z_test_scores, z_pf = z_score_scores(train_scaled, feature_cols)
    # For PCA, IsolationForest, OneClass, AE we need to apply to test rows
    # PCA
    from .models import pca_reconstruction_scores as pca_fn, isolation_forest_score as iso_fn, oneclass_svm_score as oc_fn
    pca_err_test, pca_feat_err = pca_fn(train_scaled, test_scaled, feature_cols)
    iso_err_test, _ = iso_fn(train_scaled, test_scaled, feature_cols)
    oc_err_test, _ = oc_fn(train_scaled, test_scaled, feature_cols)
    ae_err_test, ae_feat_err, _ = dense_autoencoder_train_predict(train_scaled, test_scaled, feature_cols, epochs=1)

    # For each unit determine detection cycle
    for unit in units:
        unit_rows = test_scaled[test_scaled['unit']==unit].sort_values('time')
        idx = unit_rows.index
        # row-wise arrays
        pca_scores_u = pca_err_test[idx]
        iso_scores_u = iso_err_test[idx]
        oc_scores_u = oc_err_test[idx]
        ae_scores_u = ae_err_test[idx]

        # sequence windows for this unit
        win = build_sequence_windows(test_scaled, unit, feature_cols, window_size=window_size)
        if win.shape[0] > 0 and 'lstm' in results and results['lstm']['threshold']<np.inf:
            lstm_err_u, _, _ = lstm_autoencoder_train_predict(train_windows, win, epochs=1)
        else:
            lstm_err_u = np.array([])
        if win.shape[0] > 0 and 'tcn' in results and results['tcn']['threshold']<np.inf:
            tcn_err_u, _, _ = tcn_vae_train_predict(train_windows, win, epochs=1)
        else:
            tcn_err_u = np.array([])

        # detect points using thresholds
        detectors = {
            'pca': (pca_scores_u, results['pca']['threshold']),
            'isolation': (iso_scores_u, results['isolation']['threshold']),
            'oneclass': (oc_scores_u, results['oneclass']['threshold']),
            'ae': (ae_scores_u, results['ae']['threshold']),
        }
        if lstm_err_u.size:
            detectors['lstm'] = (lstm_err_u, results['lstm']['threshold'])
        if tcn_err_u.size:
            detectors['tcn'] = (tcn_err_u, results['tcn']['threshold'])

        for m, (scores, thr) in detectors.items():
            if scores.size == 0:
                per_unit_detection[m][unit] = None
                continue
            det = detect_point(scores, thr, min_consec=1)
            # detection cycle mapping: for sequence methods det is window index -> map to last cycle of that window
            if m in ('lstm','tcn') and det is not None:
                # map window index (1-based) to cycle index = window_start + window_size - 1
                cycle = det + window_size - 1
            else:
                cycle = det
            per_unit_detection[m][unit] = cycle

    return results, per_unit_detection, feature_cols

def plot_scores_over_time(test_df, unit, scores_dict, thresholds, outpath=None):
    df = test_df[test_df['unit']==unit].sort_values('time')
    plt.figure(figsize=(10,5))
    for name, scores in scores_dict.items():
        plt.plot(df['time'].values[:len(scores)], scores, label=name)
        if name in thresholds:
            plt.hlines(thresholds[name], df['time'].min(), df['time'].max(), linestyles='--')
    plt.xlabel('cycle')
    plt.ylabel('anomaly score')
    plt.title(f'Unit {unit} anomaly scores')
    plt.legend()
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()
