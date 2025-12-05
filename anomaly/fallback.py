"""Fallback wrapper for dense autoencoder training/prediction.
If TensorFlow is available in `anomaly.models`, use it; otherwise use a scikit-learn MLPRegressor autoencoder.
This module is intended to be imported from notebooks to avoid hard RuntimeErrors when TF is not installed.
"""
import numpy as np

try:
    from anomaly import models as _models
except Exception:
    _models = None


def dense_autoencoder_train_predict_wrapper(train_df, df, feature_cols, latent_dim=8, epochs=30, batch_size=256):
    """Call the dense AE in anomaly.models if available and working; otherwise use an sklearn MLP fallback.

    Returns: (rec_err, feature_err, model)
    """
    # Try to use the module implementation first
    if _models is not None:
        try:
            return _models.dense_autoencoder_train_predict(train_df, df, feature_cols, latent_dim=latent_dim, epochs=epochs, batch_size=batch_size)
        except RuntimeError:
            # fall through to sklearn fallback
            pass

    # sklearn fallback
    from sklearn.neural_network import MLPRegressor
    Xtr = train_df[feature_cols].values
    X = df[feature_cols].values
    hidden = (max(32, Xtr.shape[1]*2), 64, max(8, latent_dim), 64, max(32, Xtr.shape[1]*2))
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                       max_iter=max(1, int(epochs)), batch_size=max(16, int(batch_size)),
                       random_state=42, verbose=False)
    mlp.fit(Xtr, Xtr)
    recon_tr = mlp.predict(Xtr)
    recon = mlp.predict(X)
    rec_err = np.mean((X - recon)**2, axis=1)
    feature_err = (X - recon)
    return rec_err, feature_err, mlp


def lstm_autoencoder_train_predict_wrapper(train_windows, windows, epochs=20, batch_size=64):
    """Wrapper for LSTM-AE. If TF is available, call the model. Otherwise flatten windows and use an MLPRegressor to reconstruct flattened sequences.

    Returns: (rec_err, feature_err, model)
    """
    if _models is not None:
        try:
            return _models.lstm_autoencoder_train_predict(train_windows, windows, epochs=epochs, batch_size=batch_size)
        except RuntimeError:
            pass

    # Fallback: flatten windows (w * f) and use MLP to reconstruct
    from sklearn.neural_network import MLPRegressor
    if train_windows.size == 0 or windows.size == 0:
        return np.array([]), np.array([]), None
    n_tr, w, f = train_windows.shape
    Xtr = train_windows.reshape((n_tr, w * f))
    X = windows.reshape((windows.shape[0], w * f))
    hidden = (max(128, w * f // 2), max(64, f), max(32, f))
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation='relu', solver='adam',
                       max_iter=max(1, int(epochs)), batch_size=max(8, int(batch_size)), random_state=42, verbose=False)
    mlp.fit(Xtr, Xtr)
    recon = mlp.predict(X)
    rec_err = np.mean((X - recon)**2, axis=1)
    feature_err = (X - recon).reshape((X.shape[0], w, f))
    return rec_err, feature_err, mlp


def tcn_vae_train_predict_wrapper(train_windows, windows, epochs=20, batch_size=64):
    """Wrapper for TCN-VAE. If TF is available, call original. Otherwise use same flattened-MLP fallback as LSTM wrapper.

    Returns: (rec_err, feature_err, model)
    """
    if _models is not None:
        try:
            return _models.tcn_vae_train_predict(train_windows, windows, epochs=epochs, batch_size=batch_size)
        except RuntimeError:
            pass

    # Reuse LSTM fallback behavior
    return lstm_autoencoder_train_predict_wrapper(train_windows, windows, epochs=epochs, batch_size=batch_size)
